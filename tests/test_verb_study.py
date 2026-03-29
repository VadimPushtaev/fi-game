from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fi_game.verb_study import (
    RowRange,
    VerbStudyLexiconSlice,
    VerbStudyOutputStore,
    VerbStudyRunner,
    deduplicate_study_verbs,
)


def write_yaml(path: Path, data: list[dict[str, object]]) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_lexicon_slice_uses_independent_ranges_for_verbs_and_vocabulary(tmp_path: Path) -> None:
    lexicon_path = tmp_path / "fi_50k.yaml"
    write_yaml(
        lexicon_path,
        [
            {"word": "olen", "pos": "verb", "english": "am"},
            {"word": "olen", "pos": "auxiliary", "english": "am"},
            {"word": "oli", "pos": "verb", "english": "was"},
            {"word": "voi", "pos": "verb", "english": "can"},
            {"word": "voi", "pos": "noun", "english": "butter"},
            {"word": "olet", "pos": "verb", "english": "are"},
            {"word": "poika", "pos": "noun", "english": "boy"},
            {"word": "pieni", "pos": "adjective", "english": "small"},
        ],
    )

    lexicon_slice = VerbStudyLexiconSlice(
        lexicon_path=lexicon_path,
        verb_range=RowRange(1, 5),
        vocabulary_range=RowRange(5, 8),
    )

    assert [row["word"] for row in lexicon_slice.load_verb_rows()] == [
        "olen",
        "olen",
        "oli",
        "voi",
        "voi",
    ]
    assert [row["word"] for row in lexicon_slice.load_vocabulary_rows()] == [
        "voi",
        "olet",
        "poika",
        "pieni",
    ]
    assert lexicon_slice.eligible_surface_verbs() == ["olen", "oli", "voi"]
    assert lexicon_slice.allowed_words_by_pos() == {
        "noun": ["voi", "poika"],
        "verb": ["olet"],
        "adjective": ["pieni"],
    }


def test_deduplicate_study_verbs_preserves_first_seen_lemma_order() -> None:
    study_verbs = deduplicate_study_verbs(
        {
            "olen": "olla",
            "oli": "olla",
            "voin": "voida",
            "voi": "voida",
            "tulee": "tulla",
        }
    )

    assert [study_verb.lemma for study_verb in study_verbs] == ["olla", "voida", "tulla"]
    assert study_verbs[0].source_forms == ("olen", "oli")
    assert study_verbs[1].source_forms == ("voin", "voi")


def test_output_store_replaces_partial_records_with_complete_set(tmp_path: Path) -> None:
    output_path = tmp_path / "verb_study.yaml"
    store = VerbStudyOutputStore(output_path)
    generation = {
        "lexicon_path": "data/fi_50k.yaml",
        "variant": "baseline",
        "verb_rows": {"start": 1, "end": 20},
        "vocabulary_rows": {"start": 101, "end": 160},
    }

    partial_records = [
        {
            "verb": "olla",
            "verb_form": "1sg",
            "sentence_fi": "Minä olen täällä.",
            "sentence_fi_masked": "Minä %%%% täällä.",
            "answer_fi": "olen",
            "sentence_en": "I am here.",
            "sentence_en_masked": "I %%%% here.",
            "generation": generation,
        }
    ]
    write_yaml(output_path, partial_records)

    complete_records = []
    for verb_form in ("1sg", "2sg", "3sg", "1pl", "2pl", "3pl", "imperative", "negative"):
        for index in range(2):
            answer = f"olla_{verb_form}_{index}"
            complete_records.append(
                {
                    "verb": "olla",
                    "verb_form": verb_form,
                    "sentence_fi": f"Lause {verb_form} {index} {answer}.",
                    "sentence_fi_masked": f"Lause {verb_form} {index} %%%%.",
                    "answer_fi": answer,
                    "sentence_en": f"Sentence {verb_form} {index} answer.",
                    "sentence_en_masked": f"Sentence {verb_form} {index} %%%%.",
                }
            )

    assert not store.is_complete("olla")
    store.replace_records_for_lemma("olla", complete_records, generation_metadata=generation)
    assert store.is_complete("olla")
    assert len(store.load_records()) == 16
    assert store.load_records()[0]["generation"] == {
        "lexicon_path": "data/fi_50k.yaml",
        "variant": "baseline",
        "verb_rows": {"start": "1", "end": "20"},
        "vocabulary_rows": {"start": "101", "end": "160"},
    }


def test_output_store_validates_masking_rules(tmp_path: Path) -> None:
    output_path = tmp_path / "verb_study.yaml"
    store = VerbStudyOutputStore(output_path)

    bad_records = []
    for verb_form in ("1sg", "2sg", "3sg", "1pl", "2pl", "3pl", "imperative", "negative"):
        for index in range(2):
            bad_records.append(
                {
                    "verb": "puhua",
                    "verb_form": verb_form,
                    "sentence_fi": f"Minä puhun {verb_form} {index}.",
                    "sentence_fi_masked": f"Minä puhun {verb_form} {index}.",
                    "answer_fi": "puhun",
                    "sentence_en": f"I speak {verb_form} {index}.",
                    "sentence_en_masked": f"I %%%% {verb_form} {index}.",
                }
            )

    with pytest.raises(ValueError, match="masked Finnish sentence"):
        store.replace_records_for_lemma("puhua", bad_records)


def test_output_store_allows_ai_provided_masking_without_mechanical_replacement(tmp_path: Path) -> None:
    output_path = tmp_path / "verb_study.yaml"
    store = VerbStudyOutputStore(output_path)

    records = []
    for verb_form in ("1sg", "2sg", "3sg", "1pl", "2pl", "3pl", "imperative", "negative"):
        for index in range(2):
            records.append(
                {
                    "verb": "olla",
                    "verb_form": verb_form,
                    "sentence_fi": f"Minä olen taas {verb_form} {index}.",
                    "sentence_fi_masked": f"Minä %%%% taas {verb_form} {index}.",
                    "answer_fi": "on",
                    "sentence_en": f"I am again {verb_form} {index}.",
                    "sentence_en_masked": f"I %%%% again {verb_form} {index}.",
                }
            )

    store.replace_records_for_lemma(
        "olla",
        records,
        generation_metadata={
            "lexicon_path": "data/fi_50k.yaml",
            "variant": "baseline",
            "verb_rows": {"start": 1, "end": 5},
            "vocabulary_rows": {"start": 10, "end": 20},
        },
    )
    assert store.is_complete("olla")


def test_lexicon_slice_preserves_yaml_boolean_like_words(tmp_path: Path) -> None:
    lexicon_path = tmp_path / "fi_50k.yaml"
    lexicon_path.write_text(
        "- word: on\n"
        "  pos: verb\n"
        "  english: is\n"
        "- word: ei\n"
        "  pos: auxiliary\n"
        "  english: not\n",
        encoding="utf-8",
    )

    lexicon_slice = VerbStudyLexiconSlice(
        lexicon_path=lexicon_path,
        verb_range=RowRange(1, 2),
        vocabulary_range=RowRange(1, 2),
    )

    assert lexicon_slice.eligible_surface_verbs() == ["on"]


def test_row_range_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        RowRange(0, 3)

    with pytest.raises(ValueError, match=">= start"):
        RowRange(4, 3)


def test_output_store_scopes_completion_by_variant(tmp_path: Path) -> None:
    output_path = tmp_path / "verb_study.yaml"
    store = VerbStudyOutputStore(output_path)

    records = []
    for verb_form in ("1sg", "2sg", "3sg", "1pl", "2pl", "3pl", "imperative", "negative"):
        for index in range(2):
            answer = f"antaa_{verb_form}_{index}"
            records.append(
                {
                    "verb": "antaa",
                    "verb_form": verb_form,
                    "sentence_fi": f"Lause {verb_form} {index} {answer}.",
                    "sentence_fi_masked": f"Lause {verb_form} {index} %%%%.",
                    "answer_fi": answer,
                    "sentence_en": f"Sentence {verb_form} {index} answer.",
                    "sentence_en_masked": f"Sentence {verb_form} {index} %%%%.",
                }
            )

    store.replace_records_for_lemma(
        "antaa",
        records,
        generation_metadata={
            "lexicon_path": "data/fi_50k.yaml",
            "variant": "nouns_1_1000",
            "verb_rows": {"start": 1, "end": 200},
            "vocabulary_rows": {"start": 1, "end": 1000},
        },
        variant="nouns_1_1000",
    )

    assert store.is_complete("antaa", variant="nouns_1_1000")
    assert not store.is_complete("antaa", variant="nouns_1001_2000")


def test_runner_force_regenerates_even_when_variant_is_complete(tmp_path: Path) -> None:
    output_path = tmp_path / "verb_study.yaml"
    store = VerbStudyOutputStore(output_path)

    complete_records = []
    for verb_form in ("1sg", "2sg", "3sg", "1pl", "2pl", "3pl", "imperative", "negative"):
        for index in range(2):
            answer = f"antaa_{verb_form}_{index}"
            complete_records.append(
                {
                    "verb": "antaa",
                    "verb_form": verb_form,
                    "sentence_fi": f"Lause {verb_form} {index} {answer}.",
                    "sentence_fi_masked": f"Lause {verb_form} {index} %%%%.",
                    "answer_fi": answer,
                    "sentence_en": f"Sentence {verb_form} {index} answer.",
                    "sentence_en_masked": f"Sentence {verb_form} {index} %%%%.",
                }
            )

    store.replace_records_for_lemma(
        "antaa",
        complete_records,
        generation_metadata={
            "lexicon_path": "data/fi_50k.yaml",
            "variant": "baseline",
            "verb_rows": {"start": 1, "end": 200},
            "vocabulary_rows": {"start": 1, "end": 1000},
        },
        variant="baseline",
    )

    class StubLexiconSlice:
        verb_range = RowRange(1, 200)
        vocabulary_range = RowRange(1, 1000)

        def allowed_words_by_pos(self) -> dict[str, list[str]]:
            return {}

    class StubLemmaResolver:
        def resolve(self, surfaces, *, model=None):
            return {}

    class StubGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate_records_for_verb(self, **kwargs):
            self.calls += 1
            return complete_records

    runner = VerbStudyRunner(
        lexicon_slice=StubLexiconSlice(),
        lemma_resolver=StubLemmaResolver(),
        generator=StubGenerator(),
        output_store=store,
    )
    runner.build_study_verbs = lambda model=None: [type("StudyVerbStub", (), {"lemma": "antaa"})()]

    result = runner.generate_all(
        lexicon_path_for_prompt="data/fi_50k.yaml",
        output_yaml_path_for_prompt="data/out.yaml",
        variant="baseline",
        force=True,
    )

    assert result.generated_verbs == 1
    assert result.skipped_verbs == 0

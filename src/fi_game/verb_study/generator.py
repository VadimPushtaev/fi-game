from __future__ import annotations

import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from jinja2 import Template

from fi_game.lexicon import YamlLexiconStore
from fi_game.lexicon.batch import load_yaml_sequence


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_LEXICON_PATH = ROOT_DIR / "data" / "fi_50k.yaml"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "fi_50k_verb_study.yaml"
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "verb_study_prompt.j2"
FORM_ORDER = ("1sg", "2sg", "3sg", "1pl", "2pl", "3pl", "imperative", "negative")
RECORDS_PER_FORM = 2
EXPECTED_RECORDS_PER_VERB = len(FORM_ORDER) * RECORDS_PER_FORM

LEMMA_RESOLUTION_PROMPT = """You are normalizing Finnish verb surface forms to verb lemmas.

Return YAML only.

Task
- For each Finnish verb surface form below, infer the Finnish dictionary lemma.
- Use the standard 1st infinitive dictionary form as the lemma, for example:
  - olen -> olla
  - voin -> voida
  - tulee -> tulla
- Preserve input order exactly.
- Return one YAML item per input form.

Output schema
- surface: <input surface form>
  lemma: <Finnish lemma>

Rules
- Do not add commentary or Markdown fences.
- The lemma must be lowercase.
- Keep the surface form exactly as given.

SURFACE_FORMS
{% for surface in surfaces -%}
- {{ surface }}
{% endfor %}
"""


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


@dataclass(frozen=True, slots=True)
class StudyVerb:
    lemma: str
    source_forms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RowRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError("Row range start must be >= 1.")
        if self.end < self.start:
            raise ValueError("Row range end must be >= start.")

    def slice_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return rows[self.start - 1 : self.end]

    def as_metadata(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class VerbStudyRunResult:
    total_verbs: int
    generated_verbs: int
    skipped_verbs: int


def deduplicate_study_verbs(surface_to_lemma: dict[str, str]) -> list[StudyVerb]:
    grouped: dict[str, list[str]] = {}
    ordered_lemmas: list[str] = []

    for surface, lemma in surface_to_lemma.items():
        normalized_lemma = lemma.strip().lower()
        if not normalized_lemma:
            raise ValueError(f"Resolved lemma is empty for surface form '{surface}'.")

        if normalized_lemma not in grouped:
            grouped[normalized_lemma] = []
            ordered_lemmas.append(normalized_lemma)
        grouped[normalized_lemma].append(surface)

    return [
        StudyVerb(lemma=lemma, source_forms=tuple(grouped[lemma]))
        for lemma in ordered_lemmas
    ]


class CodexPromptRunner:
    def __init__(self, root_dir: Path = ROOT_DIR) -> None:
        self.root_dir = root_dir

    def run(self, prompt: str, *, model: str | None = None) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", prefix="codex-verb-study-", delete=False
        ) as tmp_file:
            output_path = Path(tmp_file.name)

        command = [
            "codex",
            "exec",
            "--full-auto",
            "--cd",
            str(self.root_dir),
            "--skip-git-repo-check",
            "--color",
            "never",
            "--output-last-message",
            str(output_path),
            "-",
        ]
        if model:
            command[2:2] = ["--model", model]

        try:
            result = subprocess.run(
                command,
                input=prompt,
                text=True,
                cwd=self.root_dir,
                capture_output=True,
            )
            if result.returncode != 0:
                error_output = result.stderr.strip() or result.stdout.strip()
                raise SystemExit(error_output or result.returncode)

            return output_path.read_text(encoding="utf-8")
        finally:
            output_path.unlink(missing_ok=True)


class VerbStudyLexiconSlice:
    def __init__(self, lexicon_path: Path, *, verb_range: RowRange, vocabulary_range: RowRange) -> None:
        self.lexicon_path = lexicon_path
        self.verb_range = verb_range
        self.vocabulary_range = vocabulary_range

    def load_all_rows(self) -> list[dict[str, Any]]:
        return YamlLexiconStore(self.lexicon_path).load_entries()

    def load_verb_rows(self) -> list[dict[str, Any]]:
        return self.verb_range.slice_rows(self.load_all_rows())

    def load_vocabulary_rows(self) -> list[dict[str, Any]]:
        return self.vocabulary_range.slice_rows(self.load_all_rows())

    def eligible_surface_verbs(self) -> list[str]:
        seen: set[str] = set()
        surfaces: list[str] = []

        for row in self.load_verb_rows():
            if row.get("pos") != "verb":
                continue

            word = str(row.get("word", "")).strip()
            if not word or word in seen:
                continue

            seen.add(word)
            surfaces.append(word)

        return surfaces

    def allowed_words_by_pos(self) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = {}
        seen_by_pos: dict[str, set[str]] = {}

        for row in self.load_vocabulary_rows():
            pos = str(row.get("pos", "")).strip()
            word = str(row.get("word", "")).strip()
            if not pos or not word:
                continue

            if pos not in seen_by_pos:
                seen_by_pos[pos] = set()
                buckets[pos] = []

            if word in seen_by_pos[pos]:
                continue

            seen_by_pos[pos].add(word)
            buckets[pos].append(word)

        return buckets


class VerbLemmaResolver:
    def __init__(self, runner: CodexPromptRunner) -> None:
        self.runner = runner

    def resolve(self, surfaces: list[str], *, model: str | None = None) -> dict[str, str]:
        if not surfaces:
            return {}

        prompt = Template(LEMMA_RESOLUTION_PROMPT).render(surfaces=surfaces)
        raw_output = self.runner.run(prompt, model=model)
        parsed = yaml.load(raw_output, Loader=yaml.BaseLoader) or []
        if not isinstance(parsed, list):
            raise ValueError("Expected lemma resolver output to be a YAML sequence.")

        resolved: dict[str, str] = {}
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("Each lemma resolver item must be a mapping.")
            surface = str(item.get("surface", "")).strip()
            lemma = str(item.get("lemma", "")).strip()
            if not surface or not lemma:
                raise ValueError("Each lemma resolver item must include non-empty surface and lemma.")
            resolved[surface] = lemma

        missing = [surface for surface in surfaces if surface not in resolved]
        if missing:
            raise ValueError(f"Lemma resolver missed surface forms: {missing}")

        return {surface: resolved[surface] for surface in surfaces}


class VerbStudyOutputStore:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def load_records(self) -> list[dict[str, Any]]:
        if not self.output_path.exists():
            return []

        return load_yaml_sequence(self.output_path)

    def is_complete(self, lemma: str, *, variant: str | None = None) -> bool:
        records = [
            record
            for record in self.load_records()
            if record.get("verb") == lemma
            and self._variant_matches(record, variant)
        ]
        if len(records) != EXPECTED_RECORDS_PER_VERB:
            return False

        counts = Counter(str(record.get("verb_form")) for record in records)
        return all(counts.get(form, 0) == RECORDS_PER_FORM for form in FORM_ORDER)

    def replace_records_for_lemma(
        self,
        lemma: str,
        records: list[dict[str, Any]],
        *,
        generation_metadata: dict[str, Any] | None = None,
        variant: str | None = None,
    ) -> None:
        if generation_metadata is not None:
            records = [
                {
                    **record,
                    "generation": generation_metadata,
                }
                for record in records
            ]
        self._validate_records_for_lemma(lemma, records)
        existing = [
            record
            for record in self.load_records()
            if not (
                record.get("verb") == lemma
                and self._variant_matches(record, variant)
            )
        ]
        combined = existing + records

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            yaml.safe_dump(combined, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _validate_records_for_lemma(self, lemma: str, records: list[dict[str, Any]]) -> None:
        if len(records) != EXPECTED_RECORDS_PER_VERB:
            raise ValueError(
                f"Expected {EXPECTED_RECORDS_PER_VERB} records for '{lemma}', got {len(records)}."
            )

        seen_pairs: set[tuple[str, str]] = set()
        counts = Counter()
        for record in records:
            self._validate_record(record)
            if record["verb"] != lemma:
                raise ValueError(
                    f"Expected every record verb to be '{lemma}', got '{record['verb']}'."
                )
            counts[record["verb_form"]] += 1

            uniqueness_key = (record["verb_form"], record["sentence_fi"])
            if uniqueness_key in seen_pairs:
                raise ValueError(
                    f"Duplicate sentence detected for lemma '{lemma}' and form "
                    f"'{record['verb_form']}'."
                )
            seen_pairs.add(uniqueness_key)

        for form in FORM_ORDER:
            if counts.get(form, 0) != RECORDS_PER_FORM:
                raise ValueError(
                    f"Expected {RECORDS_PER_FORM} records for '{lemma}' form '{form}', "
                    f"got {counts.get(form, 0)}."
                )

    def _validate_record(self, record: dict[str, Any]) -> None:
        required_fields = {
            "verb",
            "verb_form",
            "sentence_fi",
            "sentence_fi_masked",
            "answer_fi",
            "sentence_en",
            "sentence_en_masked",
        }
        missing_fields = sorted(field for field in required_fields if field not in record)
        if missing_fields:
            raise ValueError(f"Study record is missing required fields: {missing_fields}")

        for field in required_fields:
            value = record[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Study record field '{field}' must be a non-empty string.")

        if record["verb_form"] not in FORM_ORDER:
            raise ValueError(f"Unknown verb form '{record['verb_form']}'.")

        generation = record.get("generation")
        if generation is not None:
            if not isinstance(generation, dict):
                raise ValueError("Study record field 'generation' must be a mapping when present.")

            lexicon_path = generation.get("lexicon_path")
            if lexicon_path is not None and (not isinstance(lexicon_path, str) or not lexicon_path.strip()):
                raise ValueError(
                    "Study record generation field 'lexicon_path' must be a non-empty string."
                )
            variant = generation.get("variant")
            if variant is not None and (
                not isinstance(variant, str) or not variant.strip()
            ):
                raise ValueError(
                    "Study record generation field 'variant' must be a non-empty string."
                )

            for range_name in ("verb_rows", "vocabulary_rows"):
                range_data = generation.get(range_name)
                if not isinstance(range_data, dict):
                    raise ValueError(
                        f"Study record generation field '{range_name}' must be a mapping."
                    )
                start = self._parse_positive_int(range_data.get("start"))
                end = self._parse_positive_int(range_data.get("end"))
                if start is None or end is None:
                    raise ValueError(
                        f"Study record generation field '{range_name}' must contain integer "
                        "'start' and 'end'."
                    )
                if start < 1 or end < start:
                    raise ValueError(
                        f"Study record generation field '{range_name}' has an invalid range."
                    )

        answer = record["answer_fi"]
        sentence = record["sentence_fi"]
        masked_sentence = record["sentence_fi_masked"]
        if "%%%%" in sentence:
            raise ValueError("The unmasked Finnish sentence must not contain the %%%% placeholder.")
        if "%%%%" in answer:
            raise ValueError("The Finnish answer form must not contain the %%%% placeholder.")

        if masked_sentence.count("%%%%") != 1:
            raise ValueError("The masked Finnish sentence must contain exactly one %%%% placeholder.")
        if record["sentence_en_masked"].count("%%%%") != 1:
            raise ValueError("The masked English sentence must contain exactly one %%%% placeholder.")

    @staticmethod
    def _parse_positive_int(value: Any) -> int | None:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return None

    @staticmethod
    def _record_variant(record: dict[str, Any]) -> str | None:
        generation = record.get("generation")
        if not isinstance(generation, dict):
            return None

        variant = generation.get("variant")
        if variant is None:
            return None
        return str(variant).strip() or None

    @classmethod
    def _variant_matches(cls, record: dict[str, Any], variant: str | None) -> bool:
        if variant is None:
            return True
        return cls._record_variant(record) == variant


class VerbStudyGenerator:
    def __init__(self, prompt_path: Path, runner: CodexPromptRunner) -> None:
        self.prompt_path = prompt_path
        self.runner = runner

    def render_prompt(
        self,
        *,
        lexicon_path: str,
        output_yaml_path: str,
        verb_range: RowRange,
        vocabulary_range: RowRange,
        study_verb: StudyVerb,
        allowed_words_by_pos: dict[str, list[str]],
    ) -> str:
        template = Template(self.prompt_path.read_text(encoding="utf-8"))
        return template.render(
            lexicon_path=lexicon_path,
            output_yaml_path=output_yaml_path,
            verb_range=verb_range,
            vocabulary_range=vocabulary_range,
            form_order=FORM_ORDER,
            study_verb=study_verb,
            allowed_words_by_pos=allowed_words_by_pos,
        )

    def generate_records_for_verb(
        self,
        *,
        lexicon_path: str,
        output_yaml_path: str,
        verb_range: RowRange,
        vocabulary_range: RowRange,
        study_verb: StudyVerb,
        allowed_words_by_pos: dict[str, list[str]],
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        prompt = self.render_prompt(
            lexicon_path=lexicon_path,
            output_yaml_path=output_yaml_path,
            verb_range=verb_range,
            vocabulary_range=vocabulary_range,
            study_verb=study_verb,
            allowed_words_by_pos=allowed_words_by_pos,
        )
        raw_output = self.runner.run(prompt, model=model)
        parsed = yaml.load(raw_output, Loader=yaml.BaseLoader) or []
        if not isinstance(parsed, list):
            raise ValueError("Expected study generator output to be a YAML sequence.")

        records: list[dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("Each study sentence item must be a mapping.")
            records.append(item)

        return records


class VerbStudyRunner:
    def __init__(
        self,
        *,
        lexicon_slice: VerbStudyLexiconSlice,
        lemma_resolver: VerbLemmaResolver,
        generator: VerbStudyGenerator,
        output_store: VerbStudyOutputStore,
    ) -> None:
        self.lexicon_slice = lexicon_slice
        self.lemma_resolver = lemma_resolver
        self.generator = generator
        self.output_store = output_store

    def build_study_verbs(self, *, model: str | None = None) -> list[StudyVerb]:
        surfaces = self.lexicon_slice.eligible_surface_verbs()
        surface_to_lemma = self.lemma_resolver.resolve(surfaces, model=model)
        return deduplicate_study_verbs(surface_to_lemma)

    def generate_all(
        self,
        *,
        lexicon_path_for_prompt: str,
        output_yaml_path_for_prompt: str,
        variant: str | None = None,
        force: bool = False,
        model: str | None = None,
        dry_run: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ) -> VerbStudyRunResult:
        if progress_callback is not None:
            progress_callback("Collecting eligible verb forms from the lexicon slice.")
        study_verbs = self.build_study_verbs(model=model)
        if progress_callback is not None:
            progress_callback(f"Resolved {len(study_verbs)} verb lemma(s) to process.")
        allowed_words_by_pos = self.lexicon_slice.allowed_words_by_pos()
        generated_verbs = 0
        skipped_verbs = 0

        for index, study_verb in enumerate(study_verbs, start=1):
            if not force and self.output_store.is_complete(study_verb.lemma, variant=variant):
                skipped_verbs += 1
                if progress_callback is not None:
                    progress_callback(
                        f"[{index}/{len(study_verbs)}] Skipping '{study_verb.lemma}' "
                        "because it is already complete."
                    )
                continue

            if dry_run:
                raise ValueError("Dry-run should be handled before generate_all is called.")

            if progress_callback is not None:
                progress_callback(
                    f"[{index}/{len(study_verbs)}] Generating study sentences for "
                    f"'{study_verb.lemma}'."
                )
            records = self.generator.generate_records_for_verb(
                lexicon_path=lexicon_path_for_prompt,
                output_yaml_path=output_yaml_path_for_prompt,
                verb_range=self.lexicon_slice.verb_range,
                vocabulary_range=self.lexicon_slice.vocabulary_range,
                study_verb=study_verb,
                allowed_words_by_pos=allowed_words_by_pos,
                model=model,
            )
            self.output_store.replace_records_for_lemma(
                study_verb.lemma,
                records,
                generation_metadata={
                    "lexicon_path": lexicon_path_for_prompt,
                    "variant": variant,
                    "verb_rows": self.lexicon_slice.verb_range.as_metadata(),
                    "vocabulary_rows": self.lexicon_slice.vocabulary_range.as_metadata(),
                },
                variant=variant,
            )
            generated_verbs += 1
            if progress_callback is not None:
                progress_callback(
                    f"[{index}/{len(study_verbs)}] Wrote 16 records for '{study_verb.lemma}'."
                )

        return VerbStudyRunResult(
            total_verbs=len(study_verbs),
            generated_verbs=generated_verbs,
            skipped_verbs=skipped_verbs,
        )

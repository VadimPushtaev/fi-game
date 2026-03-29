#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fi_sv_spel.verb_study import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PROMPT_PATH,
    CodexPromptRunner,
    VerbLemmaResolver,
    VerbStudyGenerator,
    VerbStudyLexiconSlice,
    VerbStudyOutputStore,
    VerbStudyRunner,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YAML verb-study sentences from the first N rows of fi_50k.yaml."
    )
    parser.add_argument("first_n_rows", type=int, help="Number of raw YAML rows to use")
    parser.add_argument(
        "--lexicon",
        dest="lexicon_path",
        default="data/fi_50k.yaml",
        help="Path to the source lexicon YAML, relative to the repo root by default",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=str(DEFAULT_OUTPUT_PATH.relative_to(ROOT_DIR)),
        help="Path to the generated study YAML, relative to the repo root by default",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_path",
        default=str(DEFAULT_PROMPT_PATH.relative_to(ROOT_DIR)),
        help="Path to the Jinja prompt template, relative to the repo root by default",
    )
    parser.add_argument("--model", help="Optional codex model override")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the prompt for the first pending verb and print it without invoking codex",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.first_n_rows < 1:
        raise SystemExit("first_n_rows must be >= 1")

    lexicon_path = resolve_repo_path(args.lexicon_path)
    output_path = resolve_repo_path(args.output_path)
    prompt_path = resolve_repo_path(args.prompt_path)

    if not lexicon_path.exists():
        raise SystemExit(f"Lexicon YAML not found: {lexicon_path}")
    if not prompt_path.exists():
        raise SystemExit(f"Prompt template not found: {prompt_path}")

    runner = CodexPromptRunner()
    lexicon_slice = VerbStudyLexiconSlice(lexicon_path=lexicon_path, first_n_rows=args.first_n_rows)
    output_store = VerbStudyOutputStore(output_path=output_path)
    study_runner = VerbStudyRunner(
        lexicon_slice=lexicon_slice,
        lemma_resolver=VerbLemmaResolver(runner),
        generator=VerbStudyGenerator(prompt_path=prompt_path, runner=runner),
        output_store=output_store,
    )

    def report_progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    if args.dry_run:
        report_progress("Resolving verb lemmas for dry-run prompt rendering.")
        study_verbs = study_runner.build_study_verbs(model=args.model)
        report_progress(f"Found {len(study_verbs)} verb lemma(s) in the selected lexicon slice.")
        allowed_words_by_pos = lexicon_slice.allowed_words_by_pos()
        generator = study_runner.generator
        pending_verb = next(
            (study_verb for study_verb in study_verbs if not output_store.is_complete(study_verb.lemma)),
            None,
        )
        if pending_verb is None:
            report_progress("All eligible verbs are already complete.")
            return 0

        report_progress(f"Rendering prompt for the first pending lemma: '{pending_verb.lemma}'.")
        prompt = generator.render_prompt(
            lexicon_path=args.lexicon_path,
            output_yaml_path=args.output_path,
            first_n_rows=args.first_n_rows,
            study_verb=pending_verb,
            allowed_words_by_pos=allowed_words_by_pos,
        )
        sys.stdout.write(prompt)
        if not prompt.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    result = study_runner.generate_all(
        lexicon_path_for_prompt=args.lexicon_path,
        output_yaml_path_for_prompt=args.output_path,
        model=args.model,
        progress_callback=report_progress,
    )
    print(
        f"Processed {result.total_verbs} verb lemma(s): "
        f"{result.generated_verbs} generated, {result.skipped_verbs} skipped. "
        f"Output: {output_path.relative_to(ROOT_DIR)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fi_game.lexicon import DEFAULT_PROMPT_PATH, DEFAULT_YAML_PATH, BatchGenerator, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Finnish YAML batch prompt and run it with codex."
    )
    parser.add_argument("first_line", type=int, help="1-based starting line in data/fi_50k.txt")
    parser.add_argument("batch_length", type=int, help="Number of words to process")
    parser.add_argument(
        "--yaml",
        dest="yaml_path",
        default=str(DEFAULT_YAML_PATH.relative_to(ROOT_DIR)),
        help="Path to the accumulated YAML file, relative to the repo root by default",
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
        help="Print the rendered prompt and exit without invoking codex",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.first_line < 1:
        raise SystemExit("first_line must be >= 1")
    if args.batch_length < 1:
        raise SystemExit("batch_length must be >= 1")

    prompt_path = resolve_repo_path(args.prompt_path)
    yaml_path = resolve_repo_path(args.yaml_path)
    if not prompt_path.exists():
        raise SystemExit(f"Prompt template not found: {prompt_path}")

    generator = BatchGenerator(prompt_path=prompt_path)
    if args.dry_run:
        prompt = generator.render_prompt(args.first_line, args.batch_length, args.yaml_path)
        sys.stdout.write(prompt)
        if not prompt.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    generator.generate_and_append(
        first_line=args.first_line,
        batch_length=args.batch_length,
        yaml_path=yaml_path,
        yaml_path_for_prompt=args.yaml_path,
        model=args.model,
    )
    print(f"Appended batch to {yaml_path.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

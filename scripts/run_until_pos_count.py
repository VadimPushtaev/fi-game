#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fi_game.lexicon import (
    DEFAULT_PROMPT_PATH,
    DEFAULT_YAML_PATH,
    BatchGenerator,
    PosTargetBatchRunner,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YAML batches until a part-of-speech count reaches a target."
    )
    parser.add_argument("batch_size", type=int, help="Number of source words per generation batch")
    parser.add_argument("pos", help="Part of speech to track, for example 'pronoun'")
    parser.add_argument("target_count", type=int, help="Stop once this many YAML entries exist")
    parser.add_argument(
        "--start-line",
        type=int,
        default=1,
        help="1-based starting line in data/fi_50k.txt for the first batch",
    )
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt_path = resolve_repo_path(args.prompt_path)
    yaml_path = resolve_repo_path(args.yaml_path)
    if not prompt_path.exists():
        raise SystemExit(f"Prompt template not found: {prompt_path}")

    generator = BatchGenerator(prompt_path=prompt_path)
    runner = PosTargetBatchRunner(generator=generator)
    result = runner.run_until_target(
        batch_size=args.batch_size,
        pos=args.pos,
        target_count=args.target_count,
        start_line=args.start_line,
        yaml_path=yaml_path,
        yaml_path_for_prompt=args.yaml_path,
        model=args.model,
    )
    print(
        f"Reached {result.final_count} '{result.pos}' entries after "
        f"{result.batches_run} batch(es). Next first line: {result.next_first_line}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

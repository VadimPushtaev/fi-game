#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fi_sv_spel.lexicon import DEFAULT_YAML_PATH, YamlLexiconStore, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count total YAML records.")
    parser.add_argument(
        "yaml_path",
        nargs="?",
        default=str(DEFAULT_YAML_PATH.relative_to(ROOT_DIR)),
        help="Path to the YAML file, relative to the repo root by default",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    yaml_path = resolve_repo_path(args.yaml_path)
    if not yaml_path.exists():
        raise SystemExit(f"YAML file not found: {yaml_path}")

    print(YamlLexiconStore(yaml_path).count_records())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

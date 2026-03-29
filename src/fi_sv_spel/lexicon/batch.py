from __future__ import annotations

import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_PATH = ROOT_DIR / "data" / "fi_50k.txt"
DEFAULT_YAML_PATH = ROOT_DIR / "data" / "fi_50k.yaml"
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "fi_50k_yaml_batch_prompt.j2"


def load_yaml_sequence(path: Path) -> list[dict[str, Any]]:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader) or []
    if not isinstance(data, list):
        raise ValueError("Expected a top-level YAML sequence.")

    validated: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Expected every YAML item to be a mapping.")
        validated.append(item)

    return validated


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


class YamlLexiconStore:
    def __init__(self, yaml_path: Path) -> None:
        self.yaml_path = yaml_path

    def load_entries(self) -> list[dict[str, Any]]:
        if not self.yaml_path.exists():
            return []

        return load_yaml_sequence(self.yaml_path)

    def count_by_pos(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for entry in self.load_entries():
            pos = entry.get("pos")
            if not pos:
                raise ValueError("Each YAML item must include a non-empty 'pos'.")
            counts[str(pos)] += 1
        return counts

    def count_for_pos(self, pos: str) -> int:
        return self.count_by_pos().get(pos, 0)

    def count_records(self) -> int:
        return len(self.load_entries())

    def append_batch(self, batch_yaml: str) -> None:
        self.yaml_path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.yaml_path.read_text(encoding="utf-8") if self.yaml_path.exists() else ""
        batch_text = batch_yaml.strip()
        if not batch_text:
            raise ValueError("Codex returned an empty final message.")

        if existing.strip():
            separator = "" if existing.endswith("\n") else "\n"
            self.yaml_path.write_text(f"{existing}{separator}{batch_text}\n", encoding="utf-8")
            return

        self.yaml_path.write_text(f"{batch_text}\n", encoding="utf-8")


class BatchGenerator:
    def __init__(self, prompt_path: Path, root_dir: Path = ROOT_DIR) -> None:
        self.prompt_path = prompt_path
        self.root_dir = root_dir

    def render_prompt(self, first_line: int, batch_length: int, yaml_path_for_prompt: str) -> str:
        template = Template(self.prompt_path.read_text(encoding="utf-8"))
        return template.render(
            first_line=first_line,
            batch_length=batch_length,
            existing_yaml_path=yaml_path_for_prompt,
        )

    def generate_batch(
        self,
        first_line: int,
        batch_length: int,
        yaml_path_for_prompt: str,
        *,
        model: str | None = None,
    ) -> str:
        prompt = self.render_prompt(first_line, batch_length, yaml_path_for_prompt)
        return self._run_codex(prompt, model=model)

    def generate_and_append(
        self,
        first_line: int,
        batch_length: int,
        yaml_path: Path,
        yaml_path_for_prompt: str,
        *,
        model: str | None = None,
    ) -> str:
        batch_yaml = self.generate_batch(
            first_line=first_line,
            batch_length=batch_length,
            yaml_path_for_prompt=yaml_path_for_prompt,
            model=model,
        )
        YamlLexiconStore(yaml_path).append_batch(batch_yaml)
        return batch_yaml

    def _run_codex(self, prompt: str, *, model: str | None = None) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", prefix="codex-fi-50k-", delete=False
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


@dataclass(slots=True)
class PosTargetRunResult:
    pos: str
    target_count: int
    final_count: int
    batches_run: int
    next_first_line: int


class PosTargetBatchRunner:
    def __init__(self, generator: BatchGenerator, source_path: Path = DEFAULT_SOURCE_PATH) -> None:
        self.generator = generator
        self.source_path = source_path

    def run_until_target(
        self,
        *,
        batch_size: int,
        pos: str,
        target_count: int,
        start_line: int,
        yaml_path: Path,
        yaml_path_for_prompt: str,
        model: str | None = None,
    ) -> PosTargetRunResult:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if target_count < 0:
            raise ValueError("target_count must be >= 0")
        if start_line < 1:
            raise ValueError("start_line must be >= 1")

        total_source_lines = self._count_source_lines()
        store = YamlLexiconStore(yaml_path)
        current_count = store.count_for_pos(pos)
        next_first_line = start_line
        batches_run = 0

        while current_count < target_count:
            if next_first_line > total_source_lines:
                raise RuntimeError(
                    f"Reached end of {self.source_path.relative_to(ROOT_DIR)} before "
                    f"'{pos}' reached {target_count}. Current count: {current_count}."
                )

            current_batch_size = min(batch_size, total_source_lines - next_first_line + 1)
            self.generator.generate_and_append(
                first_line=next_first_line,
                batch_length=current_batch_size,
                yaml_path=yaml_path,
                yaml_path_for_prompt=yaml_path_for_prompt,
                model=model,
            )
            batches_run += 1
            next_first_line += current_batch_size
            current_count = store.count_for_pos(pos)

        return PosTargetRunResult(
            pos=pos,
            target_count=target_count,
            final_count=current_count,
            batches_run=batches_run,
            next_first_line=next_first_line,
        )

    def _count_source_lines(self) -> int:
        with self.source_path.open(encoding="utf-8") as handle:
            return sum(1 for _ in handle)

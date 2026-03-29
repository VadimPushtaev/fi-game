from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs

import yaml
from fastapi import Request


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DECK_PATH = BASE_DIR / "data" / "custom_verb_study.yaml"
DEFAULT_SESSION_KEY = os.environ.get("FI_GAME_SECRET_KEY", "fi-game-dev-secret")
GAME_SESSION_KEY = "verb_game"
ACTIVE_STATUS = "active"
SOLVED_STATUS = "solved"
REVEALED_STATUS = "revealed"
FORM_LABELS = {
    "1sg": "I",
    "2sg": "You",
    "3sg": "He / She / It",
    "1pl": "We",
    "2pl": "You all",
    "3pl": "They",
    "imperative": "Imperative",
    "negative": "Negative",
}
HINT_SEQUENCE = (
    ("english_masked", "English without the verb"),
    ("english_full", "English with the verb"),
    ("lemma", "Infinitive"),
    ("answer", "Correct form"),
)
REQUIRED_CARD_FIELDS = (
    "verb",
    "verb_form",
    "sentence_fi",
    "sentence_fi_masked",
    "answer_fi",
    "sentence_en",
    "sentence_en_masked",
)


@dataclass(frozen=True, slots=True)
class StudyCard:
    verb: str
    verb_form: str
    sentence_fi: str
    sentence_fi_masked: str
    answer_fi: str
    sentence_en: str
    sentence_en_masked: str

    @property
    def verb_form_label(self) -> str:
        return FORM_LABELS.get(self.verb_form, self.verb_form)


def _load_yaml_rows(path: Path) -> list[dict[str, Any]]:
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader) or []
    if not isinstance(data, list):
        raise ValueError("Expected the study deck YAML to be a top-level sequence.")

    validated: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("Expected every study deck item to be a mapping.")
        validated.append(row)
    return validated


@lru_cache(maxsize=8)
def load_study_deck(path: Path) -> tuple[StudyCard, ...]:
    if not path.exists():
        raise FileNotFoundError(f"Study deck not found: {path}")

    cards: list[StudyCard] = []
    for row in _load_yaml_rows(path):
        missing = [field for field in REQUIRED_CARD_FIELDS if not str(row.get(field, "")).strip()]
        if missing:
            raise ValueError(f"Study deck row is missing required fields: {missing}")

        cards.append(
            StudyCard(
                verb=str(row["verb"]).strip(),
                verb_form=str(row["verb_form"]).strip(),
                sentence_fi=str(row["sentence_fi"]).strip(),
                sentence_fi_masked=str(row["sentence_fi_masked"]).strip(),
                answer_fi=str(row["answer_fi"]).strip(),
                sentence_en=str(row["sentence_en"]).strip(),
                sentence_en_masked=str(row["sentence_en_masked"]).strip(),
            )
        )

    if not cards:
        raise ValueError("Study deck is empty.")

    return tuple(cards)


def _normalize_answer(value: str) -> str:
    return value.strip().lower()


async def read_form_value(request: Request, key: str) -> str:
    body = (await request.body()).decode("utf-8")
    values = parse_qs(body)
    return values.get(key, [""])[0]


class GameService:
    def __init__(
        self,
        deck_path: Path = DEFAULT_DECK_PATH,
        *,
        seed_factory: Callable[[], int] | None = None,
    ) -> None:
        self.deck_path = deck_path
        self.seed_factory = seed_factory or (lambda: secrets.randbits(32))

    def deck(self) -> tuple[StudyCard, ...]:
        return load_study_deck(self.deck_path)

    def build_view(self, request: Request, *, feedback: dict[str, str] | None = None) -> dict[str, Any]:
        deck = self.deck()
        state = self._ensure_state(request, len(deck))
        current_card = deck[int(state["card_index"])]
        hint_level = max(0, min(int(state["hint_level"]), len(HINT_SEQUENCE)))
        status = str(state["status"])
        draft_answer = str(state.get("draft_answer", ""))
        hints = self._build_hints(current_card, hint_level, status)
        available_hint_actions = self._build_available_hint_actions(hint_level, status)
        before_text, after_text = current_card.sentence_fi_masked.split("%%%%", 1)

        return {
            "screen": "card",
            "card": current_card,
            "show_input": status == ACTIVE_STATUS,
            "can_next": status in {SOLVED_STATUS, REVEALED_STATUS},
            "feedback": feedback,
            "hints": hints,
            "available_hint_actions": available_hint_actions,
            "sentence_before": before_text,
            "sentence_after": after_text,
            "draft_answer": draft_answer,
            "draft_answer_width": max(3, len(draft_answer) + 1),
        }

    def check_answer(self, request: Request, answer: str) -> dict[str, Any]:
        state = self._ensure_state(request, len(self.deck()))
        if state["status"] != ACTIVE_STATUS:
            return self.build_view(request)

        current_card = self.current_card(request)
        if _normalize_answer(answer) == _normalize_answer(current_card.answer_fi):
            state["status"] = SOLVED_STATUS
            state["draft_answer"] = ""
            self._save_state(request, state)
            return self.build_view(
                request,
                feedback={"kind": "success", "message": "Exactly right. Nice work."},
            )

        state["draft_answer"] = answer
        self._save_state(request, state)
        return self.build_view(
            request,
            feedback={"kind": "error", "message": "Not quite. Try again or reveal a hint."},
        )

    def reveal_hint(self, request: Request, target_level: int) -> dict[str, Any]:
        state = self._ensure_state(request, len(self.deck()))
        if state["status"] != ACTIVE_STATUS:
            return self.build_view(request)

        current_level = int(state["hint_level"])
        clamped_target_level = max(current_level + 1, min(target_level, len(HINT_SEQUENCE)))
        state["hint_level"] = clamped_target_level
        if state["hint_level"] >= len(HINT_SEQUENCE):
            state["status"] = REVEALED_STATUS
        self._save_state(request, state)

        feedback = None
        if state["status"] == REVEALED_STATUS:
            feedback = {"kind": "info", "message": "Answer revealed. Move on when you're ready."}

        return self.build_view(request, feedback=feedback)

    def move_next(self, request: Request) -> dict[str, Any]:
        deck = self.deck()
        state = self._ensure_state(request, len(deck))
        if state["status"] not in {SOLVED_STATUS, REVEALED_STATUS}:
            return self.build_view(request)

        state["card_index"] = self._pick_card_index(len(deck), exclude=int(state["card_index"]))
        state["hint_level"] = 0
        state["status"] = ACTIVE_STATUS
        state["draft_answer"] = ""
        self._save_state(request, state)
        return self.build_view(request)

    def restart(self, request: Request) -> dict[str, Any]:
        self._save_state(request, self._initial_state())
        return self.build_view(request)

    def current_card(self, request: Request) -> StudyCard:
        deck = self.deck()
        state = self._ensure_state(request, len(deck))
        return deck[int(state["card_index"])]

    def _build_hints(self, card: StudyCard, hint_level: int, status: str) -> list[dict[str, str]]:
        hints: list[dict[str, str]] = []
        if hint_level >= 1:
            hints.append(
                {
                    "title": HINT_SEQUENCE[0][1],
                    "value": card.sentence_en_masked,
                    "tone": "soft",
                }
            )
        if hint_level >= 2:
            hints.append(
                {
                    "title": HINT_SEQUENCE[1][1],
                    "value": card.sentence_en,
                    "tone": "warm",
                }
            )
        if hint_level >= 3:
            hints.append(
                {
                    "title": HINT_SEQUENCE[2][1],
                    "value": card.verb,
                    "tone": "accent",
                }
            )
        if hint_level >= 4 or status in {SOLVED_STATUS, REVEALED_STATUS}:
            hints.append(
                {
                    "title": HINT_SEQUENCE[3][1],
                    "value": card.answer_fi,
                    "tone": "strong",
                }
            )
        return hints

    def _build_available_hint_actions(self, hint_level: int, status: str) -> list[dict[str, int | str]]:
        if status != ACTIVE_STATUS:
            return []

        return [
            {
                "level": index,
                "label": label,
            }
            for index, (_, label) in enumerate(HINT_SEQUENCE, start=1)
            if index > hint_level
        ]

    def _initial_state(self) -> dict[str, int | str]:
        return {
            "card_index": self._pick_card_index(len(self.deck())),
            "hint_level": 0,
            "status": ACTIVE_STATUS,
            "draft_answer": "",
        }

    def _ensure_state(self, request: Request, deck_size: int) -> dict[str, int | str]:
        game_state = request.session.get(GAME_SESSION_KEY)
        if not isinstance(game_state, dict):
            game_state = self._initial_state()
            self._save_state(request, game_state)

        normalized = {
            "card_index": max(0, int(game_state.get("card_index", self._pick_card_index(deck_size)))),
            "hint_level": max(0, int(game_state.get("hint_level", 0))),
            "status": str(game_state.get("status", ACTIVE_STATUS)),
            "draft_answer": str(game_state.get("draft_answer", "")),
        }

        if normalized["status"] not in {ACTIVE_STATUS, SOLVED_STATUS, REVEALED_STATUS}:
            normalized["status"] = ACTIVE_STATUS

        if normalized["card_index"] >= deck_size:
            normalized["card_index"] = self._pick_card_index(deck_size)
            normalized["hint_level"] = 0

        self._save_state(request, normalized)
        return normalized

    def _save_state(self, request: Request, state: dict[str, int | str]) -> None:
        request.session[GAME_SESSION_KEY] = state

    def _pick_card_index(self, deck_size: int, *, exclude: int | None = None) -> int:
        if deck_size < 1:
            raise ValueError("Deck size must be at least 1.")

        candidate = self.seed_factory() % deck_size
        if exclude is not None and deck_size > 1 and candidate == exclude:
            return (candidate + 1) % deck_size
        return candidate

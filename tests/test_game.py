from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml

from fi_game.main import create_app


def write_deck(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            [
                {
                    "verb": "olla",
                    "verb_form": "1sg",
                    "sentence_fi": "Minä olen täällä nyt.",
                    "sentence_fi_masked": "Minä %%%% täällä nyt.",
                    "answer_fi": "olen",
                    "sentence_en": "I am here now.",
                    "sentence_en_masked": "I %%%% here now.",
                },
                {
                    "verb": "voida",
                    "verb_form": "2sg",
                    "sentence_fi": "Sinä voit auttaa nyt.",
                    "sentence_fi_masked": "Sinä %%%% auttaa nyt.",
                    "answer_fi": "voit",
                    "sentence_en": "You can help now.",
                    "sentence_en_masked": "You %%%% help now.",
                },
            ],
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def test_app(tmp_path: Path):
    deck_path = tmp_path / "deck.yaml"
    write_deck(deck_path)
    return create_app(deck_path=deck_path, secret_key="test-secret", seed_factory=lambda: 0)


@pytest.mark.anyio
async def test_index_renders_first_card_from_deck(test_app) -> None:
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert "Card 1 of 2" not in response.text
    assert "Minä" in response.text
    assert 'name="answer"' in response.text
    assert "täällä nyt." in response.text
    assert "Check" in response.text
    assert "Next sentence" in response.text
    assert "disabled" in response.text


@pytest.mark.anyio
async def test_wrong_answer_keeps_same_card_active(test_app) -> None:
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        await client.get("/")
        response = await client.post("/game/check", data={"answer": "ole"})

    assert response.status_code == 200
    assert "Not quite. Try again or reveal a hint." in response.text
    assert "Next sentence" in response.text
    assert "disabled" in response.text
    assert 'name="answer"' in response.text
    assert 'value="ole"' in response.text


@pytest.mark.anyio
async def test_correct_answer_reveals_next_button(test_app) -> None:
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        await client.get("/")
        response = await client.post("/game/check", data={"answer": "olen"})

    assert response.status_code == 200
    assert "Exactly right. Nice work." in response.text
    assert "Correct form" in response.text
    assert '<span class="inline-answer-solved">olen</span>' in response.text
    assert "Next sentence" in response.text


@pytest.mark.anyio
async def test_hint_flow_reveals_in_order_and_persists_on_reload(test_app) -> None:
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        await client.get("/")
        first_hint = await client.post("/game/hint", data={"hint_level": "1"})
        second_hint = await client.post("/game/hint", data={"hint_level": "2"})
        third_hint = await client.post("/game/hint", data={"hint_level": "3"})
        reloaded = await client.get("/")

    assert 'I <span class="blank">____</span> here now.' in first_hint.text
    assert "I am here now." in second_hint.text
    assert "Infinitive" in third_hint.text
    assert "olla" in third_hint.text
    assert "Infinitive" in reloaded.text
    assert "olla" in reloaded.text


@pytest.mark.anyio
async def test_can_jump_directly_to_deeper_hint_level(test_app) -> None:
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        await client.get("/")
        response = await client.post("/game/hint", data={"hint_level": "2"})

    assert response.status_code == 200
    assert "English without the verb" in response.text
    assert "English with the verb" in response.text
    assert '<article class="hint-card hint-accent">' not in response.text


@pytest.mark.anyio
async def test_final_hint_reveals_answer_and_next(test_app) -> None:
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        await client.get("/")
        response = await client.post("/game/hint", data={"hint_level": "4"})

    assert response.status_code == 200
    assert "Answer revealed. Move on when you&#39;re ready." in response.text
    assert "Correct form" in response.text
    assert '<span class="inline-answer-solved">olen</span>' in response.text
    assert "Next sentence" in response.text


@pytest.mark.anyio
async def test_next_moves_to_a_new_random_card(test_app) -> None:
    transport = httpx.ASGITransport(app=test_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        await client.get("/")
        await client.post("/game/check", data={"answer": "olen"})
        next_card = await client.post("/game/next")

    assert "Sinä" in next_card.text
    assert "voit" not in next_card.text
    assert 'name="answer"' in next_card.text
    assert "Next sentence" in next_card.text
    assert "disabled" in next_card.text

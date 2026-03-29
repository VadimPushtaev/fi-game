from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from fi_game.game import DEFAULT_DECK_PATH, DEFAULT_SESSION_KEY, GameService, read_form_value

BASE_DIR = Path(__file__).resolve().parents[2]
FAVICON_PATH = BASE_DIR / "favicon.png"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def create_app(
    *,
    deck_path: Path = DEFAULT_DECK_PATH,
    secret_key: str = DEFAULT_SESSION_KEY,
    seed_factory: Callable[[], int] | None = None,
) -> FastAPI:
    app = FastAPI(title="fi-game")
    app.add_middleware(SessionMiddleware, secret_key=secret_key)
    game_service = GameService(deck_path=deck_path, seed_factory=seed_factory)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "index.html", game_service.build_view(request))

    @app.get("/favicon.png")
    async def favicon() -> FileResponse:
        return FileResponse(FAVICON_PATH)

    @app.post("/game/check", response_class=HTMLResponse)
    async def check_answer(request: Request) -> HTMLResponse:
        answer = await read_form_value(request, "answer")
        return templates.TemplateResponse(request, "partials/game_panel.html", game_service.check_answer(request, answer))

    @app.post("/game/hint", response_class=HTMLResponse)
    async def reveal_hint(request: Request) -> HTMLResponse:
        target_level = int(await read_form_value(request, "hint_level") or "1")
        return templates.TemplateResponse(
            request,
            "partials/game_panel.html",
            game_service.reveal_hint(request, target_level),
        )

    @app.post("/game/next", response_class=HTMLResponse)
    async def next_card(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "partials/game_panel.html", game_service.move_next(request))

    @app.post("/game/restart", response_class=HTMLResponse)
    async def restart(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, "partials/game_panel.html", game_service.restart(request))

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

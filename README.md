# fi-sv-spel

`fi-sv-spel` is a web-based game for learning Finnish through Swedish.

This repository currently contains the project bootstrap only:

- a Python web app built with FastAPI
- server-rendered HTML templates via Jinja2
- HTMX included for future interactive UI work
- Poetry for dependency management
- Docker for local containerized development
- pre-commit hooks for basic code quality checks

There is no gameplay yet. The home page is intentionally empty so the app structure can evolve without committing to early UX decisions.

## Stack

- Python 3.12
- Poetry
- FastAPI
- Jinja2
- HTMX
- Pytest
- Ruff
- Docker Compose

## Local Development

Install dependencies:

```bash
poetry install
```

Run the app locally:

```bash
poetry run uvicorn fi_sv_spel.main:app --reload
```

The app will be available at `http://127.0.0.1:8000`.

## Docker Development

Build and start the app with Docker Compose:

```bash
docker compose up --build
```

The app will be available at `http://127.0.0.1:8000`.

If port `8000` is already in use, override the host port:

```bash
APP_PORT=8080 docker compose up --build
```

## Quality Checks

Run the test suite:

```bash
poetry run pytest
```

Run Ruff checks:

```bash
poetry run ruff check .
poetry run ruff format --check .
```

Run pre-commit hooks across the repository:

```bash
poetry run pre-commit run --all-files
```

## Project Layout

```text
.
|-- compose.yaml
|-- Dockerfile
|-- pyproject.toml
|-- src/
|   `-- fi_sv_spel/
|       |-- __init__.py
|       `-- main.py
|-- templates/
|   |-- base.html
|   `-- index.html
`-- tests/
    `-- test_app.py
```

## Initial Routes

- `GET /` renders the empty entry page
- `GET /health` returns a small JSON health response

## Roadmap

- add the first actual learning loop
- define the initial Finnish vocabulary/content model
- add HTMX-driven interactions
- introduce styling and game feedback

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    POETRY_NO_INTERACTION=1 \
    POETRY_VERSION=2.1.4 \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

WORKDIR /app

COPY pyproject.toml poetry.lock README.md ./
COPY src ./src
COPY templates ./templates

RUN poetry install --only main --no-root

EXPOSE 8000

CMD ["uvicorn", "fi_sv_spel.main:app", "--host", "0.0.0.0", "--port", "8000"]

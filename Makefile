pre-commit:
	poetry run pre-commit run --all-files && poetry run pytest tests/

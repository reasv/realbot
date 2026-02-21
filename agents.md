# Agent Runtime Rules

All coding agents working in this repository must use the project virtual environment at `.venv`.

## Required

- Run Python commands with `.venv/bin/python`.
- Run tests with `.venv/bin/pytest` (or `.venv/bin/python -m pytest`).
- Run package installs with `.venv/bin/pip`.
- Do not use system `python`, `pip`, or `pytest` for this repo.

## Examples

- `.venv/bin/python run_matrix.py`
- `.venv/bin/python -m pytest tests/test_matrix_chatbot.py -k mention`

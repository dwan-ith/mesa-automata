# Contributing to Mesa Von Neumann Automata

Thank you for your interest in contributing to this project!

## Development Setup

1. Ensure Python 3.10+ and [`uv`](https://github.com/astral-sh/uv) are installed.
2. Create virtual environment and install in editable mode with development dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

## Running Tests

All changes must pass the test suite:
```bash
pytest -v
```

When adding new transition rules or patterns, please include corresponding unit tests in `tests/`.

## Code Style

- Follow PEP 8 guidelines.
- Keep transition functions pure and free of side effects.
- Ensure state metadata in `vn_automata/states.py` stays aligned with literature standards (JvN29).

# Job Radar task runner. Run `just` to list targets.

# Show available recipes
default:
    @just --list

# Lint with ruff
lint:
    uv run ruff check .

# Auto-format (and apply lint fixes) with ruff
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Run the test suite
test:
    uv run pytest

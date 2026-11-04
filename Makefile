# Development tasks for the rydberg-dynamics project.
# On Windows, run these via Git Bash or WSL.

.DEFAULT_GOAL := help
.PHONY: help setup lint format typecheck test test-fast figures report poster slides clean

UV ?= uv
LATEXMK ?= latexmk -pdf -interaction=nonstopmode -halt-on-error

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the environment and install git hooks
	$(UV) sync
	$(UV) run pre-commit install
	$(UV) run nbstripout --install

lint: ## Lint and check formatting with ruff
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Auto-format and fix lint issues with ruff
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck: ## Type-check the package with mypy
	$(UV) run mypy src

test: ## Run the full test suite with coverage
	$(UV) run pytest --cov

test-fast: ## Run tests, skipping those marked slow
	$(UV) run pytest -m "not slow"

figures: ## Run every script in scripts/ to regenerate results
	@for script in scripts/*.py; do \
		echo ">>> $$script"; \
		$(UV) run python "$$script" || exit 1; \
	done

report: ## Build the report PDF
	cd reports/report && $(LATEXMK) main.tex

poster: ## Build the poster PDF
	cd reports/poster && $(LATEXMK) main.tex

slides: ## Build the slides PDF
	cd reports/slides && $(LATEXMK) main.tex

clean: ## Remove caches and LaTeX build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
	for d in reports/report reports/poster reports/slides; do \
		(cd $$d && latexmk -C >/dev/null 2>&1 || true); \
	done

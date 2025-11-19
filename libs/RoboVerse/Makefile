# Commands for formatting and linting
.PHONY: format lint

# Lint only staged Python files
lint:
	@files=$$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$$'); \
	if [ -n "$$files" ]; then \
		echo "Linting staged Python files..."; \
		echo $$files | xargs flake8; \
	else \
		echo "No Python files to lint."; \
	fi

# Format staged Python files (actual formatting, not checking)
format:
	@files=$$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$$'); \
	if [ -n "$$files" ]; then \
		echo "Formatting staged Python files..."; \
		echo $$files | xargs black; \
		echo $$files | xargs isort; \
	else \
		echo "No Python files to format."; \
	fi

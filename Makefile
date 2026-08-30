.PHONY: test
test:
	uv run pytest -v

.PHONY: test-cov
test-cov:
	uv run pytest --cov=frameworthy --cov-report=xml

.PHONY: release
release:
	@echo "Current Version: $$(uv version | cut -d' ' -f2)"
	@read -p "New Version: " VERSION; \
	read -s -p "PyPI Token: " TOKEN; echo; \
	uv version "$$VERSION"
	uv build && uv publish --token "$$TOKEN"

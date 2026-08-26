.PHONY: test
test:
	uv run pytest -v

.PHONY: release
release:
	@echo "Current Version: $$(uv version | cut -d' ' -f2)"
	@read -p "New Version: " VERSION; \
	read -s -p "PyPI Token: " TOKEN; echo; \
	uv version "$$VERSION"
	uv build && uv publish --token "$$TOKEN"

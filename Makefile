.PHONY: test lint demo build

test:
	env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check src tests

demo:
	.venv/bin/kincal demo --output outputs/demo

build:
	.venv/bin/python -m build

# Traxr — single command surface (build-plan execution contract).
# Targets land incrementally: stubbed ones fail with a clear
# "not yet built — lands in MN" message until their milestone.

PYTHON ?= python

.PHONY: install lint typecheck test cov property mutation analyzer-goldens standalone-check golden \
	external-golden selfcheck notebook build site verify-all verify-deep

install:
	$(PYTHON) -m pip install -e ".[dev,document,openai,langgraph,viz]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy src/traxr

test:
	$(PYTHON) -m pytest -q

# mas/ is omitted via [tool.coverage.run] in pyproject (informational only).
# Sources are paths (not dotted packages): coverage resolves dotted sources by
# importing them inside its sys_modules_saved block, which unloads the numpy
# C extension that `import traxr` pulls in transitively (fitz/PyMuPDF) — any
# later pandas/numpy import then fails with "cannot load module more than
# once per process".
cov:
	$(PYTHON) -m pytest --cov=src/traxr/metrics --cov=src/traxr/perturb --cov=src/traxr/trace --cov-fail-under=90
	$(PYTHON) -m pytest --cov=src/traxr/capture --cov=src/traxr/agents --cov-fail-under=85
	$(PYTHON) -m pytest --cov=src/traxr --cov-fail-under=75

property:
	$(PYTHON) -m pytest tests/property -q --hypothesis-seed=0

mutation:
	$(PYTHON) -m mutmut run

analyzer-goldens:
	$(PYTHON) scripts/check_analyzer_goldens.py

# Standalone gate: the repo must contain zero source-repo references.
standalone-check:
	@! git grep -rIiE 'mas[_-]?debug|mas[_-]?eval|72ede6d|huzaifasuri|Speena' -- . ':!Makefile' || (echo 'standalone-check: FAIL — forbidden references found' >&2; exit 1)
	@echo 'standalone-check: PASS'

golden:
	$(PYTHON) -m pytest tests/e2e/test_golden.py -q

external-golden:
	$(PYTHON) -m pytest tests/e2e/test_external_golden.py -q

selfcheck:
	$(PYTHON) -m traxr.selfcheck

# Executes top-to-bottom with NO API key (real-model cells are skip-safe).
# Output goes to a scratch dir so the committed notebook stays output-free.
notebook:
	rm -rf .nbbuild && mkdir -p .nbbuild
	$(PYTHON) -m nbconvert --to notebook --execute --output-dir .nbbuild notebooks/traxr_quickstart.ipynb

build:
	rm -rf dist .build-venv
	$(PYTHON) -m build
	$(PYTHON) -m venv .build-venv
	.build-venv/bin/python -m pip install --quiet dist/*.whl
	# Bare install: selfcheck degrades to the metrics-only check (no pandas).
	.build-venv/bin/python -c "import traxr; print('traxr', traxr.__version__); traxr.selfcheck()"

site:
	@echo "site: not yet built — lands in M6 (web/ + mkdocs build --strict)" >&2; exit 1

# The global Definition of Done: everything except mutation.
verify-all: install lint typecheck test cov property analyzer-goldens standalone-check golden external-golden selfcheck notebook build site

# Milestone boundaries + nightly.
verify-deep: verify-all mutation

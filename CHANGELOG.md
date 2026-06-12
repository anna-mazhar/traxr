# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Repo scaffolding: `src/` layout, `pyproject.toml` with extras
  (`document`, `openai`, `langgraph`, `viz`, `pandas`, `report`, `dev`),
  ruff/mypy/pytest configuration.
- `traxr.errors`: typed exception and warning hierarchy
  (base `TraxrError` / `TraxrWarning`).
- `Makefile` command surface per the execution contract; targets for later
  milestones fail with a clear "not yet built — lands in MN" message.
- Test suite scaffold (`tests/{unit,integration,e2e,property,fixtures}/`)
  with unit tests for the error hierarchy.
- Analyzer golden fixtures (`tests/fixtures/analyzer_goldens/`): inputs and
  ground-truth outputs of the trace divergence analyzer on 5 hand-crafted
  trace pairs.
- Repo practice: `PROGRESS.md`, `ROADMAP.md`, `CONTRIBUTING.md`,
  issue/PR templates, CI workflow skeleton (lint + typecheck + test matrix
  + `[langgraph]` job stub).

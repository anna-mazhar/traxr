# Contributing to Traxr

Thanks for your interest in contributing!

Every pull request needs a human who has read, understood, and can stand behind the code — please don't point an autonomous agent at this repo.

## Development setup

Requires Python >= 3.10.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install        # pip install -e ".[dev,document,openai,langgraph,viz]"
```

## Workflow

- Branch from `main`; open a PR against `main`.
- Use [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `test:`, `chore:`, `ci:`, `docs:`).
- Keep commits small and readable.
- Update `CHANGELOG.md` (Unreleased section) for user-visible changes.

## Gates

Before pushing, run the relevant Makefile targets:

```bash
make lint typecheck test
```

`make verify-all` is the global Definition of Done (some targets are stubbed
until their milestone lands — see `Makefile` and `PROGRESS.md`).

Never weaken a gate: no lowering coverage thresholds, relaxing assertions,
blanket ignores, or stubbing a gate to pass. If a gate is wrong, fix it
deliberately and record why in `PROGRESS.md`.

## Tests

Suite layout: `tests/{unit,integration,e2e,property,fixtures}/`.
`tests/fixtures/analyzer_goldens/` contains committed ground-truth analyzer
outputs — never regenerate or edit them by hand; they are the
behavior-preservation oracle for refactors.

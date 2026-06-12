# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `traxr.llm`: public `LLMClient` protocol, `LLMResponse`/`LLMToolResponse`
  types, `OpenAICompatibleClient` (extracted `OpenAIClient` + `base_url`;
  lazy `openai` import; `temperature=0` + fixed seed defaults), and the
  `DeterministicLLMStub` with five scripted scenarios (`identity`,
  `wrong_answer`, `reroute`, `halt_early`, `loop`) — zero keys, zero network.
- `traxr.mas`: the internal reference multi-agent system
  (core/agents/routing/retrieval/tools/provenance/planning), wired onto the M1 trace collector; the runner now emits `tool_failure`
  and `agent_halt` events.
- `traxr.agents.builtin`: `builtin_agent(llm=...)` factory facade over the
  EpisodeRunner with per-file-type tool wiring, opt-in web tools
  (default OFF), and the `injected_pdf_content` consumption path.

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

### Changed
- `python_tool` hardened: LLM-written code runs in a subprocess with a
  timeout instead of a raw in-process `exec()`; hanging scripts now fail
  cleanly.
- `make cov` uses path-based coverage sources (dotted sources unloaded the
  numpy C extension via coverage's module isolation, breaking later
  pandas imports).

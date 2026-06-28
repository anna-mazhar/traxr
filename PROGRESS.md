# Traxr build progress

Mirrors the build plan's milestone checklist. Tick only when the milestone's
exit gate is green. Open blockers are recorded at the bottom (dated, with
symptom/hypothesis/attempts/failing command).

## M0 — Scaffolding + repo practice
- [x] Repo skeleton: `src/traxr`, `pyproject.toml` with extras, ruff/mypy/pytest config, CI skeleton (incl. `[langgraph]` job stub), `errors.py`, `Makefile`, `PROGRESS.md`, `ROADMAP.md` (mirror of the backlog), `CHANGELOG.md`, CONTRIBUTING + issue/PR templates.
- [x] **Analyzer golden fixtures (FIRST task — purely analytical, no LLM/keys):** committed 5 hand-crafted trace pairs (identical, single edit, known-`t*`, empty, fully disjoint) with their ground-truth analyzer **inputs AND outputs as JSON** to `tests/fixtures/analyzer_goldens/`. M1 asserts the analyzer reproduces them exactly. Stubbed `scripts/check_analyzer_goldens.py` with a clear `"not yet built — lands in M1"` failure.

M0 notes / assumptions:
- Version is `0.1.0.dev0` (PEP 440 form of `0.1.0-dev`); `1.0.0` lands at M6.
- `[report]` extra membership is provisional (`traxr[pandas,viz]`) — the plan
  does not enumerate it; refine when `results.py` lands in M4.
- Goldens "empty" case = populated clean vs empty perturbed trace (EMPTY-run
  scenario; exercises the `n==0` edit-distance branch), not both-empty.
- `make build` smokes `import traxr` + version; M4 upgrades it to
  `traxr.selfcheck()` per the execution contract.

## M1 — Extract + refactor the analytical core (registry work happens HERE)
- [x] `divergence/` → `traxr/trace/{events,collector}.py` + `traxr/metrics/analyzer.py`; add `tool_failure`/`agent_halt`; open `event_type` validation; add `emit()` lock.
- [x] Build `trace/registry.py` and refactor `_event_to_signature`, `_classify_divergence`, `STRUCTURAL_DIVERGENCE_TYPES`, `_key_field_compare` to registry dispatch; register built-in types with existing behavior; add external types (`llm_call`, `tool_request`, `tool_result`, `agent_error`) with signatures + divergence classifiers. Analyzer-golden gate proves built-in behavior unchanged.
- [x] `classify_manifestation` → `traxr/metrics/manifest.py`; `metrics/` → `traxr/metrics/cost.py`. Tests: categories 3, 4, 6.

M1 notes:
- Analyzer-golden gate: PASS — all 5 fixture cases reproduce the committed
  golden outputs exactly.
- Coverage on `traxr.trace` + `traxr.metrics`: 99.45% (gate: 90%). Full
  `make cov` still fails on `traxr.perturb` — that module lands in M2.

## M2 — Extract perturbation + data; build PDF in-place
- [x] `perturbations/` → `traxr/perturb/`; `file_handler`+`file_inspector` → `traxr/data/loader.py`; DOCUMENT = PDF/TXT/MD (update `UnsupportedModalityError` messages).
- [x] Build `perturb/pdf_inplace.py` (span-level selection; redact+reinsert; `PAGE_REMOVAL`/`PAGE_SHUFFLE`; overflow/skip handling; metadata scrub). Guard tests T1–T4 green per operator.
- [x] Build `traxr/data/sources.py` + `traxr/perturb/matrix.py` (agent-kind-aware operator enumeration). Tests: categories 1, 2.

## M3 — Extract the reference agent + build the LLM boundary
- [ ] `core/`, `agents/`, `routing/`, `retrieval/`, `tools/`, `provenance/`, `planning/` → `traxr/mas/` (three-commit discipline; prints→logging; utcnow fixed; web tools behind flag).
- [ ] Harden `python_tool`: subprocess + timeout; `enable_python_tool` flag; hanging-script-times-out test.
- [ ] `traxr/llm/`: `LLMClient` protocol, types, `OpenAICompatibleClient` (+`base_url`), `DeterministicLLMStub`; validate every stub scenario produces its intended trace shape before M4.
- [ ] `agents/builtin.py`: `builtin_agent(llm=...)` factory facade over `EpisodeRunner`. Preserve `PDFTool.inject_perturbed_content()`.
- [ ] Smoke: reference agent answers a fixture CSV question end-to-end under the stub, well-formed trace. Tests: categories 5, 9.

## M3b — Capture layer + AgentRunner contract
- [ ] `capture/context.py` (contextvar binding, thread fallback, Tier-1 suppression flag), `capture/openai_wrap.py` (`instrument()`: sync + async + streaming delta-reassembly + usage injection + `max_llm_calls_per_run` budget), `capture/patch.py` (`patch_openai()`), `traxr.emit()` escape hatch.
- [ ] `agents/task.py`: `Task`, `AgentRunner`, `AgentContractError`; concurrency detection → `ConcurrentTraceWarning`.
- [ ] Fixture external agent on `httpx.MockTransport`. Gate: category 7 + 8 tests green, fully offline.

## M4 — Experiment runner + results + CLI
- [ ] `traxr/experiment.py`: `agent`/`agent_factory`/`llm` resolution; run loop (fresh temp dirs, original basenames, contextvar binding, harness-emitted `final_answer`, noise-floor with external default = 1, `dry_run`, caps); wire the built-in injection producer and the external pdf_inplace path into perturbation delivery.
- [ ] `traxr/results.py` (incl. `order_nondeterministic`, noise-floor caveat in `summary()`), `traxr/cli.py` (`--agent module:callable`, `--dry-run`), `python -m traxr.selfcheck`.
- [ ] Tests: categories 10, 12, 13, 14 (both goldens; selfcheck).

## M4b — LangGraph adapter
- [ ] `agents/langgraph.py`: `BaseCallbackHandler` mapping; `from_langgraph()` with `input_builder`/`output_extractor`; double-count suppression; version pins.
- [ ] Fixture graph (`GenericFakeChatModel`); dedicated CI job for the `[langgraph]` extra. Tests: category 11.

## M5 — Deep tests + SDK polish + Colab
- [ ] Coverage/property/negative-corpus gates green; mutation baseline.
- [ ] Logo, README (badges, BYO-agent quickstart hero, operator table per agent kind, "is my agent traceable?" section, security section, LLM-connection guide, roadmap link), Colab notebook (incl. BYO-agent + LangGraph cells), API docstrings.

## M6 — Website + v1.0.0 release (FINAL)
- [ ] Landing page (`web/`) + MkDocs docs (`--strict`, mkdocstrings), shared logo/palette, deploy assembly to GitHub Pages.
- [ ] Tag `v1.0.0`, GitHub Release with notes + wheel, CHANGELOG entry, social preview set.

## Open blockers

None.

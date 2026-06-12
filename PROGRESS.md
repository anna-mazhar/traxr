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
- [x] `core/`, `agents/`, `routing/`, `retrieval/`, `tools/`, `provenance/`, `planning/` → `traxr/mas/` (three-commit discipline; prints→logging; utcnow fixed; web tools behind flag).
- [x] Harden `python_tool`: subprocess + timeout; `enable_python_tool` flag; hanging-script-times-out test.
- [x] `traxr/llm/`: `LLMClient` protocol, types, `OpenAICompatibleClient` (+`base_url`), `DeterministicLLMStub`; validate every stub scenario produces its intended trace shape before M4.
- [x] `agents/builtin.py`: `builtin_agent(llm=...)` factory facade over `EpisodeRunner`. Preserve `PDFTool.inject_perturbed_content()`.
- [x] Smoke: reference agent answers a fixture CSV question end-to-end under the stub, well-formed trace. Tests: categories 5, 9.

M3 notes:
- Three-commit extraction discipline: verbatim copy → mechanical fixes
  (imports/logging/utcnow + lint/mypy config activation) → behavioral
  changes (python_tool subprocess sandbox, tool_failure/agent_halt
  emission, Tinker leave-behind).
- All five stub scenarios validated against their intended trace shapes
  (identity / wrong_answer / reroute / halt_early / loop) in
  `tests/integration/test_stub_scenarios.py`.
- The runner now actually emits `tool_failure` + `agent_halt` (documented
  in the M1 schema but never fired by the source).
- `mas/utils/code_extraction.py` carried along (agents dependency not in
  the reuse ledger); `llm/tinker_client.py` left behind as planned.
- `make cov` legs use **path-based** `--cov` sources: dotted sources make
  coverage import `traxr` inside its `sys_modules_saved` block, unloading
  the transitively-imported numpy C extension (via fitz) and breaking
  later pandas imports. Coverage: metrics/perturb/trace 97%, agents 94%,
  total 95%.

## M3b — Capture layer + AgentRunner contract
- [x] `capture/context.py` (contextvar binding, thread fallback, Tier-1 suppression flag), `capture/openai_wrap.py` (`instrument()`: sync + async + streaming delta-reassembly + usage injection + `max_llm_calls_per_run` budget), `capture/patch.py` (`patch_openai()`), `traxr.emit()` escape hatch.
- [x] `agents/task.py`: `Task`, `AgentRunner`, `AgentContractError`; concurrency detection → `ConcurrentTraceWarning`.
- [x] Fixture external agent on `httpx.MockTransport`. Gate: category 7 + 8 tests green, fully offline.

M3b notes:
- One wrapper factory serves sync, async, and `patch_openai()`: the SDK's
  async `create` is a plain `def` returning an awaitable (not a coroutine
  function), so sync/async is decided per call via `inspect.isawaitable`
  on the result.
- Run binding lives in `CaptureSession` (`capture/context.py`): step
  counter, `tool_call_id → (name, step)` map, budget, store-content flag,
  and concurrency detection (multi-thread emission OR overlapping
  in-flight calls → `ConcurrentTraceWarning` once + `concurrent_detected`
  for M4's `order_nondeterministic`). Resolution: contextvar → process
  global (user-thread fallback) → None (passthrough outside runs).
- `tool_result` events are emitted at the step of the `llm_call` that
  requested them (joined via the call-id map), keeping per-step grouping
  faithful even though they surface one request later.
- `agents/task.py` also ships `invoke_agent()` — the single-run primitive
  M4's `Experiment` composes: binds the session, validates the `str`
  return, emits `final_answer`, converts crashes to `agent_error` and
  re-raises (record-vs-raise stays an M4 `on_run_error` concern, as does
  `agent`/`agent_factory`/`llm` XOR validation).
- Mock-client doubles + the fixture external agent live in
  `tests/unit/_openai_mock.py` (real `openai` clients over
  `httpx.MockTransport`, SSE bodies for streaming) — promote alongside the
  M4 external golden if e2e needs them.
- Gates: 486 tests (35 new), zero network; coverage capture+agents 94%
  (gate 85), metrics/perturb/trace 97.55%; analyzer-goldens 5/5;
  standalone-check PASS; mypy strict clean on `capture/` + `agents/task.py`.

## M4 — Experiment runner + results + CLI
- [x] `traxr/experiment.py`: `agent`/`agent_factory`/`llm` resolution; run loop (fresh temp dirs, original basenames, contextvar binding, harness-emitted `final_answer`, noise-floor with external default = 1, `dry_run`, caps); wire the built-in injection producer and the external pdf_inplace path into perturbation delivery.
- [x] `traxr/results.py` (incl. `order_nondeterministic`, noise-floor caveat in `summary()`), `traxr/cli.py` (`--agent module:callable`, `--dry-run`), `python -m traxr.selfcheck`.
- [x] Tests: categories 10, 12, 13, 14 (both goldens; selfcheck).

M4 notes:
- New modules: `experiment.py`, `results.py`, `scoring.py`, `cli.py`
  (`traxr` console script in pyproject), `selfcheck.py` (degrades to a
  metrics-only check on a bare install, so the wheel smoke passes without
  pandas). Public API += Experiment/ExperimentConfig/ExperimentResults/
  PairResult/selfcheck (selfcheck via lazy `__getattr__` to keep
  `python -m traxr.selfcheck` warning-free).
- Surgical extensions, not rewrites: `invoke_agent(session=...)` lets the
  runner read `concurrent_detected` per run; `BuiltinAgent.last_cost`
  exposes the per-run token CostProxy. External token counts come from
  captured `llm_call` usage.
- Injection delivery extracts PDF text via the data loader (the engine's
  `apply_from_file` reads raw bytes — wrong input for text operators).
- Run statuses: ok / crashed / empty / skipped. "Empty" for external runs
  means zero captured `llm_call`s (the harness's own final_answer event
  doesn't count). Skipped perturbations never invoke the agent.
- Goldens (`tests/fixtures/goldens/`): byte-stable run-vs-run, plus a
  committed-snapshot compare with `fingerprint.environment` normalized
  (version strings vary across CI matrix entries).
- Determinism fix for goldens: memory entry IDs are now content-derived
  hashes instead of random (`mas/core/state.py`) — also makes paired
  memory-event comparison meaningful.
- Fixed a latent test bug exposed by the new suites: an M2 pdf_inplace
  test restored a saved `staticmethod` as a bare function, breaking
  `_fit_fontsize` for any later caller in the same session.
- Gates: 562 tests (76 new); coverage 97.49/95.06/93.75 vs gates 90/85/75;
  selfcheck/golden/external-golden/build un-stubbed and green;
  analyzer-goldens 5/5; standalone-check PASS.

## M4b — LangGraph adapter
- [x] `agents/langgraph.py`: `BaseCallbackHandler` mapping; `from_langgraph()` with `input_builder`/`output_extractor`; double-count suppression; version pins.
- [x] Fixture graph (`GenericFakeChatModel`); dedicated CI job for the `[langgraph]` extra. Tests: category 11.

M4b notes:
- Mapping verified empirically against langgraph 1.2.4 / langchain-core
  1.4.6 before coding: node entry arrives as `on_chain_start` with
  `langgraph_node`/`langgraph_step` metadata (deduped per
  node+step+checkpoint_ns — inner runnables inherit the metadata); model
  identity comes from `invocation_params` at `*_start` time; usage from
  `message.usage_metadata` (fallback `llm_output.token_usage`, else None
  + TokenUnavailableWarning).
- Version pins set to the CI-exercised range (`langchain-core>=1.0,<2.0`,
  `langgraph>=1.0,<2.0`) — the build plan's `<1.0` figure predates
  langgraph 1.0.
- Concurrency heuristic refined: LangGraph fires tool callbacks on
  executor threads even for sequential execution, so handler emissions
  use the new `CaptureSession.emit(count_thread=False)` opt-out and the
  adapter flags real parallelism itself (overlapping in-flight LLM runs
  → `session.note_concurrency()`). Session step floor now advances on
  explicit step_num so the harness final_answer lands at the last step.
- `max_llm_calls_per_run` is NOT enforced for LangGraph runs in v1
  (Tier 0 is suppressed; the budget lives in the Tier 0 wrapper) —
  documented in the module docstring.
- CI `langgraph` job installs only `[dev,langgraph]` (proves no other
  extras needed); the one openai-dependent test skips there. Verified in
  a fresh venv: 10 passed, 1 skipped.
- Gates: 573 tests (11 new); coverage 97.49/94.88/93.83; goldens +
  selfcheck unchanged-green; standalone-check PASS.

## M5 — Deep tests + SDK polish + Colab
- [x] Coverage/property/negative-corpus gates green; mutation baseline.
- [x] Logo, README (badges, BYO-agent quickstart hero, operator table per agent kind, "is my agent traceable?" section, security section, LLM-connection guide, roadmap link), Colab notebook (incl. BYO-agent + LangGraph cells), API docstrings.

M5 notes:
- **Mutation baseline (make mutation, [tool.mutmut] in pyproject):**
  2799 mutants over `metrics/` + `perturb/` (image/audio research modules
  excluded — unexported backlog code, mirroring the coverage gates):
  1787 killed/timeout, 1010 survived, 2 uncovered → **63.9%**. Survivors
  concentrate in perturbation operators (939/1010); the paper-critical
  `metrics/` math has 76. The 70% gate from the build plan is the ratchet
  target, tracked here; mutation stays verify-deep/nightly only.
- New `traxr/viz.py` ([viz] extra): per-pair d_norm bars (+ noise-floor
  line), t*/T histogram, manifestation breakdown — needed by the notebook.
- Logo: pixelated diverging-traces glyph (shared prefix, amber t*, blue
  clean run, coral corrupted branch); `assets/{logo,mark}{,-dark}.svg`;
  README uses a light/dark `<picture>`.
- README rewritten per plan §M5; security text split into SECURITY.md
  (5 points near-verbatim from the plan) + a README summary; honest-limits
  caveats inline in "Is my agent traceable?".
- Notebook `notebooks/traxr_quickstart.ipynb`: executes top-to-bottom with
  NO key (`make notebook` via nbconvert into a scratch dir; nbconvert +
  ipykernel added to dev extras). Real-model cell gated on
  OPENAI_API_KEY; LangGraph cell is keyless (fake chat model) and
  import-gated. Ruff lints notebook cells too.
- Doctests: examples added to `scoring.py`; `tests/unit/test_doctests.py`
  runs doctest over the curated modules.
- Gates: 584 tests; coverage 97.36/94.88/93.93; notebook/property/
  analyzer-goldens/selfcheck/goldens/standalone-check green. README
  quickstart fresh-venv verification recorded below after push.

## M6 — Website + v1.0.0 release (FINAL)
- [ ] Landing page (`web/`) + MkDocs docs (`--strict`, mkdocstrings), shared logo/palette, deploy assembly to GitHub Pages.
- [ ] Tag `v1.0.0`, GitHub Release with notes + wheel, CHANGELOG entry, social preview set.

## Open blockers

None.

## Maintenance
- 2026-06-12: analyzer golden fixtures renamed from `tests/fixtures/parity/`
  to `tests/fixtures/analyzer_goldens/` (`make analyzer-goldens`); new
  `make standalone-check` gate added to the Makefile and CI.

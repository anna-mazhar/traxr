# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-12

The first release: controlled-perturbation experiments for your own agent
and your own data, operationalizing *“Trace-Level Analysis of Information
Contamination in Multi-Agent Systems”* (Mazhar, Suri, Galhotra) as an SDK.

### Added

**Experiments**

- `traxr.Experiment`: paired clean/perturbed runs over `agent=` (any
  `(Task) -> str` callable), `agent_factory=` (fresh state per run), or
  `llm=` (the built-in reference agent). Agent-kind-aware permutation
  matrix per file; fresh temp dirs with original basenames; identically
  re-derived seeds (the controlled-variable invariant); measured noise
  floor (default 1 clean re-run for external agents); `run(dry_run=True)`
  prints the full plan with zero LLM calls; `on_run_error="record"|"raise"`
  with crash/empty/skip statuses and partial-trace metrics.
- `traxr.ExperimentResults` / `traxr.PairResult`: `d_norm`, `t*` (+
  normalized position), divergence type, control-flow changes,
  manifestation (+ paper group), `task_success`, `answer_changed`,
  recovery, token overhead, `within_noise_floor`,
  `order_nondeterministic`; aggregates; canonical timestamp-free
  `to_json()` (byte-stable for deterministic experiments),
  `to_dataframe()`, `to_report()`, `summary()` with an unmeasured-floor
  caveat.

**Capture**

- Tier 0: `traxr.instrument(client)` wraps OpenAI-SDK
  `chat.completions.create` (sync, async, streaming with delta
  reassembly, tool calls, usage capture) on any OpenAI-compatible
  endpoint; `traxr.capture.patch_openai()` class-level fallback;
  `max_llm_calls_per_run` enforced inside the wrapper; hash-only payloads
  by default (`store_llm_content` opt-in); concurrency detection with
  `order_nondeterministic` flagging and `require_sequential`.
- Tier 1: `traxr.from_langgraph(graph)` — LangGraph callbacks mapped onto
  the existing event vocabulary (node entry → routing events, full tool
  success/failure fidelity), with Tier 0 double-count suppression.
- `traxr.emit()` manual escape hatch and `traxr.register_signature()` to
  lift custom event types into the structural metrics.

**Analysis**

- Trace model + thread-safe collector; registry-driven event vocabulary
  (built-in, external, and custom types).
- Divergence analyzer: normalized edit distance over structural
  signatures, first-divergence point, control-flow change counts; pinned
  by committed golden fixtures.
- Manifestation taxonomy (9 fine categories, 4 paper groups), answer
  scoring, token-cost comparison.

**Perturbation + data**

- Operators for tabular (7), text (7), and PDF inputs; surgical in-place
  PDF editing (PyMuPDF) preserving extraction fidelity, with
  `page_removal`/`page_shuffle`; built-in-agent content-injection delivery
  for whole-text-flow PDF operators; deterministic per-permutation seeds.
- Data loaders for CSV/XLSX/TXT/MD/PDF with typed failure modes.

**Reference agent + LLM boundary**

- Bundled multi-agent reference system (`llm=...` path) with subprocess-
  sandboxed `python_tool`, opt-in web tools, and per-file-type tool wiring.
- `LLMClient` protocol, `OpenAICompatibleClient` (any OpenAI-compatible
  `base_url`), and the offline `DeterministicLLMStub` (five scripted
  scenarios) powering demos, goldens, and `python -m traxr.selfcheck`.

**Tooling, docs, site**

- `traxr` CLI: `run` (`--agent module:callable` or `--model/--base-url`),
  `operators`, `selfcheck`; results JSON export.
- Quickstart notebook that executes top-to-bottom with no API key;
  `traxr.viz` plots; doctests on curated modules; mutation-testing
  baseline; typed error/warning hierarchy; SECURITY.md.
- Landing page + MkDocs documentation site with Pages deployment, and
  this tag-triggered release workflow.

[1.0.0]: https://github.com/anna-mazhar/traxr/releases/tag/v1.0.0

# Traxr roadmap — deferred feature backlog

Deferred to v2/v3 (triage order TBD). Mirrors the build plan's backlog;
created in M0 and kept current.

1. **Localhost LLM proxy** — the path to non-Python/polyglot agents and non-OpenAI SDKs; a real server (TLS upstream, SSE passthrough, credential forwarding — **must never log `Authorization`**). *(The headline v2 candidate.)*
2. **OpenTelemetry trace ingester** — zero-instrumentation path for teams already emitting OTel spans (LangSmith/Weave/OpenInference users).
3. **More framework adapters** — AutoGen, CrewAI; reusable adapter conformance test suite.
4. **Per-lane trace alignment** — `traxr.lane()` contextvar + per-lane divergence analysis for concurrent agents.
5. **Subprocess isolation of agent runs** — fresh-process execution per run (statefulness + crash containment).
6. **Full-text-flow PDF operators for external agents** — `OCR_NOISE`/`PARAGRAPH_SHUFFLE`/`ENCODING_ERROR` via PDF rebuild (accepting the layout-confound trade-off, or smarter span-tiling).
7. **Image + audio modalities** — operators exist; need tests, extras, promotion.
8. **docx / pptx documents** — loader support exists; needs operator validation + tests.
9. **PyPI publishing** — name reservation, trusted publishing via CI on tag.
10. **Perturbation composition** — multi-operator/multi-locus stacks; cross-modal; adaptive intensity sweeps.
11. **Provenance as a public feature** — surface `TaintTracker`/`ProvenanceTracker` for the built-in agent; contamination-origin attribution (backtrack from `t*` — the paper's open direction).
12. **LLM-judge scorer** extra.
13. **GAIA public benchmark mode** (HF-gated, documented).
14. **More LLM clients** — native Anthropic client, Tinker revival, retry/rate-limit middleware.
15. **New modality types** — `SLIDE_DECK`, `VIDEO`, `SEMI_STRUCTURED` (JSON/YAML/logs), `WEB_PAGE`/HTML.
16. **Parallel experiment runner** (per-perturbation concurrency; needs context-safe trace collection).
17. **Marketing-grade landing animations / interactive divergence visualizer** on the website.

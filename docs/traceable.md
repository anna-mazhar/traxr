# Is my agent traceable?

The honest headline: traxr v1 measures **any Python agent that talks to an
OpenAI-compatible endpoint via the OpenAI SDK** — not "any agent".

## Covered

- **`openai.OpenAI` / `openai.AsyncOpenAI` clients** you can pass into your
  agent: wrap once with `traxr.instrument(client)`. Sync, async, and
  streaming calls (with tool-call delta reassembly) all become trace
  events. Works against any OpenAI-compatible endpoint — OpenAI, Azure,
  Ollama, vLLM, Together, Groq, OpenRouter, a proxied Anthropic.
- **Clients constructed inside code you can't change**: the
  `traxr.capture.patch_openai()` context manager patches the SDK at class
  level for the duration of the experiment.
- **LangGraph graphs**: `traxr.from_langgraph(graph)` captures node
  transitions (as routing events — reroute metrics work unchanged), tool
  calls with full success/failure fidelity, and LLM calls, via
  `langchain-core` callbacks.
- **Anything you emit yourself**: `traxr.emit("my_event", {...})` is the
  manual escape hatch; upgrade custom event types into the structural
  metrics with `traxr.register_signature()`.

## Not covered (v1)

- Other provider SDKs (native Anthropic, Gemini, …)
- Raw HTTP calls
- The OpenAI Responses / Assistants APIs
- LLM calls made in subprocesses

Runs that capture nothing are flagged with `EmptyTraceWarning` and status
`empty` — never silently reported as zero divergence. A localhost proxy and
OpenTelemetry ingestion are the roadmap answers for the rest.

## Caveats that keep the numbers honest

**External traces are coarser than built-in traces.** Tier 0 sees LLM
calls, tool requests, and tool results — no memory or retrieval events, and
tool success/failure is unknown at the SDK boundary. `d_norm` and `t*`
remain valid (the metrics are vocabulary-agnostic), but **built-in and
external values are not cross-comparable**.

**Concurrency degrades comparability.** Parallel LLM calls interleave
nondeterministically; scheduling noise inflates `d_norm`. traxr detects it
(`ConcurrentTraceWarning`, `order_nondeterministic` on the pair), the
[noise floor](metrics.md#the-noise-floor) absorbs it empirically, and
`require_sequential=True` fails fast instead.

**Budget enforcement is Tier 0 only.** `max_llm_calls_per_run` is enforced
inside the instrument wrapper. LangGraph (Tier 1) runs suppress Tier 0 to
avoid double counting, so the budget is not enforced there in v1.

# Security

Traxr runs **your agent** against **deliberately corrupted copies of your
data**. That combination deserves clear-eyed handling. Read this before
running experiments with an agent that has real tools.

## 1. Perturbed data is an injection-adjacent vector into your agent

Traxr's operators corrupt content rather than inject instructions — but
corrupted content can still drive an LLM agent with side-effectful tools
(shell, email, payments, file writes) into pathological actions, and
**Traxr cannot sandbox your agent**.

Recommendations:

- Run experiments with side-effectful tools disabled, or inside a
  container/VM.
- Use `Experiment.run(dry_run=True)` first — it enumerates every run before
  any agent executes.
- The bundled reference agent's `python_tool` runs model-written code in a
  subprocess with a timeout (and can be disabled with
  `enable_python_tool=False`), but a container/VM is still recommended when
  feeding it untrusted data.

## 2. Traces are sensitive data at rest

By default, trace payloads carry **hashes only** (`store_llm_content=False`).
Opting into `store_llm_content=True` writes raw prompts/completions/tool
arguments into the trace.

Documented carve-out: **raw final-answer strings are always stored** —
scoring and `answer_changed` need them.

`keep_artifacts=True` preserves the per-run temp dirs, which contain
perturbed copies of your input files.

## 3. Credentials

Tier 0 capture wraps the OpenAI SDK **above** HTTP: it never sees, stores,
or logs `Authorization` headers or API keys. (Standing requirement for any
future proxy-based capture: never log `Authorization`.)

## 4. Cost containment — what Traxr can and cannot bound

Traxr **cannot bound an external agent's spend**. What v1 gives you:

- `max_llm_calls_per_run` — enforced inside the Tier 0 wrapper
  (`RunBudgetExceeded` is raised before the over-budget call goes out).
  Not enforced for LangGraph (Tier 1) runs in v1.
- Live token totals per run, from captured usage.
- `dry_run=True` — the full execution plan with zero LLM calls.
- Built-in agent only: `max_steps` / `max_tokens` hard caps.

An agent making raw HTTP calls bypasses even the budget. Plan experiments
on a key with a spending limit.

## Reporting a vulnerability

Open a GitHub security advisory or issue on this repository.

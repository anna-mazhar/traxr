# Security

traxr runs **your agent** against **deliberately corrupted copies of your
data**. Read this before running an agent that has real tools. (This page
mirrors [`SECURITY.md`](https://github.com/anna-mazhar/traxr/blob/main/SECURITY.md)
in the repository.)

## Perturbed data is an injection-adjacent vector

traxr's operators corrupt content rather than inject instructions, but
corrupted content can still drive an LLM agent with side-effectful tools
(shell, email, payments, file writes) into pathological actions, and
**traxr cannot sandbox your agent**.

- Run experiments with side-effectful tools disabled, or inside a
  container/VM.
- `Experiment.run(dry_run=True)` enumerates every run before any agent
  executes.
- The built-in reference agent's `python_tool` runs model-written code in a
  subprocess with a timeout, and can be disabled
  (`enable_python_tool=False`); a container is still recommended for
  untrusted data.

## Traces are sensitive data at rest

Trace payloads carry **hashes only** by default
(`store_llm_content=False`). Opting in stores raw prompts, completions, and
tool arguments. Documented carve-out: **raw final-answer strings are always
stored**, since scoring and `answer_changed` need them. `keep_artifacts=True`
preserves perturbed copies of your input files.

## Credentials

Tier 0 capture wraps the OpenAI SDK **above** HTTP: it never sees, stores,
or logs `Authorization` headers or API keys.

## Cost containment, honestly

traxr cannot bound an external agent's spend. What v1 gives you:

- `ExperimentConfig(max_llm_calls_per_run=...)` (default 50, external agents),
  enforced inside the Tier 0 wrapper: `RunBudgetExceeded` is raised *before*
  the over-budget call goes out. Set it to `None` to disable the cap.
- `OpenAICompatibleClient(max_retries=...)` (default 2, the OpenAI SDK default)
  bounds the built-in agent's transient-failure retries; `0` disables them.
- Live token totals per run, from captured usage.
- `dry_run=True`: the full plan with zero LLM calls.
- Built-in agent only: `max_steps` / `max_tokens` hard caps.

From the CLI: `traxr run ... --max-llm-calls 20 --max-retries 0`.

An agent making raw HTTP calls bypasses even the budget. Use a key with a
spending limit.

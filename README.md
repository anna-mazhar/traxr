<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="traxr" width="280">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/anna-mazhar/traxr/actions/workflows/ci.yml"><img src="https://github.com/anna-mazhar/traxr/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python versions">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
  <a href="https://dl.acm.org/doi/10.1145/3786335.3813147"><img src="https://img.shields.io/badge/paper-CAIS%202026-8b5cf6" alt="Paper"></a>
</p>

**Evaluate multi-agent systems beyond final-answer accuracy.** A multi-agent
system can land the right answer through the wrong process — and answer-level
metrics never see it. Traxr evaluates the **execution trace itself**: point
your own agent at your own data, run paired experiments, and measure how its
behavior diverged — how much (`d_norm`), where it started (`t*`), how it
manifested, and what it cost in tokens. Controlled input perturbation is the
instrument; the trace is the measurement.

Traxr operationalizes the paper *“Trace-Level Analysis of Information
Contamination in Multi-Agent Systems”* ([CAIS 2026](https://dl.acm.org/doi/10.1145/3786335.3813147);
Mazhar, Suri, Galhotra) as an SDK, for any Python agent that talks to an
OpenAI-compatible endpoint through the OpenAI SDK.

## Install

```bash
pip install "traxr[document,openai,pandas] @ git+https://github.com/anna-mazhar/traxr.git@main"
```

Extras: `[document]` (PDF/XLSX support), `[openai]` (the built-in reference
agent's client), `[pandas]` (DataFrame export; also required by the built-in
reference agent), `[langgraph]` (LangGraph adapter), `[viz]` (plots).
External agents with their own OpenAI client need **no extras at all**.

## Quickstart — bring your own agent

Your agent is any callable `(Task) -> str`. Wrap its OpenAI client with
`traxr.instrument()` and every `chat.completions` call (sync, async, or
streaming, including tool calls) is captured into the trace:

```python
import openai, traxr

client = traxr.instrument(openai.OpenAI())  # same client, now traced

def my_agent(task: traxr.Task) -> str:
    data = task.files[0].read_text()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"{task.question}\n\n{data}"}],
    )
    return response.choices[0].message.content or ""

experiment = traxr.Experiment(
    files="examples/sales.csv",
    question="Which region had the highest Q3 revenue?",
    expected_answer="EMEA",
    agent=my_agent,
)
experiment.run(dry_run=True)   # the full plan — zero LLM calls, zero spend
results = experiment.run()     # baseline + perturbed runs (+ noise floor)
print(results.summary())
results.to_json("results.json")
```

Each perturbation gets a fresh temp dir with **original file basenames** —
your agent can never tell which condition it is in. Stateful agents (memory,
vector stores) should use `agent_factory=` instead of `agent=` so every run
starts fresh.

### Scoring free-text answers

A real agent answers in full sentences, so the default scorer
(`check_answer_match`, exact normalized string equality) won't match a bare
`expected_answer` against a verbose reply. Bring your own via
`ExperimentConfig(scorer=...)` — e.g. the built-in `llm_judge_match` for
semantic matching (non-deterministic, costs an extra LLM call):

```python
from traxr import ExperimentConfig
from traxr.scoring import llm_judge_match

experiment = traxr.Experiment(
    files="examples/sales.csv",
    question="Which region had the highest Q3 revenue?",
    expected_answer="EMEA",
    agent=my_agent,
    config=ExperimentConfig(scorer=llm_judge_match),
)
```

See [docs/quickstart.md](docs/quickstart.md#scoring-free-text-answers) for
details and how to plug in your own deterministic scorer instead.

### No API key? Try the built-in reference agent + stub

```python
import traxr

experiment = traxr.Experiment(
    files="examples/sales.csv",
    question="Which region had the highest Q3 revenue?",
    expected_answer="EMEA",
    llm=traxr.DeterministicLLMStub(scenario="identity", final_answer="EMEA"),
)
results = experiment.run()
```

The bundled multi-agent reference system (`llm=...`) runs entirely offline
under the deterministic stub — it powers the demos, goldens, and
`python -m traxr.selfcheck`.

### LangGraph

```python
agent = traxr.from_langgraph(compiled_graph)   # Tier 1 capture via callbacks
experiment = traxr.Experiment(files="report.pdf", question="...", agent=agent)
```

Node transitions become routing events (so reroute metrics work), tool calls
keep success/failure fidelity, and double-counting with an instrumented
client is suppressed automatically. For non-messages-state graphs, pass
`input_builder=` / `output_extractor=`.

## How it works

1. **Perturb**: one operator is applied to a copy of your file (seeded,
   deterministic, single-variable).
2. **Paired runs**: your agent runs on the clean file, then on each
   perturbed copy — identical seeds, fresh temp dirs.
3. **Diverging traces**: each run's LLM/tool/routing events form a trace;
   paired traces are aligned and compared structurally.

**The metrics:**

| metric | meaning |
|---|---|
| `d_norm` | normalized edit distance between paired traces — 0 = identical process, 1 = completely different |
| `t*` (+ `t*/T`) | the step where divergence first appears, and how early in the run that is |
| manifestation | how the damage showed up: silent semantic corruption, strategy reroute, early termination, catastrophic failure, recovered, … |
| `token_overhead` | perturbed-run tokens / baseline tokens |
| noise floor | baseline-vs-itself `d_norm` from clean re-runs — divergence at or below it is indistinguishable from sampling noise (**defaults to 1 re-run for external agents; don't skip it**) |

## Perturbation operators (v1)

| input | operators | delivery |
|---|---|---|
| CSV / XLSX | `column_swap`, `label_corrupt`, `data_type_corrupt`, `row_duplicate`, `irrelevant_columns`, `unit_change`, `null_content` | file round-trip |
| TXT / MD | `ocr_noise`, `number_corruption`, `text_redaction`, `paragraph_shuffle`, `encoding_error`, `section_removal`, `null_content` | file round-trip |
| PDF (any agent) | `number_corruption`, `text_redaction`, `section_removal`, `page_removal`, `page_shuffle`, `null_content` | surgical in-place edits (extraction-fidelity preserving) |
| PDF (built-in agent only) | `ocr_noise`, `paragraph_shuffle`, `encoding_error` | extracted-content injection |

`traxr operators` prints the live catalog.

## Is my agent traceable?

Tier 0 capture sees **OpenAI-SDK `chat.completions` calls** — that's the
honest scope. You're covered if your agent:

- uses `openai.OpenAI` / `openai.AsyncOpenAI` against any OpenAI-compatible
  endpoint (OpenAI, Azure, Ollama, vLLM, Together, Groq, OpenRouter, …) and
  you can pass the instrumented client in,
- constructs clients internally — use the `traxr.capture.patch_openai()`
  context manager instead, or
- is a LangGraph graph (`traxr.from_langgraph`).

**Not captured (yet):** other provider SDKs, raw HTTP, the
Responses/Assistants APIs, subprocess-spawned LLM calls. Runs that capture
nothing are flagged (`EmptyTraceWarning`) rather than silently reported as
zero divergence. A local proxy and OTel ingestion are on the
[roadmap](ROADMAP.md). You can also hand-place events with `traxr.emit()`.

Two caveats that keep the numbers honest:

- External traces are **coarser** than built-in-agent traces (no
  memory/retrieval events; tool success unknown at Tier 0). `d_norm`/`t*`
  remain valid, but built-in and external values are **not
  cross-comparable**.
- Concurrent LLM calls make event order scheduling-dependent. Traxr detects
  this (`order_nondeterministic` + `ConcurrentTraceWarning`), the noise
  floor absorbs it empirically, and `require_sequential=True` fails fast
  instead.

## Cost, honestly

One experiment = 1 baseline + up to ~7 perturbation runs (+ noise-floor
re-runs) of **your agent on your key**. Spend cannot be estimated up front,
so Traxr gives you enforcement instead: `run(dry_run=True)` prints the full
plan with zero LLM calls; `max_llm_calls_per_run` is enforced *inside* the
Tier 0 wrapper; live token totals print per run. See
[SECURITY.md](SECURITY.md) for what cannot be bounded.

## Security

Perturbed data is an injection-adjacent vector into your agent, and Traxr
cannot sandbox your agent: run experiments with side-effectful tools
disabled or inside a container/VM. Trace payloads are hash-only by default
(raw final answers are stored — scoring needs them). Tier 0 never touches
HTTP headers or keys. Full notes: [SECURITY.md](SECURITY.md).

## Bring any LLM provider (built-in agent)

The reference agent speaks to anything OpenAI-compatible:

```python
llm = traxr.OpenAICompatibleClient(model="llama3.1", base_url="http://localhost:11434/v1")
experiment = traxr.Experiment(files="examples/sales.csv", question="...", llm=llm)
```

For other providers, implement the two-method
[`traxr.LLMClient`](src/traxr/llm/protocol.py) protocol (`generate`,
`generate_with_tools`). External agents don't need any of this — they own
their LLM and are captured at the SDK boundary.

## CLI

```bash
traxr run --agent mypkg.agents:answer --file examples/sales.csv \
          --question "Which region won Q3?" --expected-answer EMEA \
          --out results.json
traxr run --model gpt-4o-mini --file report.pdf --question "..." --dry-run
traxr operators
traxr selfcheck
```

## Notebook

[`notebooks/traxr_quickstart.ipynb`](notebooks/traxr_quickstart.ipynb) runs
top-to-bottom **without an API key** (the real-model cells are skip-safe) —
open it in Colab to try Traxr in two minutes.

## Development

```bash
make install     # editable install with all extras
make test        # full offline suite
make verify-all  # the global definition of done
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [PROGRESS.md](PROGRESS.md), and the
deferred-feature [ROADMAP.md](ROADMAP.md).

## Citation

If Traxr is useful in your research, cite *“Trace-Level Analysis of
Information Contamination in Multi-Agent Systems”* (Mazhar, Suri, Galhotra).

## License

MIT

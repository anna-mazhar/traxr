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
system can land the right answer through the wrong process, and answer-level
metrics never see it. Traxr evaluates the **execution trace itself**: point your
agent at your data, run paired experiments, and measure how its behavior
diverged. How much (`d_norm`), where it started (`t*`), how it manifested, and
what it cost in tokens. Controlled input perturbation is the instrument; the
trace is the measurement.

Traxr operationalizes the paper *“Trace-Level Analysis of Information
Contamination in Multi-Agent Systems”* ([CAIS 2026](https://dl.acm.org/doi/10.1145/3786335.3813147);
Mazhar, Suri, Galhotra) as an SDK, for any Python agent that talks to an
OpenAI-compatible endpoint through the OpenAI SDK.

## Install

```bash
pip install "traxr[document,openai,pandas] @ git+https://github.com/anna-mazhar/traxr.git@main"
```

<details>
<summary>What the extras pull in</summary>

`[document]` (PDF/XLSX support), `[openai]` (the built-in reference agent's
client), `[pandas]` (DataFrame export; also required by the built-in reference
agent), `[langgraph]` (LangGraph adapter), `[viz]` (plots). **External agents
that bring an OpenAI client need no extras at all.**
</details>

## API keys

traxr never reads or stores a key of its own. Your instrumented agent uses its
own OpenAI client, which picks up `OPENAI_API_KEY` (or the `api_key=` you pass to
`openai.OpenAI(...)`) exactly as it always does. Tier 0 wraps the SDK above HTTP,
so it never sees keys or headers.

The built-in reference agent, the `llm_judge_match` scorer, and `traxr run
--model ...` go through `OpenAICompatibleClient`, which reads `OPENAI_API_KEY`
(or an explicit `api_key=`). For local OpenAI-compatible servers (Ollama, vLLM,
LM Studio) any non-empty string works: `OpenAICompatibleClient(base_url=...,
api_key="local")`.

## Quickstart

Bring your agent, point it at your data. Three steps: instrument, expose
agent-level structure, run.

### 1. Your agent

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
```

Each perturbation runs in a fresh temp dir under the **original basenames**, so
your agent can't tell which condition it's in. Stateful agents (memory, vector
stores) should pass `agent_factory=` instead of `agent=`.

### 2. Your multi-agent system

traxr captures your LLM traffic automatically, but it can't see *which* agent
is acting. `traxr.emit()` is the escape hatch: call it where your code knows who
is in charge (routing decisions, handoffs, memory reads) to expose agent-level
structure in the trace.

```python
# agent_name = the agent acting now; chosen_agent = who the orchestrator picked.
# Reuses the built-in "routing_decision" type, so it counts toward the metrics.
# Outside a Traxr run this is a no-op, so it's safe to leave in production code.
traxr.emit("routing_decision", {"chosen_agent": "researcher"}, agent_name="orchestrator")
```

<details>
<summary>Full multi-agent example (orchestrator + workers)</summary>

```python
import openai, traxr

client = traxr.instrument(openai.OpenAI())      # every LLM call below is captured

def call(prompt: str) -> str:                   # one LLM hop, shared by the agents
    r = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content or ""

# Worker agents tag their own output. They don't decide who runs next.
def planner(task, notes):
    traxr.emit("agent_output", {"action": "plan"}, agent_name="planner")
    return call(f"Plan how to answer: {task.question}")

def researcher(task, notes):
    traxr.emit("agent_output", {"action": "research"}, agent_name="researcher")
    return call(f"{task.question}\n\n{task.files[0].read_text()}")

def writer(task, notes):
    traxr.emit("agent_output", {"action": "synthesize"}, agent_name="writer")
    return call(f"Write the final answer from:\n{notes}")

WORKERS = {"planner": planner, "researcher": researcher, "writer": writer}

def my_mas(task: traxr.Task) -> str:            # your agent: still just (Task) -> str
    notes = ""
    while True:
        nxt = orchestrate(task, notes)          # YOUR routing logic (an LLM, rules, a graph)
        # The orchestrator picks who acts next, so it emits the routing decision:
        traxr.emit("routing_decision", {"chosen_agent": nxt}, agent_name="orchestrator")
        if nxt == "done":
            return notes
        notes = WORKERS[nxt](task, notes)
```

`agent_name` is who is acting now; `chosen_agent` is who the orchestrator routes
to next. Only the orchestrator emits `routing_decision`; the workers emit their
own `agent_output` and tool events. Custom event types work too: register them
with `register_signature()` so they count toward `d_norm`. Full guide:
[Expose your agents](docs/quickstart.md).
</details>

### 3. Run it (Python or CLI)

```python
experiment = traxr.Experiment(
    files="examples/sales.csv",
    question="Which region had the highest Q3 revenue?",
    expected_answer="EMEA",
    agent=my_mas,                 # or agent=my_agent
)
experiment.run(dry_run=True)      # the full plan: zero LLM calls, zero spend
results = experiment.run()        # baseline + perturbed runs (+ noise floor)
print(results.summary())
results.to_json("results.json")
```

Prefer the shell? The CLI takes an `import:path` to your agent:

```bash
traxr run --agent mypkg.agents:my_mas --file examples/sales.csv \
          --question "Which region won Q3?" --expected-answer EMEA \
          --out results.json
traxr run --agent mypkg.agents:my_mas --file report.pdf --question "..." --dry-run
traxr operators      # the live operator catalog
traxr selfcheck      # offline end-to-end smoke test
```

### Scoring free-text answers

A real agent answers in full sentences, so the default scorer
(`check_answer_match`, exact normalized string equality) won't match a bare
`expected_answer` against a verbose reply. Bring your own via
`ExperimentConfig(scorer=...)`, e.g. the built-in `llm_judge_match` for
semantic matching (non-deterministic, costs an extra LLM call):

```python
from traxr import ExperimentConfig
from traxr.scoring import llm_judge_match

experiment = traxr.Experiment(
    files="examples/sales.csv",
    question="Which region had the highest Q3 revenue?",
    expected_answer="EMEA",
    agent=my_agent,
    config=ExperimentConfig(scorer=llm_judge_match),  # semantic match instead of exact
)
```

See [the quickstart](docs/quickstart.md#scoring-free-text-answers) for details
and how to plug in your own deterministic scorer instead.

## Is my agent traceable?

traxr captures traces at two tiers. **Tier 0** is automatic capture at the
OpenAI-SDK boundary: wrap your client with `instrument()` and every
`chat.completions` call (sync, async, streaming, tool calls) becomes a trace
event, against any OpenAI-compatible endpoint. **Tier 1** is framework-native
capture via callbacks: the [LangGraph adapter](#langgraph) is richer, since it
also sees node transitions and tool success/failure.

Tier 0 is the default, and its honest scope is OpenAI-SDK `chat.completions`
calls. You're covered if your agent:

- uses `openai.OpenAI` / `openai.AsyncOpenAI` against any OpenAI-compatible
  endpoint (OpenAI, Azure, Ollama, vLLM, Together, Groq, OpenRouter, …) and
  you can pass the instrumented client in,
- constructs clients internally: use the `traxr.capture.patch_openai()`
  context manager instead, or
- is a LangGraph graph (`traxr.from_langgraph`).

Anything Tier 0 can't see (orchestration, memory reads, retrieval) you surface
yourself with [`traxr.emit()`](#2-your-multi-agent-system).

**Not captured (yet):** other provider SDKs, raw HTTP, the
Responses/Assistants APIs, subprocess-spawned LLM calls. Runs that capture
nothing are flagged (`EmptyTraceWarning`) rather than silently reported as
zero divergence. Full scope and caveats: [is my agent traceable?](docs/traceable.md).
A local proxy and OTel ingestion are on the [roadmap](ROADMAP.md).

### LangGraph

```python
agent = traxr.from_langgraph(compiled_graph)   # Tier 1 capture via callbacks
experiment = traxr.Experiment(files="report.pdf", question="...", agent=agent)
```

Node transitions become routing events (so reroute metrics work) and carry
agent names onto LLM events automatically, tool calls keep success/failure
fidelity, and double-counting with an instrumented client is suppressed. For
non-messages-state graphs, pass `input_builder=` / `output_extractor=`.

## How it works

1. **Perturb**: one operator is applied to a copy of your file (seeded,
   deterministic, single-variable).
2. **Paired runs**: your agent runs on the clean file, then on each
   perturbed copy, with identical seeds and fresh temp dirs.
3. **Diverging traces**: each run's LLM/tool/routing events form a trace;
   paired traces are aligned and compared structurally.

**The metrics:**

| metric | meaning |
|---|---|
| `d_norm` | normalized edit distance between paired traces: 0 = identical process, 1 = completely different |
| `t*` (+ `t*/T`) | the step where divergence first appears, and how early in the run that is |
| manifestation | how the damage showed up: silent semantic corruption, strategy reroute, early termination, catastrophic failure, recovered, … |
| `token_overhead` | perturbed-run tokens / baseline tokens |
| noise floor | baseline-vs-itself `d_norm` from clean re-runs; divergence at or below it is indistinguishable from sampling noise (**defaults to 1 re-run for external agents; don't skip it**) |

Full reference: [the metrics](docs/metrics.md).

## Perturbation operators (v1)

| input | operators | delivery |
|---|---|---|
| CSV / XLSX | `column_swap`, `label_corrupt`, `data_type_corrupt`, `row_duplicate`, `irrelevant_columns`, `unit_change`, `null_content` | file round-trip |
| TXT / MD | `ocr_noise`, `number_corruption`, `text_redaction`, `paragraph_shuffle`, `encoding_error`, `section_removal`, `null_content` | file round-trip |
| PDF (any agent) | `number_corruption`, `text_redaction`, `section_removal`, `page_removal`, `page_shuffle`, `null_content` | surgical in-place edits (extraction-fidelity preserving) |
| PDF (built-in agent only) | `ocr_noise`, `paragraph_shuffle`, `encoding_error` | extracted-content injection |

`traxr operators` prints the live catalog; full notes in
[the operator catalog](docs/operators.md).

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
(raw final answers are stored, since scoring needs them). Tier 0 never touches
HTTP headers or keys. Full notes: [SECURITY.md](SECURITY.md).

## Don't have an agentic system yet? Use ours.

Traxr ships a multi-agent reference system (planner, researcher, tools,
synthesizer). Point it at your data with `llm=` plus your API key, and you get
the full trace-level analysis without writing an agent:

```python
import traxr

experiment = traxr.Experiment(
    files="examples/sales.csv",
    question="Which region had the highest Q3 revenue?",
    expected_answer="EMEA",
    llm=traxr.OpenAICompatibleClient(model="gpt-4o-mini"),  # reads OPENAI_API_KEY
)
results = experiment.run()
```

<details>
<summary>Other providers (Azure, Ollama, vLLM, Together, …)</summary>

```python
# any OpenAI-compatible endpoint: pass base_url= and api_key=
llm = traxr.OpenAICompatibleClient(
    model="llama3.1", base_url="http://localhost:11434/v1", api_key="local")
experiment = traxr.Experiment(files="examples/sales.csv", question="...", llm=llm)
```

For providers that aren't OpenAI-compatible, implement the two-method
[`traxr.LLMClient`](src/traxr/llm/protocol.py) protocol (`generate`,
`generate_with_tools`). External agents don't need any of this: they own their
LLM and are captured at the SDK boundary.
</details>

## Notebook

[`notebooks/traxr_quickstart.ipynb`](notebooks/traxr_quickstart.ipynb) runs
top-to-bottom **without an API key** (the real-model cells are skip-safe).
Open it in Colab to try Traxr in two minutes.

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

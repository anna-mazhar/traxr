# Quickstart

## Bring your agent

Your agent is any callable `(Task) -> str`. Wrap its OpenAI client with
[`traxr.instrument()`](api.md#capture) and every
`chat.completions` call (sync, async, or streaming, including tool calls)
is captured into the trace:

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
experiment.run(dry_run=True)   # the full plan: zero LLM calls, zero spend
results = experiment.run()     # baseline + perturbed runs (+ noise floor)
print(results.summary())
results.to_json("results.json")
```

Stateful agents (memory, vector stores) should pass `agent_factory=`
(a zero-arg factory called once per run) so every run starts fresh.

## Scoring free-text answers

The default scorer, `check_answer_match`, is normalized string equality
with numeric tolerance, exact on purpose. A real agent answers in full
sentences ("The region with the highest Q3 revenue is EMEA, with a
revenue of 181,400."), which will never literally equal a bare
`expected_answer="EMEA"`. Bring your own scorer via `ExperimentConfig`:

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

`llm_judge_match` asks an LLM whether the candidate answer reaches the
same core conclusion as the expected one. Useful for verbose answers,
but **not deterministic**: it costs an extra LLM call per scored answer
and can vary between runs. By default it lazily builds an
`OpenAICompatibleClient(model="gpt-4o-mini")` from `OPENAI_API_KEY`; pass
`functools.partial(llm_judge_match, llm=my_llm)` to use another provider.
A `scorer` is just `(expected, actual) -> bool`, so a plain function works
too if you want something deterministic but less strict than the default.

## Expose your agents with `traxr.emit`

`instrument()` captures your LLM traffic automatically, but it cannot see
*which* agent is acting; those calls are tagged `agent_name="external"`. In a
multi-agent system that's the part you most want in the trace. [`traxr.emit()`](api.md#capture) is the escape
hatch: call it from inside your code, where you know who is in charge and what
they just decided.

```python
# agent_name = the agent acting now; chosen_agent = who the orchestrator picks next.
traxr.emit("routing_decision", {"chosen_agent": "researcher"}, agent_name="orchestrator")
```

Two rules carry most of the value:

- **Actor vs. destination.** `agent_name` is *who is acting now*; the payload's
  `chosen_agent` is *who runs next*. Read the `chosen_agent`s down the trace and
  you reconstruct the route, which is exactly what lets the metrics flag a
  *reroute* when a perturbation changes it.
- **Emit at decision points** the automatic capture misses: route changes,
  handoffs, memory reads, retrieval. Keep `instrument()` for the LLM/tool
  traffic; don't duplicate it with `emit`.

Outside a Traxr run `emit()` is a **no-op** (same passthrough principle as
`instrument()`), so the calls are safe to leave in production code.

### Centralized orchestrator

If one orchestrator decides who runs next, the routing event belongs to *it*,
not to the workers. Emit one `routing_decision` per hop from the orchestrator;
each worker emits its own `agent_output` / tool events:

```python
def orchestrate(task: traxr.Task) -> str:
    state = init_state(task)
    while not done(state):
        next_agent = decide_next_agent(state)        # your routing logic
        traxr.emit(
            "routing_decision",
            {"chosen_agent": next_agent},             # destination = who you picked
            agent_name="orchestrator",                # actor = the orchestrator
        )
        state = run_agent(next_agent, state)          # worker emits its own events
    return final_answer(state)
```

### Custom event types

Reusing the built-in types (`routing_decision`, `agent_output`,
`tool_invocation`, `memory_write`) makes events count toward `d_norm` out of
the box. A custom type still appears in the trace JSON, but its signature
collapses to `unknown:<type>` until you upgrade it with
[`register_signature()`](api.md#extending-the-event-vocabulary):

```python
# once, before experiment.run()
traxr.register_signature(
    "handoff",
    lambda p: f"handoff:{p.get('from_agent')}->{p.get('to_agent')}",
    classifier=lambda clean, pert: (
        "different_handoff" if clean.get("to_agent") != pert.get("to_agent") else None
    ),
    structural_types={"different_handoff"},
)
```

## The CLI

The same experiment from the shell. `--agent` takes an `import:path` to your
callable:

```bash
traxr run --agent mypkg.agents:answer --file examples/sales.csv \
          --question "Which region won Q3?" --expected-answer EMEA \
          --out results.json
traxr run --model gpt-4o-mini --file report.pdf --question "..." --dry-run
traxr operators
traxr selfcheck
```

Two spend knobs: `--max-llm-calls N` caps LLM calls per run for an external
`--agent` (the `max_llm_calls_per_run` budget, default 50; `RunBudgetExceeded`
fires before the over-budget call), and `--max-retries N` sets how many times
the built-in `--model` client retries a transient failure (default 2, the
OpenAI SDK default; `0` disables). In Python they are
`ExperimentConfig(max_llm_calls_per_run=...)` and
`OpenAICompatibleClient(max_retries=...)`.

## Use the built-in reference system

No agent of your own? Traxr ships a multi-agent reference system (planner,
researcher, tools, synthesizer). Point it at your data with `llm=` and an
API key:

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

[`OpenAICompatibleClient`](api.md#llm-clients) works against any
OpenAI-compatible endpoint (OpenAI, Azure, Ollama, vLLM, Together, Groq,
OpenRouter): pass `base_url=` and `api_key=`. For other providers, implement the
[`LLMClient`](api.md#llm-clients) protocol.

To run the reference system with no API key at all (deterministic output, used
by the test suite and `traxr selfcheck`), pass
[`DeterministicLLMStub`](api.md#llm-clients) instead of a real client.

## LangGraph

```python
agent = traxr.from_langgraph(compiled_graph)
experiment = traxr.Experiment(files="report.pdf", question="...", agent=agent)
```

Node transitions become routing events (and carry agent names onto LLM events
automatically), tool calls keep success/failure fidelity, and double-counting
with an instrumented client is suppressed. For non-messages-state graphs, pass
`input_builder=` / `output_extractor=`.

## The notebook

[`notebooks/traxr_quickstart.ipynb`](https://github.com/anna-mazhar/traxr/blob/main/notebooks/traxr_quickstart.ipynb)
runs top-to-bottom without an API key. Open it in Colab.

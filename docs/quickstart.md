# Quickstart

## Bring your own agent

Your agent is any callable `(Task) -> str`. Wrap its OpenAI client with
[`traxr.instrument()`](api.md#capture) and every
`chat.completions` call — sync, async, or streaming, including tool calls —
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
experiment.run(dry_run=True)   # the full plan — zero LLM calls, zero spend
results = experiment.run()     # baseline + perturbed runs (+ noise floor)
print(results.summary())
results.to_json("results.json")
```

Stateful agents (memory, vector stores) should pass `agent_factory=` —
a zero-arg factory called once per run — so every run starts fresh.

## Scoring free-text answers

The default scorer, `check_answer_match`, is normalized string equality
with numeric tolerance — exact on purpose. A real agent answers in full
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
same core conclusion as the expected one — useful for verbose answers,
but **not deterministic**: it costs an extra LLM call per scored answer
and can vary between runs. By default it lazily builds an
`OpenAICompatibleClient(model="gpt-4o-mini")` from `OPENAI_API_KEY`; pass
`functools.partial(llm_judge_match, llm=my_llm)` to use another provider.
A `scorer` is just `(expected, actual) -> bool`, so a plain function works
too if you want something deterministic but less strict than the default.

## No API key: the built-in reference agent

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

Swap the stub for a real endpoint with
[`OpenAICompatibleClient`](api.md#llm-clients) — OpenAI, Azure, Ollama,
vLLM, Together, Groq, OpenRouter, anything OpenAI-compatible.

## LangGraph

```python
agent = traxr.from_langgraph(compiled_graph)
experiment = traxr.Experiment(files="report.pdf", question="...", agent=agent)
```

Node transitions become routing events, tool calls keep success/failure
fidelity, and double-counting with an instrumented client is suppressed.
For non-messages-state graphs, pass `input_builder=` / `output_extractor=`.

## The CLI

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

## The notebook

[`notebooks/traxr_quickstart.ipynb`](https://github.com/anna-mazhar/traxr/blob/main/notebooks/traxr_quickstart.ipynb)
runs top-to-bottom without an API key — open it in Colab.

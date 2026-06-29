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

## Try it on a real benchmark task

[`examples/gaia_menu_sales.py`](https://github.com/anna-mazhar/traxr/blob/main/examples/gaia_menu_sales.py)
runs a real GAIA benchmark question end-to-end through Traxr's built-in
multi-agent system — no setup required (falls back to a deterministic stub
without an API key, upgrades automatically to a real `gpt-4o-mini` agent if
`OPENAI_API_KEY` is set). See
[examples/README.md](https://github.com/anna-mazhar/traxr/blob/main/examples/README.md)
for a worked-through results writeup.

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

## The notebook

[`notebooks/traxr_quickstart.ipynb`](https://github.com/anna-mazhar/traxr/blob/main/notebooks/traxr_quickstart.ipynb)
runs top-to-bottom without an API key — open it in Colab.

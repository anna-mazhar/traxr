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
    files="sales.csv",
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

## No API key: the built-in reference agent

```python
import traxr

experiment = traxr.Experiment(
    files="sales.csv",
    question="What is the total revenue?",
    expected_answer="42",
    llm=traxr.DeterministicLLMStub(scenario="identity", final_answer="42"),
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
traxr run --agent mypkg.agents:answer --file sales.csv \
          --question "Which region won Q3?" --expected-answer EMEA \
          --out results.json
traxr run --model gpt-4o-mini --file report.pdf --question "..." --dry-run
traxr operators
traxr selfcheck
```

## The notebook

[`notebooks/traxr_quickstart.ipynb`](https://github.com/anna-mazhar/traxr/blob/main/notebooks/traxr_quickstart.ipynb)
runs top-to-bottom without an API key — open it in Colab.

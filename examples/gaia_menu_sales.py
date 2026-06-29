"""A real GAIA benchmark task, run through Traxr's built-in reference agent.

GAIA (General AI Assistants) is a public benchmark of real-world questions
for AI agents. This is one of its well-known Level-1 tasks, reproduced here
with the actual GAIA spreadsheet (``gaia_menu_sales.xlsx``):

    The attached Excel file contains the sales of menu items for a local
    fast-food chain. What were the total sales that the chain made from
    food (not including drinks)? Express your answer in USD with two
    decimal places.

The sheet is a location x item-category table — five food columns
(Burgers, Hot Dogs, Salads, Fries, Ice Cream) and one drink column (Soda).
Answering correctly means filtering out the drink column *before* summing.

This uses ``llm=`` (Traxr's bundled multi-agent reference system —
``DataAnalystAgent`` reads the sheet, writes and executes Python against
it, a supervisor routes the task) rather than a single bare LLM call.
That choice matters: a one-shot "read the file, ask the model" agent has
only one possible trace shape (one LLM call, one answer) no matter what
the input looks like, so Traxr's trace-level metrics (``d_norm``, ``t*``,
manifestation) can never see anything beyond "the answer changed or it
didn't." A multi-step agent that reads data, writes code, and routes
between strategies actually has a *process* — so a corrupted column or a
swapped header can change how many tool calls happen, whether the agent
retries, or which strategy it routes to, not just what number comes out.
That's the gap between answer-level eval and trace-level eval that Traxr
measures. See examples/README.md for a worked-through results writeup.

Needs: pip install "traxr[document,openai,pandas]"

Run it as-is — no API key required (falls back to a deterministic,
zero-cost stub, though the stub's scripted replies don't vary with the
file's content, so it can't show content-driven divergence — only the
real-model path below can). Set OPENAI_API_KEY to run the real agent.
"""

import os
from pathlib import Path

import traxr
from traxr import ExperimentConfig
from traxr.scoring import llm_judge_match

XLSX_PATH = Path(__file__).parent / "gaia_menu_sales.xlsx"
QUESTION = (
    "The attached Excel file contains the sales of menu items for a local "
    "fast-food chain. What were the total sales that the chain made from "
    "food (not including drinks)? Express your answer in USD with two "
    "decimal places."
)
EXPECTED_ANSWER = "89706"

if os.environ.get("OPENAI_API_KEY"):
    llm = traxr.OpenAICompatibleClient(model="gpt-4o-mini")
    experiment = traxr.Experiment(
        files=str(XLSX_PATH),
        question=QUESTION,
        expected_answer=EXPECTED_ANSWER,
        llm=llm,
        # enable_python_tool defaults to True: the agent writes and runs
        # real Python against the sheet instead of eyeballing a number.
        # A real agent answers in a sentence, not a bare number —
        # llm_judge_match handles that.
        config=ExperimentConfig(scorer=llm_judge_match),
    )
else:
    print("No OPENAI_API_KEY found — running offline with a deterministic stub.")
    print("Set OPENAI_API_KEY to try this with the real multi-agent system instead.\n")
    experiment = traxr.Experiment(
        files=str(XLSX_PATH),
        question=QUESTION,
        expected_answer=EXPECTED_ANSWER,
        llm=traxr.DeterministicLLMStub(scenario="identity", final_answer=EXPECTED_ANSWER),
    )


experiment.run(dry_run=True)  # the full plan, zero LLM calls, zero spend
results = experiment.run()  # baseline + perturbed runs (+ noise floor)
print(results.summary())
results.to_json(str(Path(__file__).parent / "gaia_menu_sales_results.json"))

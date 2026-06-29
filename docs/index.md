# traxr

**Evaluate multi-agent systems beyond final-answer accuracy.** A multi-agent
system can land the right answer through the wrong process, and answer-level
metrics never see it. traxr evaluates the **execution trace itself**: point your
agent at your data, run paired experiments, and measure how its behavior
diverged. How much (`d_norm`), where it started (`t*`), how it manifested, and
what it cost in tokens. Controlled input perturbation is the instrument; the
trace is the measurement.

traxr operationalizes the paper *“Trace-Level Analysis of Information
Contamination in Multi-Agent Systems”*
([CAIS 2026](https://dl.acm.org/doi/10.1145/3786335.3813147);
Mazhar, Suri, Galhotra) as an SDK, for any Python agent that talks to an
OpenAI-compatible endpoint through the OpenAI SDK.

## How an experiment works

1. **Perturb**: one operator corrupts a copy of one file, seeded,
   deterministic, single-variable.
2. **Run paired**: your agent runs on the clean file, then on each
   corrupted copy, with identical seeds in fresh temp dirs under original
   basenames. The agent cannot tell which condition it is in.
3. **Compare traces**: every LLM call, tool call, and routing decision is
   an event; paired traces are aligned and compared structurally.

Start with the [quickstart](quickstart.md), including how to
[score free-text answers](quickstart.md#scoring-free-text-answers) and how to
[expose your agents with `traxr.emit`](quickstart.md#expose-your-agents-with-traxremit).
Check [which agents are traceable](traceable.md), and read the
[security notes](security.md) before running an agent with real tools.

## Install

```bash
pip install "traxr[document,openai,pandas] @ git+https://github.com/anna-mazhar/traxr.git@main"
```

| extra | provides |
|---|---|
| `document` | PDF + XLSX support (PyMuPDF, pdfplumber, openpyxl) |
| `openai` | the built-in reference agent's LLM client |
| `pandas` | DataFrame export; required by the built-in reference agent |
| `langgraph` | the LangGraph adapter |
| `viz` | matplotlib plots over results |

External agents that bring their own OpenAI client need **no extras at all**.

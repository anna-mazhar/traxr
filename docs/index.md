# traxr

Point **your own agent** at **your own data**, run controlled-perturbation
experiments, and get back contamination/divergence metrics: how much the
execution trace diverged (`d_norm`), where it started (`t*`), how the damage
manifested, and what it cost in tokens.

traxr operationalizes the paper *“Trace-Level Analysis of Information
Contamination in Multi-Agent Systems”* (Mazhar, Suri, Galhotra) as an SDK:
the same paired clean-vs-perturbed methodology, for any Python agent that
talks to an OpenAI-compatible endpoint through the OpenAI SDK.

## How an experiment works

1. **Perturb** — one operator corrupts a copy of one file: seeded,
   deterministic, single-variable.
2. **Run paired** — your agent runs on the clean file, then on each
   corrupted copy, with identical seeds in fresh temp dirs under original
   basenames. The agent cannot tell which condition it is in.
3. **Compare traces** — every LLM call, tool call, and routing decision is
   an event; paired traces are aligned and compared structurally.

Start with the [quickstart](quickstart.md), check
[which agents are traceable](traceable.md), and read the
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

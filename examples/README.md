# Examples

## `gaia_menu_sales.py`

**The task** (a real, public [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) Level-1 question):

> The attached Excel file contains the sales of menu items for a local
> fast-food chain. What were the total sales that the chain made from food
> (not including drinks)? Express your answer in USD with two decimal
> places.

`gaia_menu_sales.xlsx` is a location × item-category table — five food
columns (Burgers, Hot Dogs, Salads, Fries, Ice Cream) and one drink column
(Soda). The verified ground truth is **89706** (sum of the food columns,
Soda excluded).

**Run it:**

```bash
python examples/gaia_menu_sales.py            # offline, zero-cost stub
OPENAI_API_KEY=sk-... python examples/gaia_menu_sales.py   # the real demo
```

### When Traxr earns its keep in an agentic setting

Traxr's metrics (`d_norm`, `t*`, manifestation) describe *how* an agent
got to its answer, not just whether the answer matches — so they're most
informative when the agent's process has something to show: tool calls,
retries, retrieval, routing between strategies. This example uses `llm=`
to route the task through Traxr's bundled reference multi-agent system: a
`DataAnalystAgent` reads the sheet, writes and executes real Python
against it, and a supervisor can reroute or retry. A corrupted input can
then change the path taken — an extra tool call, a retry, a different
strategy — independently of whether the final answer survives, which is
exactly the kind of divergence answer-only evaluation can't see.

### A real run's results

One real run (`OPENAI_API_KEY` set, `gpt-4o-mini`, `scorer=llm_judge_match`)
produced:

| perturbation | manifestation | answer changed | task succeeded | `d_norm` | `t*` | token overhead |
|---|---|---|---|---|---|---|
| `column_swap` | no_observable_effect | no | yes | 0.00 | — | 0.99x |
| `label_corrupt` | loop_or_extended_execution | yes | no | 0.69 | 0 | 3.02x |
| `data_type_corrupt` | loop_or_extended_execution | no | yes | 0.56 | 0 | 4.46x |
| `row_duplicate` | loop_or_extended_execution | yes | no | 0.25 | 2 | 1.56x |
| `irrelevant_columns` | loop_or_extended_execution | no | yes | 0.25 | 2 | 1.45x |
| `unit_change` | silent_semantic_corruption | yes | no | 0.00 | — | 1.00x |
| `null_content` | catastrophic_failure | yes | — | 0.65 | 2 | — |

Aggregates: mean `d_norm` 0.34 (max 0.69), recovery rate 40%, mean token
overhead 2.08x (max 4.46x).

**What this says that an answer-only eval can't:**

- **`data_type_corrupt` and `irrelevant_columns` still got the right
  answer** — but only after the trace visibly lengthened (`d_norm` 0.56
  and 0.25, token overhead up to 4.46x). An answer-only harness would
  score these as two clean passes; Traxr shows the agent had to fight for
  it both times, via `t*` (it started diverging at step 0 and step 2
  respectively) and the extra tool calls behind the token overhead. That's
  exactly the "right answer, wrong/costlier process" case the trace-level
  framing exists to catch.
- **`unit_change` is the inverse case**: the answer flipped
  (`task_success=False`) while the trace looked structurally identical
  (`d_norm=0.0`) — `silent_semantic_corruption`. Nothing about the
  process looked different; only the number at the end was wrong. This is
  the case answer-only eval *does* catch (wrong answer), but trace-level
  eval explains *why* it's dangerous: there was no process-level signal
  that anything had gone wrong.
- **`null_content` triggered `catastrophic_failure`** (`d_norm=0.65`,
  blank/failed run) — the harshest corruption produced the harshest
  manifestation, which is the sanity check you'd want from any of these
  metrics.
- **Recovery rate of 40%** means that across the pairs where something
  measurably changed, the agent still landed on the correct answer less
  than half the time — a concrete, single number for "how brittle is this
  agent to this class of corruption," which is the thing this whole tool
  is for.

Your own run will differ (LLM nondeterminism, model version, GAIA file
specifics) — rerun it and compare against this table rather than treating
these numbers as fixed.

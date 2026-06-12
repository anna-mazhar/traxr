# traxr v1.0.0

Point **your own agent** at **your own data**, run controlled-perturbation
experiments, and get back contamination/divergence metrics: how much the
execution trace diverged (`d_norm`), where it started (`t*`), how the
damage manifested, and what it cost in tokens.

```bash
pip install "traxr[document,openai,pandas] @ git+https://github.com/anna-mazhar/traxr.git@v1.0.0"
```

## Highlights

- **Bring your own agent** — any `(Task) -> str` callable. Wrap its OpenAI
  client with `traxr.instrument()` (sync/async/streaming/tool calls) or
  bring a LangGraph graph via `traxr.from_langgraph()`. No key? The
  bundled reference agent runs fully offline under a deterministic stub.
- **Bring your own data** — CSV/XLSX/TXT/MD/PDF, with 20+ seeded,
  deterministic perturbation operators, including surgical in-place PDF
  edits that preserve extraction fidelity.
- **Honest numbers** — a measured noise floor (on by default for external
  agents), concurrency detection, structural (not lexical) trace
  comparison, byte-stable JSON exports, and runs that capture nothing get
  flagged instead of reported as zero divergence.
- **Honest costs** — `run(dry_run=True)` plans everything with zero LLM
  calls; `max_llm_calls_per_run` is enforced inside the capture wrapper;
  live token totals per run.

Read [SECURITY.md](https://github.com/anna-mazhar/traxr/blob/main/SECURITY.md)
before running an agent with side-effectful tools: perturbed data is an
injection-adjacent vector, and traxr cannot sandbox your agent.

Full details in the
[CHANGELOG](https://github.com/anna-mazhar/traxr/blob/main/CHANGELOG.md).
Operationalizes *“Trace-Level Analysis of Information Contamination in
Multi-Agent Systems”* (Mazhar, Suri, Galhotra).

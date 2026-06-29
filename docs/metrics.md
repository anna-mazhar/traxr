# The metrics

Every perturbation produces one **pair**: the clean baseline run vs. the
perturbed run. All metrics are per-pair, aggregated in
`ExperimentResults`.

## d_norm: how much the process changed

Normalized edit distance between the paired traces' structural signatures:
the minimum number of insertions, deletions, and substitutions to turn one
event sequence into the other, normalized to `[0, 1]`. Signatures are
*structural*: they never include argument values or content, so lexical
noise does not saturate the metric.

- `0.0`: the agent did exactly the same thing.
- `1.0`: a completely different execution.

## t*: where divergence began

The step at which the traces first diverge structurally, plus the
normalized position `t*/T` (0 = diverged immediately, 1 = at the very end).
Early divergence means the corruption changed the agent's plan, not just
its final wording.

## Manifestation: how the damage showed up

A rule-based taxonomy over the pair:

| fine category | meaning |
|---|---|
| `silent_semantic_corruption` | answer changed, process identical (the scariest row) |
| `strategy_reroute` | a different agent/node was chosen |
| `early_termination` | the perturbed run gave up sooner |
| `loop_or_extended_execution` | the run thrashed or ran long |
| `catastrophic_failure` | null answer plus major disruption |
| `structural_divergence_with_outcome_change` | different path, different answer |
| `structural_divergence_recovered` | different path, same answer |
| `no_observable_effect` | nothing changed |

Each fine category rolls up to one of the paper's four groups: silent
corruption, behavioral detours, combined disruption, no observable effect.

## Token overhead

Perturbed-run tokens / baseline tokens, from captured usage. Corruption you
pay for. External agents get this from the capture wrapper's captured
`usage`; runs without usage data are reported as unavailable, never as 1.0.

## The noise floor

LLM agents are not deterministic: sampling temperature, concurrency
scheduling, and retrieval nondeterminism all produce divergence with **no
perturbation at all**. `noise_floor_runs` re-runs the clean baseline; the
baseline-vs-itself `d_norm` *is* the floor. Pairs at or below it are
flagged `within_noise_floor`; report them as contamination at your peril.

Defaults: **1 re-run for external agents** (do not turn it off), 0 for the
deterministic built-in-agent-plus-stub path. `summary()` warns prominently
whenever the floor is unmeasured for an external agent.

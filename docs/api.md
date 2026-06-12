# API reference

The curated public surface. Everything here is importable from `traxr`
unless noted.

## Experiments

::: traxr.experiment.Experiment

::: traxr.experiment.ExperimentConfig

::: traxr.experiment.ExperimentPlan

## Results

::: traxr.results.ExperimentResults

::: traxr.results.PairResult

## Capture

::: traxr.capture.openai_wrap.instrument

::: traxr.capture.patch.patch_openai

::: traxr.capture.context.emit

## The agent contract

::: traxr.agents.task.Task

::: traxr.agents.task.invoke_agent

::: traxr.agents.langgraph.from_langgraph

::: traxr.agents.builtin.builtin_agent

## LLM clients

::: traxr.llm.protocol.LLMClient

::: traxr.llm.openai_compat.OpenAICompatibleClient

::: traxr.llm.stub.DeterministicLLMStub

## Scoring and plots

::: traxr.scoring.check_answer_match

::: traxr.viz.plot_d_norm

::: traxr.viz.plot_t_star

::: traxr.viz.plot_manifestations

## Extending the event vocabulary

::: traxr.trace.registry.register_signature

## Errors and warnings

::: traxr.errors
    options:
      members_order: source
      show_if_no_docstring: false

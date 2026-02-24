# Product Research: Local Multi-Agent Swarm Platform

## Scope

Research target: products similar to a local-first, multi-model, voice-capable orchestration system that supports swarms/hiveminds/graphs and reliability methods similar to MAKER/MDAP.

## Reviewed products

- LM Studio (`https://lmstudio.ai/`)
  - Strengths: local model UX, OpenAI-compatible API, model catalog/discovery, headless mode.
- Open WebUI (`https://openwebui.com/`)
  - Strengths: all-in-one chat UI, local + cloud connectors, community assets, multimodal support.
- LangGraph (`https://www.langchain.com/langgraph`)
  - Strengths: controllable orchestration patterns, persistence, streaming, human-in-the-loop.
- CrewAI (`https://www.crewai.com/`)
  - Strengths: multi-agent orchestration, tracing/observability, enterprise controls, visual builder.
- AutoGen (`https://microsoft.github.io/autogen/stable/`)
  - Strengths: agent framework layers (studio/core/extensions), event-driven multi-agent runtimes.

## Common capabilities in successful products

- Fast local setup and model management.
- OpenAI-compatible endpoint support to swap runtimes without code rewrite.
- Clear orchestration modes (single, sequential, multi-agent, hierarchical/graph).
- Strong observability: traces, health checks, run history, retries, errors.
- Human-friendly UI for operators and prompt engineers.
- Voice and multimodal options with graceful fallback.

## Gaps we should close in this repo

Top priorities for this project direction:

1. Reliability tooling for model fleet status.
2. Run persistence and reproducibility records.
3. Runtime adaptability by hardware profile.
4. Better operator UX around health and run history.
5. Pluggable connectors to existing local stacks.

## Implemented from this research

- Added model health API: `GET /api/health/models`
- Added run history API: `GET /api/runs`
- Persisted run results to `data/runs/history.jsonl`
- Added UI controls for health checks and recent runs
- Added CLI diagnostics command: `mdap-small doctor`

## Next best additions (recommended backlog)

- Per-model latency and token accounting in every vote round.
- Structured trace viewer (step -> votes -> winner -> red flags).
- Auto-calibration pipeline: estimate `p_step`, compute `k_min`, test decorrelation.
- Guardrail policies: parser strictness levels and stop conditions.
- Live graph execution view for `graph`/`hivemind` modes.

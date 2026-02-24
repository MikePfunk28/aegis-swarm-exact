# MDAP Small Swarm (<=3B)

This project implements the core MAKER method from arXiv:2511.09030 for small open models:

- Maximal Agentic Decomposition (one micro-step per agent call)
- First-to-ahead-by-k voting
- Red-flagging unreliable samples
- MoE-style expert routing across local model pools
- Optional local tool-use via allowlisted bash/powershell commands

It is adapted to local/open models (targeting <=3B), including:

- `LiquidAI/LFM2.5-1.2B`
- `Qwen/Qwen3-1.7B`
- `google/gemma-3-1b-it`
- `google/gemma-3-270m-it`
- `Nanbeige/Nanbeige4.1-3B`

## What is included

- Exact algorithmic structure of MAKER for step-level voting
- Sequential and parallel vote sampling
- Interleaved reasoning through alternating model pools
- Mixture-of-experts swarm routing (planner/executor/verifier/general)
- Towers of Hanoi benchmark harness (streaming steps)
- Browser voice control UI (Web Speech API, free in Chrome/Edge)
- Optional hook to existing `M:\voiceai` Whisper pipeline
- Workflow UI for `swarm`, `hivemind`, `graph`, and `sequential` modes
- 1.2B leader + many 270M micro-agent worker pattern (10-15 by default)
- Tool-assisted micro-agent turns (optional, disabled by default)

## Quick start

```powershell
cd M:\swarmAI
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

Run a calibration benchmark on 10-disk random steps:

```powershell
mdap-small benchmark --disks 10 --samples 500 --ahead-k 3 --parallel-votes 3
```

Paper-style calibration (repairing parser, 2048 tokens, random step sampling):

```powershell
mdap-small paper-calibrate --disks 10 --samples 1000 --target-success 0.95
```

Write a strict MAKER profile (red-flagging parser, 750 tokens, temp 0 then 0.1):

```powershell
mdap-small paper-profile --ahead-k 3 --output configs/paper_strict.yaml
```

Auto-calibrate across eligible local <=3B models and write tuned strict profile:

```powershell
mdap-small paper-autotune --disks 10 --samples 120 --collision-samples 30 --output configs/paper_autotuned.yaml
```

Run paper-style exact validation protocol (calibration + 10K decorrelation + parser collision comparison + strict profile emit):

```powershell
mdap-small paper-validate-exact --disks 20 --calibration-samples 1000 --decorrelation-samples 10000 --output-profile configs/paper_validated_strict.yaml --output-report runs/paper_validation_report.json
```

Run full solve loop (streaming, no full-sequence preload):

```powershell
mdap-small solve --disks 12 --ahead-k 3 --parallel-votes 3
```

Sync local <=3B candidates from Ollama (strands-inspired selector):

```powershell
mdap-small strands-sync
```

Start browser voice UI:

```powershell
mdap-small-server
```

Open `http://127.0.0.1:8088`.

The UI includes:

- Voice panel
- Machine profile loader (for your 13700K / 32GB / RX6600)
- Run panel for benchmark/solve with workflow mode and agent count controls
- Model health checks
- Recent run history viewer

## Model serving

Default adapter is Ollama-compatible HTTP. You can point to any OpenAI-compatible local gateway later.

- Configure in `configs/models.yaml`
- Pull models in your chosen runtime (Ollama/vLLM/TGI)

## Diagnostics and observability

- `GET /api/health/models` returns model endpoint status + presence checks.
- `GET /api/runs` returns recent runs.
- Run records are stored in `data/runs/history.jsonl`.
- CLI doctor:

```powershell
mdap-small doctor
```

## Product research

- Research findings and prioritized roadmap are documented in `docs/PRODUCT_RESEARCH.md`.

## MAKER settings mapped from paper

- `m=1` (one action per agent call)
- decision rule: first candidate ahead by `k`
- red flags:
  - malformed structured output
  - overlong response (default 750 tokens approx)
- structured output format:
  - `move = [disk, source, target]`
  - `next_state = [[peg0...],[peg1...],[peg2...]]`
- voting strategy:
  - include one deterministic vote (`temperature=0`)
  - then stochastic votes (`temperature=0.1`) until ahead-by-k

## Validation process (paper priority)

- calibration phase: `paper-calibrate` uses random steps, repairing parser, and 2048 token cap to estimate single-step success `p`
- k estimation: computes `k_min` from Eq. 14 using estimated `p`, target success, and full-step horizon
- theorem guard: if measured `p_step <= 0.5`, CLI reports theorem-not-applicable and falls back to `k=3` in autotune profile generation
- strict run phase: `paper-profile` enforces red-flagging parser, 750 token cap, and first deterministic then stochastic voting
- collision check: calibration reports two-sample wrong/wrong collisions to expose correlated errors before scale-up

## Notes

- You can swap in your existing graph/swarm memory systems from `M:\strandsagents`.
- You can plug your existing voice stack from `M:\voiceai` for open-source Whisper STT.
- This repo does not change system or user PATH variables.

## Voice priority behavior

Voice is prioritized to always work:

- First choice: browser Web Speech API + browser speech synthesis (free)
- Optional: set `VOICEAI_GATEWAY_URL` to route through `M:\voiceai` (`/voice/reply`)
- If VoiceAI proxy fails, server falls back automatically to local responses

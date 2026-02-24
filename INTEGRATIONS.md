# Existing Project Integration Map

## `M:\strandsagents`

- Reuse graph memory and task analytics modules for step-level traces.
- Recommended bridge point: emit every vote event to your graph store.
- Minimal integration hook:
  - call graph writer after each step in `src/mdap_small/orchestrator.py`
- Detection endpoint is exposed in UI config at `/api/ui-config`.

## `M:\voiceai`

- This repo provides a browser-first free voice path through Web Speech.
- For open-source STT/TTS, connect to your existing VoiceAI stack:
  - STT: Whisper.cpp in browser worker
  - TTS: Piper routing
- Recommended bridge:
  - replace `/api/respond` in `src/mdap_small/server.py` with a proxy call to your voice gateway.
- Implemented path today:
  - set `VOICEAI_GATEWAY_URL` and server will attempt `/voice/reply` proxy first.
  - automatic fallback to browser voice response path if gateway is unavailable.

## `M:\betterAI`, `M:\HRM`, `M:\SmartBot`

- Good candidates for model-eval telemetry and experiment tracking.
- Keep this repo focused on method correctness and benchmark reproducibility.

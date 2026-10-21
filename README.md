# A decision model plays Super Mario (NES)

A real open-weight **decision model** (`vllm-sr/d3-edge`, 0.59B, Apache-2.0) plays the **real NES** Super Mario Bros, live and smooth. Each tick: decode the game's RAM into a compact text `situation`, ask the model for one typed controller `choice`, press the button. The model returns a decision with a probability per option and **0 output tokens** — no free text, no decoder. One typed decision per tick, driving a real game.

## Run

```bash
uv sync --extra nes --python 3.13      # native emulator (finicky on Apple Silicon); the SMB ROM ships inside gym-super-mario-bros
uv run python mario_nes.py             # d3-edge plays 1-1, smooth real-time (async)
uv run python mario_nes.py --debug     # also print the decoded state each tick
uv run python mario_nes.py --level 1-2 # any stage 1-1..8-4 (only 1-1 is tuned/beatable)
```

First run downloads d3-edge (~1.2 GB) from Hugging Face; it runs on the GPU (MPS) at ~110 ms/decision. See [`MARIO.md`](./MARIO.md) for how it works (perception, jump sizing, latency compensation).

![a decision model clearing SMB 1-1](./assets/mario.gif)

## Files

- **`mario_nes.py`** — the loop: NES RAM → `situation` → one `choice` → button, at 60 fps (game on the main thread, model on a background thread, latency-compensated).
- **`decision.py`** — the three interchangeable backends behind one `.choice()` interface:
  - `Decision3` — the brain: `vllm-sr/d3-edge`, a purpose-built open-weight decision model (`model.system_one(...)` answers choice/yes-no/score in one pass).
  - `SystemOne` — a ~60-line "decision model" faked from a vanilla small LLM (restricted softmax over the option tokens); the GIF recorder uses it (Qwen-1.5B is faster than d3 on some Macs). The honest contrast: this coaxes typed decisions out of a chat model; d3 *is* one.
  - `DecisionsAPI` — the same interface over a **hosted** decision model (`POST /v1/decisions` on OpenAI Decisions; a Jev / Microsoft-Decision-1 variant is a few lines different).
- **`make_mario_nes_gif.py`** — record a GIF of an async run (the one shown above).

It mirrors the real decision-model mechanism closely enough to make the point: structured state in, one typed decision out.

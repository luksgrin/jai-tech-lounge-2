# How d3-edge plays Super Mario (NES)

A real open-weight decision model drives the real NES. The loop, each tick:

```
state  = decode NES RAM into a one-line `situation`   (not pixels)
button = d3_edge.choice(instruction, state, BUTTONS)  # one pass, a typed choice
game.step(button)
```

Run: `uv run python mario_nes.py` (add `--debug` to print the state; `--level 1-2` for
another stage). That's the whole CLI — everything else is fixed: d3-edge brain, smooth
real-time async, walking pace, 60 fps.

## What the model sees

`SMBRam.observe()` reads a handful of RAM addresses and emits a compact state; the model
only reads the one-line `situation` string:

```json
{ "mario_tile": 8, "on_ground": true, "gap_ahead_in": 2, "gap_width": 2,
  "obstacle_ahead_in": null, "enemy_ahead_in": null,
  "situation": "A deadly gap 2 tile(s) wide is 2 tile(s) ahead. Choose jump_right NOW or fall in." }
```

The question is one `choice` over `["run_right","walk_right","jump_right","hop","wait"]`.

## The three things that make it work

**1. Perception is the hard part, not the model.** The RAM addresses in `SMBRam` come
from the widely-cited SMB disassembly / "SMB AI" maps; they're game- and ROM-specific.
The decoder finds pits, pipes/walls, and enemies. Two constants most likely to need a
tweak for your ROM are the playfield-top offset and the on-ground flag — run `--debug`
and watch the decoded state. **The model is only ever as good as the state you hand it.**

**2. Jump sizing.** A single max-height jump everywhere makes Mario sail over enemies and
into the next pit. So the action set has a small `hop` (stomp an enemy / clear a 1-tile
gap) and a big `jump_right` (pipe / wide gap). `ButtonDriver` turns each choice into the
right NES presses — SMB jumps are edge-triggered and variable-height, so you can't just
hold the button. `jump_right` holds A for 24 frames: a few-Hz decider can't hit a wide gap
frame-perfectly, so the big jump carries extra airtime to clear it even when mistimed
(tuned — 18 undershot 1-1's famous mid gap, 28 overshoots). Jump triggers: pipe 2 tiles,
gap 3, enemy 6 (enemies come in pairs and walk toward Mario, so they need the earliest
trigger to clear both in one arc — and 6 is load-bearing for a Goomba right before the mid gap).

**3. Latency compensation = smooth real-time.** The game runs at 60 fps on the main
thread; the model decides on a background thread and its latest choice is applied as it
arrives. While the model thinks (~110 ms) Mario keeps moving, so the decoder looks a tile
ahead — `lead = walk-speed × measured decision time` — and decides for where he *will be*
when the choice lands (enemies get ~double). That's why it stays smooth and still lands
jumps on time, instead of walking off ledges.

## d3-edge must run on the GPU

d3-edge is a Mamba/linear-attention hybrid whose fast kernels are CUDA-only, so on **CPU**
it's ~2.7 s/decision (unusable); on **MPS** it's ~110 ms. The `Decision3` backend in `decision.py` picks MPS automatically
and prints `[d3-edge on mps]` at load — if it says `cpu`, that's your slowness.

## Only 1-1 is beatable

Every stage loads and runs (the RAM decode is game-wide), but the perception + tuning are
built for 1-1's flat-ground hazards. Castles (lava/firebars), underground, water, moving
platforms, and piranha plants need mechanics this decoder doesn't model, so Mario dies
early there. Same model everywhere — **solvability is about the perception you hand it.**

And even 1-1 is only ~1-in-3: with d3-edge deciding a few times a second, it clears the
level about **32%** of runs (measured over 28 headless playthroughs; ~20% before the jump
tuning). The leftover deaths are Goomba-timing collisions — the decider is too slow to
frame-perfectly stomp a walking enemy. That ceiling is the honest takeaway: a thing fast
enough to call many times a second still isn't fast enough to play a platformer perfectly.
The GIF in the README is a winning run (the recorder retries until it clears).

## No sound

`nes-py` has no audio at all (no APU; video-only viewer), so the window is silent by
design. For a demo you'd loop a music track alongside it, or run the ROM in a full
emulator (Mesen/FCEUX) just for the sound.

## Why it's a useful demo

It shows "decision model = typed `if`" on a real game: the same primitive that routes a
support ticket can drive Mario in real time — structured state in, one typed choice out,
0 output tokens. And it makes the limit concrete: a thing fast and cheap enough to call
many times a second is fast and cheap enough to screen every untrusted input — but it is
still a local judge, not a planner.

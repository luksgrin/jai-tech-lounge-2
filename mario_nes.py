"""
Play the REAL Super Mario Bros. (NES) with a decision model.

A real open-weight decision model (d3-edge) plays the real NES, smoothly in real
time. Each tick: decode NES RAM into a compact text `situation`, ask the model for
one controller `choice`, press the button. Perception (RAM -> pits/pipes/enemies)
and a right-sized action set (small `hop` vs big `jump_right`) are the only things
the real game needs beyond the toy sim. The game runs at 60fps on the main thread
while the model decides on a background thread (latency-compensated), so it stays
smooth even though a decision takes ~100ms.

Install (the native build is finicky on Apple Silicon; needs Python 3.13+; the SMB
ROM ships inside gym-super-mario-bros, no separate ROM file):
    uv sync --extra nes --python 3.13

Run:
    uv run python mario_nes.py                 # d3-edge plays 1-1, async, smooth
    uv run python mario_nes.py --debug         # also print the decoded state each tick
    uv run python mario_nes.py --level 1-2     # any stage (only 1-1 is tuned/beatable)
"""
from __future__ import annotations

import argparse
import sys
import time

# The model picks one of these each tick. Jumps come in two SIZES: one max-height
# jump everywhere makes Mario sail over enemies and into the next pit, so a big
# jump_right (pipes, wide gaps) and a small hop (stomp an enemy, clear a 1-tile gap).
NES_ACTIONS = ["run_right", "walk_right", "jump_right", "hop", "wait"]

INSTRUCTION = (
    "You are playing Super Mario Bros. Read `situation` and pick ONE controller action:\n"
    "- jump_right: a big running jump. Use it to clear a PIPE/WALL or a WIDE gap (2+ tiles).\n"
    "- hop: a small jump. Use it to stomp an ENEMY or clear a 1-tile gap.\n"
    "- run_right: move right when the path ahead is clear.\n"
    "ONLY jump when `situation` says NOW; otherwise choose run_right. Jumping early "
    "overshoots and kills Mario. Choose exactly one action."
)

# How many tiles ahead to start a jump. Pipes and gaps are static, so jump close
# (2-3 tiles) -- jumping earlier overshoots into the next pit. A MOVING enemy walks
# toward Mario and usually comes in a pair, so it needs a big jump started EARLIER
# (6 tiles) to be airborne over both in one arc.
_TRIGGER_PIPE = 2
_TRIGGER_GAP = 3
_TRIGGER_ENEMY = 6


def _nes_situation(gap_in, gap_w, enemy_in, obstacle_in) -> str:
    """Name the nearest hazard and the right-sized action -- but only once it's close
    enough to act on. Farther out: just run (naming a jump early makes Mario jump too
    soon and die)."""
    if obstacle_in is not None and obstacle_in <= _TRIGGER_PIPE and (gap_in is None or obstacle_in <= gap_in):
        return f"A pipe/wall is {obstacle_in} tile(s) ahead. Choose jump_right NOW to clear it."
    if gap_in is not None and gap_in <= _TRIGGER_GAP and gap_in <= (enemy_in if enemy_in is not None else 99):
        act = "jump_right" if (gap_w or 1) >= 2 else "hop"     # wide gap needs distance
        return f"A deadly gap {gap_w} tile(s) wide is {gap_in} tile(s) ahead. Choose {act} NOW or fall in."
    if enemy_in is not None and enemy_in <= _TRIGGER_ENEMY:
        return f"An enemy is {enemy_in} tile(s) ahead. Choose jump_right NOW to jump over it."
    return "Clear path ahead. Run right."


# --- text action -> NES buttons ---------------------------------------------
# SIMPLE_MOVEMENT = [[NOOP],[right],[right,A],[right,B],[right,A,B],[A],[left]]
# Only GROUND moves live here; jumps are driven by ButtonDriver (they need edge-
# triggering and a per-jump hold length). Walking (default) moves ~half as fast as
# running, so Mario covers ~1.6 tiles between decisions instead of ~3.
_RUN_MAP = {"run_right": 3, "walk_right": 1, "wait": 0}   # run_right = right+B (no left: 1-1 is forward-only)
_WALK_MAP = {"run_right": 1, "walk_right": 1, "wait": 0}  # run_right = right (walk)


def action_map(walk: bool) -> dict:
    return _WALK_MAP if walk else _RUN_MAP


# jump action -> (SIMPLE_MOVEMENT index pressed, frames to hold A). Bigger hold =
# higher/longer jump. jump_right keeps B for air-speed/distance; hop drops B and
# holds A briefly so it clears an enemy/1-tile gap without flying across the screen.
JUMP_SPECS = {
    "jump_right": (4, 24),   # right+A+B, full jump  (tall pipes, wide gaps)
    "hop":        (2, 9),    # right+A,   short jump (stomp an enemy, 1-tile gap)
}
# jump_right's A-hold (24) is tuned empirically: a few-Hz async decider can't hit a
# wide gap frame-perfectly, so the jump carries extra airtime to clear it even when
# mistimed. 18 undershot 1-1's famous mid-level gap (the #1 death); 24 ~doubled the
# 1-1 win rate (20% -> ~32% over 28 runs); 28 overshoots. The remaining deaths are
# Goomba-timing collisions -- the inherent ceiling of deciding a few times a second.
_NO_A = {2: 1, 4: 3}         # same move with A released (to re-arm / fall)


class ButtonDriver:
    """Drive jumps correctly on the NES. Two facts:
      1. A jump fires only on A's RISING edge -- holding A down makes Mario jump
         once and then stand there (the "stuck, on_ground=True" bug).
      2. Jump height is VARIABLE with how long A is held; a 1-frame tap is a tiny
         hop. So for each jump we press A and hold it for the action's frame count
         (through the ~1-frame launch lag), then release A and wait to land before
         the next jump. Different actions hold for different lengths -> jump SIZE."""

    def __init__(self, amap: dict):
        self.amap = amap
        self.held = 0            # frames A has been held this jump (0 = not jumping)

    def button(self, action: str, on_ground: bool) -> int:
        if action not in JUMP_SPECS:              # a ground move
            self.held = 0
            return self.amap.get(action, 0)
        press, hold_n = JUMP_SPECS[action]
        if self.held == 0:                        # not currently jumping
            if on_ground:
                self.held = 1                     # press A now -> jump starts
                return press
            return _NO_A[press]                   # airborne but not our jump
        if self.held < hold_n:                    # hold A for height (through lag)
            self.held += 1
            return press
        if on_ground:                             # held enough and landed -> re-arm
            self.held = 0
        return _NO_A[press]                       # release A (arc/fall)

# Tiles are 16 px. SMB 1-1 flagpole is at level-x ~= 3161 px.
TILE = 16
FLAG_X = 3161
FLAG_TILE = FLAG_X // TILE


# ------------------------------------------------------------------------- #
#  PERCEPTION: decode NES RAM into a compact state dict. Addresses are the
#  widely-cited SMB disassembly / "SMB AI" map (Chrispresso, FCEUX cheat DBs).
#  They are game- and ROM-specific: if the decode looks wrong, run with --debug
#  and tweak the few constants flagged below. This is the hard part -- the model
#  is only ever as good as the state this hands it.
# ------------------------------------------------------------------------- #
class SMBRam:
    # player
    A_PLAYER_PAGE = 0x6D       # high byte of player level-x (which 256-px page)
    A_PLAYER_X    = 0x86       # player x within the page
    A_PLAYER_Y    = 0xCE       # player y on screen (pixels, smaller = higher)
    A_PLAYER_STAT = 0x1D       # player float state: 0x00 = on ground, else airborne
    # enemies (5 slots)
    A_ENEMY_DRAWN = 0x0F       # 0x0F..0x13 : nonzero => slot active
    A_ENEMY_PAGE  = 0x6E       # 0x6E..0x72 : enemy level-x page
    A_ENEMY_X     = 0x87       # 0x87..0x8B : enemy x within page
    A_ENEMY_Y     = 0xCF       # 0xCF..0xD3 : enemy y on screen
    # level tile buffer: 0x0500.. is a 2-page (32-col) x 13-row metatile grid.
    A_TILES       = 0x0500
    TILE_ROWS     = 13
    TILE_COLS     = 16         # per page; two pages wrap -> 32 columns of history
    PLAYFIELD_TOP = 32         # pixels: top of the 13-row play area (tune if needed)
    def __init__(self, ram):
        self.ram = ram

    def _b(self, addr: int) -> int:
        # RAM is a numpy uint8 array; cast to a Python int so `*256` doesn't
        # overflow the uint8 (NumPy 2 raises OverflowError instead of promoting).
        return int(self.ram[addr])

    @property
    def mario_x(self) -> int:
        return self._b(self.A_PLAYER_PAGE) * 256 + self._b(self.A_PLAYER_X)

    @property
    def mario_y(self) -> int:
        return self._b(self.A_PLAYER_Y)

    @property
    def on_ground(self) -> bool:
        return self._b(self.A_PLAYER_STAT) == 0x00

    def enemies_x(self) -> list[int]:
        out = []
        for i in range(5):
            if self._b(self.A_ENEMY_DRAWN + i):
                out.append(self._b(self.A_ENEMY_PAGE + i) * 256 + self._b(self.A_ENEMY_X + i))
        return out

    def _tile(self, level_x: int, pixel_y: int) -> int:
        """Metatile id at a level position; 0 == empty/sky, nonzero == solid.
        Chrispresso-style index into the wrapping 0x0500 buffer."""
        page = (level_x // 256) % 2
        col = (level_x % 256) // TILE
        row = (pixel_y - self.PLAYFIELD_TOP) // TILE
        if row < 0 or row >= self.TILE_ROWS:
            return 0
        addr = self.A_TILES + page * (self.TILE_ROWS * self.TILE_COLS) + row * self.TILE_COLS + col
        return self._b(addr) if 0 <= addr - self.A_TILES < 2 * self.TILE_ROWS * self.TILE_COLS else 0

    def _is_pit(self, level_x: int) -> bool:
        """A column is a pit if the GROUND rows are empty (nothing to stand on). Probe
        FIXED near-floor rows, not rows relative to Mario's y -- a jump would else read
        sky as a hole. Rock-solid on 1-1's single flat ground line (castle-level
        footing-relative perception was tried and reverted: it made 1-1 flaky)."""
        bottom = self.PLAYFIELD_TOP + self.TILE_ROWS * TILE
        return all(self._tile(level_x, bottom - k * TILE) == 0 for k in (1, 2, 3))

    def _is_obstacle(self, level_x: int) -> bool:
        """A wall to jump (pipe, stair) = a solid tile just above the ground surface;
        the floating '?'/brick rows sit higher, so Mario runs under those."""
        bottom = self.PLAYFIELD_TOP + self.TILE_ROWS * TILE
        return any(self._tile(level_x, bottom - k * TILE) != 0 for k in (4, 5))

    def observe(self, lead_px: int = 0, enemy_lead_px: int = 0) -> dict:
        """Decode the current state into a compact dict (the model reads only the
        one-line `situation`).

        LATENCY COMPENSATION (for async): while the model thinks (~110 ms) Mario keeps
        moving, so by the time its choice is applied he has advanced ~1 tile and a
        just-in-time jump lands too late. Pass `lead_px` = how far he'll travel during
        one decision: we scan hazards from where he WILL be, so distances read smaller
        and the jump is chosen early enough to land on time. Enemies also walk toward
        him, so they get an extra `enemy_lead_px`. mario_tile/dist_to_goal stay real."""
        mtile = self.mario_x // TILE
        origin = self.mario_x + lead_px
        en_origin = self.mario_x + lead_px + enemy_lead_px

        # nearest pit ahead (scan up to 10 tiles forward) and its width
        gap_in = None
        gap_w = 0
        for d in range(1, 11):
            if self._is_pit(origin + d * TILE):
                gap_in = d
                w = 0
                while w < 16 and self._is_pit(origin + (d + w) * TILE):
                    w += 1
                gap_w = w
                break

        # nearest obstacle (pipe/wall) ahead, scan up to 6 tiles
        obstacle_in = None
        for d in range(1, 7):
            if self._is_obstacle(origin + d * TILE):
                obstacle_in = d
                break

        # nearest enemy ahead within 10 tiles
        enemy_in = None
        ahead = sorted(e - en_origin for e in self.enemies_x() if 0 < e - en_origin <= 10 * TILE)
        if ahead:
            enemy_in = ahead[0] // TILE

        gap_show = gap_in if gap_in is not None and gap_in <= 8 else None
        return {
            "mario_tile": mtile,
            "on_ground": self.on_ground,
            "dist_to_goal": max(0, FLAG_TILE - mtile),
            "gap_ahead_in": gap_show,
            "gap_width": gap_w if gap_show is not None else None,
            "obstacle_ahead_in": obstacle_in,
            "enemy_ahead_in": enemy_in,
            "situation": _nes_situation(gap_in, gap_w, enemy_in, obstacle_in),
        }


def _print_debug(ram: SMBRam) -> None:
    obs = ram.observe()
    print(f"  x={ram.mario_x} (tile {obs['mario_tile']})  y={ram.mario_y}  "
          f"on_ground={ram.on_ground}  enemies={ram.enemies_x()}", file=sys.stderr)
    print(f"  state={obs}", file=sys.stderr)


# ------------------------------------------------------------------------- #
#  CONTROL LOOP
# ------------------------------------------------------------------------- #
def _decide(agent, ram_snapshot, lead_px=0, enemy_lead_px=0):
    """One decision from a RAM snapshot. The model reads just the one-line
    `situation` (not the whole state dict): fewer prompt tokens, faster forward
    pass, no accuracy loss. lead_px/enemy_lead_px apply latency compensation.
    (--debug still prints the full decoded state.)"""
    obs = SMBRam(ram_snapshot).observe(lead_px=lead_px, enemy_lead_px=enemy_lead_px)
    return agent.choice(INSTRUCTION, {"situation": obs["situation"]}, NES_ACTIONS)


def _done_flag(step) -> bool:
    # robust across old gym (obs,reward,done,info) and gymnasium (..,terminated,truncated,info)
    return bool(step[2]) or (len(step) >= 5 and bool(step[3]))


def run_async(env, agent, args, amap):
    """Game runs at real FPS on its own thread-of-control; the model decides in a
    BACKGROUND thread and its latest choice is applied as it arrives (~9/sec on
    d3-edge). This is why the game is smooth even though one decision takes ~110 ms:
    the emulator never waits for the model. It's also how you'd really deploy it."""
    import threading

    shared = {"ram": None, "action": "wait", "stop": False}
    lock = threading.Lock()
    lat: list[float] = []
    # Latency compensation: while the model thinks, Mario keeps moving, so decide for
    # where he WILL be. lead = (walk speed) x (one decision's wall-time, in frames),
    # measured live so it adapts to the model's real latency. Enemies get ~double.
    vx = 1.4                                 # px/frame walking on the ground
    est_wall = 0.12                          # seconds; initial guess, then measured

    def brain():
        nonlocal est_wall
        while not shared["stop"]:
            with lock:
                ram = shared["ram"]
            if ram is None:
                time.sleep(0.003)
                continue
            lead = int(vx * est_wall * args.fps)     # how far ahead to look
            t0 = time.perf_counter()
            d = _decide(agent, ram, lead_px=lead, enemy_lead_px=lead)
            est_wall = 0.7 * est_wall + 0.3 * (time.perf_counter() - t0)   # smooth
            lat.append(d.latency_ms)
            with lock:
                shared["action"] = d.choice
            if args.debug:
                _print_debug(SMBRam(ram))

    th = threading.Thread(target=brain, daemon=True)
    th.start()

    period = 1.0 / args.fps
    driver = ButtonDriver(amap)
    reset = env.reset()
    state = reset[0] if isinstance(reset, tuple) else reset
    frame = 0
    won = False
    try:
        while frame < args.max_frames:
            t0 = time.perf_counter()
            on_ground = SMBRam(env.unwrapped.ram).on_ground
            with lock:
                shared["ram"] = env.unwrapped.ram.copy()
                action = shared["action"]
            step = env.step(driver.button(action, on_ground))
            state, info = step[0], step[-1]
            if not args.no_render:
                env.render()
            frame += 1
            if info.get("flag_get"):
                won = True
                break
            if _done_flag(step):
                reset = env.reset()
                state = reset[0] if isinstance(reset, tuple) else reset
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        shared["stop"] = True
        th.join(timeout=1.0)
    return lat, won, frame


def main() -> None:
    ap = argparse.ArgumentParser(
        description="A decision model (d3-edge) plays the real NES Super Mario Bros, "
                    "smoothly in real time (async, latency-compensated).")
    ap.add_argument("--level", default="1-1",
                    help="which stage, e.g. 1-1, 1-2, 4-1. Only 1-1 is tuned/beatable; "
                         "others load and run but die early (firebars, lava, moving platforms, "
                         "water, piranha plants -- mechanics this decoder doesn't model).")
    ap.add_argument("--debug", action="store_true", help="print the decoded state each decision")
    args = ap.parse_args()
    # fixed configuration (the rest of the knobs were removed): d3-edge brain,
    # smooth async, walking pace, real 60fps, emulator window on.
    args.fps = 60.0
    args.no_render = False
    args.max_frames = 1_000_000

    # Imports that require the native emulator + ROM are done here so the file
    # still imports (and --help works) without them installed.
    try:
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace
    except ImportError as e:
        sys.exit(f"{e}\n\nInstall the emulator first:  uv sync --extra nes   (and supply a ROM)")

    from decision import Decision3
    agent = Decision3()               # the open-weight decision model d3-edge

    try:
        env = gym_super_mario_bros.make(f"SuperMarioBros-{args.level}-v0", render_mode="human")
    except Exception as e:
        sys.exit(f"could not load level {args.level!r}: {e}\n(try 1-1 .. 8-4)")
    if args.level != "1-1":
        print(f"note: level {args.level} runs but is NOT tuned -- only 1-1 is beatable.",
              file=sys.stderr)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    amap = action_map(walk=True)
    lat, won, frame = run_async(env, agent, args, amap)

    med = sorted(lat)[len(lat) // 2] if lat else 0
    print(f"\n{'WON (flag reached)' if won else 'stopped'} | {len(lat)} decisions over "
          f"{frame} frames | median {med:.0f} ms/decision | backend {agent.model_id}",
          file=sys.stderr)
    env.close()


if __name__ == "__main__":
    main()

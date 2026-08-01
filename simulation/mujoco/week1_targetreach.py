#!/usr/bin/env python3
"""Week 1 — you name the coordinates, the FR5 tries to reach them.

`week1_reach.py` drives the target for you on a fixed circle. Here the target
is yours: type an xyz and watch the arm solve for it — or fail to, when you
pick somewhere it cannot go. The failure is worth as much as the success;
Week 4 has to detect fruit it cannot reach and skip it.

To place the tomato with the mouse instead, see `week1_mousereach.py`. The
reach loop underneath both of them is `reach.py`.

Two things differ from week1_reach.py, and both matter:

  1. **Physics is on.** week1_reach.py writes the IK answer straight into
     `data.qpos` — the arm teleports and gravity never enters. Here the IK
     answer is a *setpoint* for the six position servos, and `mj_step` makes
     the arm drag itself there against its own weight. So the arm lags, and
     under a heavy enough load it sags. That is why two errors get reported:

       ik    — how close the solver's answer is to the target (solver quality)
       arm   — how close the real arm ends up (solver + servos + gravity)

  2. **A reach envelope** is drawn as a faint blue shell. Inside it the arm has
     a chance; outside it there is no joint configuration that gets there.

Usage:

    # interactive: type coordinates in the terminal, arm goes there
    ./.venv/bin/python simulation/mujoco/week1_targetreach.py

    # one target, no window — prints whether it was reached
    ./.venv/bin/python simulation/mujoco/week1_targetreach.py --target 0.6 -0.2 0.7

    # a sequence, recorded for the weekly capture
    ./.venv/bin/python simulation/mujoco/week1_targetreach.py --headless \
        --target 0.6 0.0 0.8 --target 0.4 -0.45 0.35 --target 1.3 0.0 0.6 \
        --out week1_targetreach.mp4
"""

import argparse
import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr5 import (  # noqa: E402
    MAX_REACH,
    SHOULDER,
    TOOL_SITE,
    add_mocap_target,
    add_reach_envelope,
    build_fr5_spec,
    exit_without_teardown,
    reach_fraction,
)
from reach import (  # noqa: E402
    CTRL_DT,
    DEFAULT_SPEED,
    HOLD_S,
    REACHED_MM,
    STANDOFF,
    Reacher,
    attempt,
    describe,
    random_target,
)

# Where the tomato sits before you move it: out in front of the arm at roughly
# the height a truss would hang.
START_TARGET = np.array([0.55, 0.0, 0.55])


def build_scene(envelope: bool = True):
    """FR5 + a mocap ball you can move + the reach shell."""
    spec = build_fr5_spec()
    add_mocap_target(spec)
    if envelope:
        add_reach_envelope(spec)
    return spec.compile()


# ---------------------------------------------------------------------------
# interactive mode
# ---------------------------------------------------------------------------

HELP = """
────────────────────────────────────────────────────────────────────────
 Type here, in THIS terminal — not in the 3D window.
────────────────────────────────────────────────────────────────────────

  0.6 -0.2 0.7   go to this xyz, in metres. +x is out in front of the
                 base, +y is to its left, +z is up.
  random         a random point inside the envelope
  home           back to the starting point
  help           this text
  quit           close

The tool deliberately stops {standoff:.0f} cm short of the ball — that gap is where
a gripper's fingers will go in Week 2.

You can also drag the ball in the window: left-click it to select, then hold
Ctrl and right-drag. Press 2 in the window to hide the blue reach shell.
""".format(standoff=STANDOFF * 100)


def _stdin_reader(q: "queue.Queue[str]"):
    """Read commands on a background thread so the sim keeps running."""
    for line in sys.stdin:
        q.put(line.strip())
    q.put("quit")


def parse_command(line, rng):
    """-> (new_goal | None, message | None, quit?)"""
    line = line.strip().lower()
    if not line:
        return None, None, False
    if line in ("q", "quit", "exit"):
        return None, None, True
    if line in ("h", "help", "?"):
        return None, HELP, False
    if line == "home":
        return START_TARGET.copy(), None, False
    if line in ("r", "random"):
        return random_target(rng), None, False

    parts = line.replace(",", " ").split()
    if len(parts) != 3:
        return None, f"  don't understand {line!r} — need three numbers, or 'help'", False
    try:
        return np.array([float(p) for p in parts]), None, False
    except ValueError:
        return None, f"  don't understand {line!r} — need three numbers, or 'help'", False


def run_interactive(reacher, seed):
    """One loop that never stops: the arm always chases wherever the ball is.

    The ball moves for two reasons — you typed coordinates, or you dragged it
    in the window — and both are handled the same way. Whenever the goal moves,
    the timer restarts and the outcome gets reported once, not every tick.
    """
    import mujoco
    import mujoco.viewer

    model, data = reacher.model, reacher.data
    rng = np.random.default_rng(seed)
    commands: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(commands,), daemon=True).start()

    def prompt():
        print("\ntarget> ", end="", flush=True)

    goal = START_TARGET.copy()
    data.mocap_pos[0] = goal
    held = elapsed = 0.0
    reported = False
    ik_err = arm_err = float("nan")

    # Two control cycles per drawn frame = 50 fps, and the sleep below keeps
    # sim time equal to wall time.
    ctrl_per_frame = 2
    frame_s = ctrl_per_frame * CTRL_DT

    # Opening the window prints two or three lines of Wayland/libdecor noise
    # from deep inside GLFW. Printing the prompt first buries it under that
    # noise and it looks like the program never asked for input — so say
    # nothing until the window is up, then print. The Python-level GLFW
    # warning can be filtered outright; the C-level ones cannot.
    warnings.filterwarnings("ignore", module="glfw")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"\nFR5 reach: {MAX_REACH:.2f} m measured from the shoulder "
              f"at z={SHOULDER[2]:.3f}   ·   arm at "
              f"{reacher.speed * 100:.0f}% of rated joint speed")
        print(HELP)
        prompt()

        next_frame = time.perf_counter()
        while viewer.is_running():
            new_goal = None

            try:
                line = commands.get_nowait()
            except queue.Empty:
                pass
            else:
                new_goal, message, should_quit = parse_command(line, rng)
                if should_quit:
                    break
                if message:
                    print(message)
                if new_goal is None:
                    prompt()
                elif reach_fraction(new_goal) > 1.0:
                    print(f"  {reach_fraction(new_goal) * 100:.0f}% of reach — "
                          f"too far. Going anyway; watch it stretch and stop.")

            # Dragged in the window? The viewer records the drag but does not
            # apply it — in passive mode the physics loop owns `data`, so it is
            # our job to write the drag into mocap_pos. Then the ball itself
            # becomes the source of truth for where the goal is.
            if new_goal is None:
                with viewer.lock():
                    mujoco.mjv_applyPerturbPose(model, data, viewer.perturb, 0)
                dragged = data.mocap_pos[0].copy()
                if np.linalg.norm(dragged - goal) > 1e-3:
                    new_goal = dragged

            if new_goal is not None:
                goal = np.asarray(new_goal, dtype=float)
                held = elapsed = 0.0
                reported = False

            for _ in range(ctrl_per_frame):
                ik_err, arm_err = reacher.step(goal)
            elapsed += frame_s
            held = held + frame_s if arm_err * 1000 < REACHED_MM else 0.0

            if not reported and held >= HOLD_S:
                print(describe(attempt(goal, True, ik_err, arm_err, elapsed)))
                prompt()
                reported = True

            viewer.sync()
            next_frame += frame_s
            time.sleep(max(0.0, next_frame - time.perf_counter()))
            next_frame = max(next_frame, time.perf_counter() - frame_s)

    print("\nclosed.")
    exit_without_teardown()


# ---------------------------------------------------------------------------
# scripted mode (works headless, and is what the weekly capture uses)
# ---------------------------------------------------------------------------

def run_scripted(reacher, goals, out_path=None, dwell=1.0):
    import mujoco

    model, data = reacher.model, reacher.data
    writer = renderer = None
    fps = 30
    if out_path:
        import cv2
        width, height = 1280, 960
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {out_path}")
        renderer = mujoco.Renderer(model, height=height, width=width)

    frame_every = max(1, int(round((1 / fps) / CTRL_DT)))
    cycles = [0]

    def tick(_t):
        cycles[0] += 1
        if writer is not None and cycles[0] % frame_every == 0:
            import cv2
            renderer.update_scene(data, camera="demo")
            writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    results = []
    try:
        for goal in goals:
            r = reacher.drive_to(goal, on_tick=tick)
            print(describe(r))
            results.append(r)
            for _ in range(int(dwell / CTRL_DT)):   # hold, so the video reads
                reacher.step(goal)
                tick(0)
    finally:
        if renderer is not None:
            renderer.close()
        if writer is not None:
            writer.release()
            print(f"\nwrote {out_path}")

    hit = sum(r["reached"] for r in results)
    print(f"\n{hit}/{len(results)} targets reached")
    return results


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", action="append", nargs=3, type=float,
                    metavar=("X", "Y", "Z"),
                    help="a target to reach; repeat for a sequence. "
                         "Skips interactive mode.")
    ap.add_argument("--random", type=int, default=0, metavar="N",
                    help="append N random reachable targets")
    ap.add_argument("--seed", type=int, default=0, help="seed for --random")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                    help="fraction of the FR5's rated joint speed "
                         f"(default {DEFAULT_SPEED}; 1.0 is full speed)")
    ap.add_argument("--dwell", type=float, default=1.0,
                    help="seconds to hold each target (scripted mode)")
    ap.add_argument("--headless", action="store_true",
                    help="no window; use with --out to record")
    ap.add_argument("--out", default=None, help="write an mp4 (implies scripted)")
    ap.add_argument("--no-envelope", action="store_true",
                    help="hide the reach shell")
    args = ap.parse_args()

    if args.headless or args.out:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    model = build_scene(envelope=not args.no_envelope)
    data = mujoco.MjData(model)
    reacher = Reacher(model, data, speed=args.speed)

    goals = [np.array(t) for t in (args.target or [])]
    if args.random:
        rng = np.random.default_rng(args.seed)
        goals += [random_target(rng) for _ in range(args.random)]

    print(f"Fairino FR5: {model.nq} DoF, tracking site '{TOOL_SITE}', "
          f"physics on, {args.speed * 100:.0f}% joint speed")

    if goals or args.headless or args.out:
        if not goals:
            ap.error("headless mode needs at least one --target or --random N")
        run_scripted(reacher, goals, args.out, args.dwell)
    else:
        run_interactive(reacher, args.seed)


if __name__ == "__main__":
    main()

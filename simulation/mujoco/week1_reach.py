#!/usr/bin/env python3
"""Week 1 — the FR5 tool tracks a moving target using differential IK.

This is the Week 1 milestone: prove the arm can be driven to an arbitrary XYZ.
The arm is the Fairino FR5 loaded from Fairino's own URDF (see fr5.py), which
is the arm in the Vinea MVP spec — so nothing here gets thrown away later.

Motion planning is `mink` differential IK, not MoveIt 2, which has no binaries
for ROS 2 Lyrical. See the roadmap's "Motion planning" section.

This script is *kinematics only*: it writes the IK answer straight into
`data.qpos` and calls `mj_forward`. No `mj_step`, so no gravity, no contact,
no motors — the arm teleports to wherever IK says. That is why the tracking
error here is tiny. `week1_targetreach.py` and `week1_mousereach.py` run the
same IK through the actual position servos with physics on, and the error
there is honest.

    # interactive window (needs a display)
    ./.venv/bin/python simulation/mujoco/week1_reach.py

    # no display: writes a video for the weekly update
    ./.venv/bin/python simulation/mujoco/week1_reach.py --headless --out week1.mp4
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr5 import (  # noqa: E402
    TOOL_SITE, add_mocap_target, build_fr5_spec, exit_without_teardown,
)

# The target sweeps a circle in front of the arm — roughly where a truss of
# tomatoes would hang relative to a robot parked in the aisle.
CIRCLE_CENTRE = np.array([0.55, 0.0, 0.55])
CIRCLE_RADIUS = 0.18
CIRCLE_PERIOD_S = 8.0


def build_scene():
    """The FR5 with a mocap target marker added."""
    spec = build_fr5_spec()
    add_mocap_target(spec, rgba=(0.9, 0.2, 0.15, 0.6))
    return spec.compile()


def make_ik(model):
    """A mink position task on the FR5 tool frame."""
    import mink

    config = mink.Configuration(model)
    task = mink.FrameTask(
        frame_name=TOOL_SITE, frame_type="site",
        position_cost=1.0,
        orientation_cost=0.0,   # Week 1: position only. Orientation matters in Week 2.
        lm_damping=1.0,
    )
    # Keeps the solver from wandering into strange joint configurations.
    posture = mink.PostureTask(model, cost=1e-2)
    return config, [task, posture]


def target_at(t):
    theta = 2.0 * np.pi * t / CIRCLE_PERIOD_S
    return CIRCLE_CENTRE + CIRCLE_RADIUS * np.array([0.0, np.cos(theta), np.sin(theta)])


def step_ik(config, tasks, data, model, t, dt):
    """Advance IK one tick and push the result into the sim."""
    import mink
    import mujoco

    goal = target_at(t)
    tasks[0].set_target(mink.SE3.from_translation(goal))
    data.mocap_pos[0] = goal

    vel = mink.solve_ik(config, tasks, dt, "daqp", 1e-3)
    config.integrate_inplace(vel, dt)
    data.qpos[: model.nq] = config.q
    mujoco.mj_forward(model, data)

    return float(np.linalg.norm(data.site(TOOL_SITE).xpos - goal))


def run_interactive(model, data, config, tasks, dt):
    import mujoco.viewer

    print("Viewer open. Red sphere = IK target; the FR5 tool should track it.")
    print("Close the window or press Ctrl-C to quit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        while viewer.is_running():
            err = step_ik(config, tasks, data, model, t, dt)
            viewer.sync()
            t += dt
            if abs(t % 1.0) < dt:
                print(f"  t={t:5.1f}s   tracking error {err * 1000:5.1f} mm")

    # MuJoCo 3.10's viewer segfaults on GLFW teardown here; see fr5.py.
    exit_without_teardown()


def run_headless(model, data, config, tasks, dt, out_path, seconds):
    import cv2
    import mujoco

    width, height, fps = 1280, 960, 30
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")

    # Skip the first second: the arm starts at home and has to seek the target,
    # so early error says nothing about tracking quality.
    settled = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        t = 0.0
        for i in range(int(seconds * fps)):
            for _ in range(max(1, int((1 / fps) / dt))):
                err = step_ik(config, tasks, data, model, t, dt)
                if t > 1.0:
                    settled.append(err)
                t += dt
            renderer.update_scene(data, camera="demo")
            writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            if i % fps == 0:
                print(f"  rendered {i // fps}/{int(seconds)}s")
    writer.release()
    steady = max(settled) * 1000 if settled else float("nan")
    print(f"\nWrote {out_path}  (steady-state tracking error {steady:.1f} mm)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headless", action="store_true", help="render to video instead of a window")
    ap.add_argument("--out", default="week1.mp4", help="output video path (headless only)")
    ap.add_argument("--seconds", type=float, default=12.0, help="video length (headless only)")
    args = ap.parse_args()

    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    import mujoco

    model = build_scene()
    data = mujoco.MjData(model)
    config, tasks = make_ik(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    config.update(data.qpos[: model.nq].copy())
    tasks[1].set_target_from_configuration(config)

    dt = 0.002
    print(f"Fairino FR5: {model.nq} DoF, tracking site '{TOOL_SITE}'")

    if args.headless:
        run_headless(model, data, config, tasks, dt, args.out, args.seconds)
    else:
        run_interactive(model, data, config, tasks, dt)


if __name__ == "__main__":
    main()

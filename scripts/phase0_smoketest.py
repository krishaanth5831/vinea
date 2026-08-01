#!/usr/bin/env python3
"""Phase 0 toolchain smoke test.

Verifies the dev machine can actually run the Vinea stack: MuJoCo physics +
offscreen render, the CV stack (torch/ultralytics/OpenCV), and the ROS 2
Python layer. Run this after any machine rebuild or distro upgrade.

    ./.venv/bin/python scripts/phase0_smoketest.py

Exits non-zero if any check fails.
"""

import os
import sys

# MuJoCo needs an offscreen GL backend when there is no display attached.
os.environ.setdefault("MUJOCO_GL", "egl")

RESULTS = []


def check(name):
    """Decorator: run a check, record pass/fail, never raise."""

    def wrap(fn):
        try:
            detail = fn()
            RESULTS.append((True, name, detail))
        except Exception as exc:  # noqa: BLE001 - a smoke test reports, never crashes
            RESULTS.append((False, name, f"{type(exc).__name__}: {exc}"))
        return fn

    return wrap


# A free-falling box: enough to prove the physics step and the renderer work.
MJCF = """
<mujoco model="vinea_smoketest">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="ground" type="plane" size="2 2 0.1" rgba="0.3 0.4 0.3 1"/>
    <body name="tomato" pos="0 0 0.5">
      <freejoint/>
      <geom name="tomato" type="sphere" size="0.04" rgba="0.8 0.15 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


@check("MuJoCo physics step")
def _mujoco_physics():
    import mujoco

    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)
    z_start = float(data.qpos[2])
    for _ in range(200):
        mujoco.mj_step(model, data)
    z_end = float(data.qpos[2])
    assert z_end < z_start, f"sphere did not fall ({z_start:.3f} -> {z_end:.3f})"
    return f"mujoco {mujoco.__version__}, sphere fell {z_start:.2f}m -> {z_end:.2f}m"


@check("MuJoCo offscreen render")
def _mujoco_render():
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=120, width=160) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()
    assert frame.shape == (120, 160, 3), f"unexpected frame shape {frame.shape}"
    assert np.asarray(frame).any(), "rendered frame is entirely black"
    return f"{frame.shape[1]}x{frame.shape[0]} RGB via MUJOCO_GL={os.environ['MUJOCO_GL']}"


@check("URDF loads in MuJoCo")
def _urdf_loads():
    """MuJoCo reads URDF directly, so the Phase 1 arm can be dropped straight in."""
    import mujoco

    urdf = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "simulation", "urdf", "vinea_arm.urdf.xacro",
    )
    if not os.path.exists(urdf):
        return "skipped - no URDF yet"
    # xacro macros are not expanded here; this only proves the importer path works
    # for a plain-URDF file. Once the real arm exists, expand xacro first.
    with open(urdf) as fh:
        src = fh.read()
    if "xacro:" in src:
        return "skipped - file still uses xacro macros (expand before loading)"
    model = mujoco.MjModel.from_xml_string(src)
    return f"{model.nbody} bodies, {model.njnt} joints"


# A 2-link planar arm with an end-effector site: the minimum needed to prove
# the IK path works. The real arm comes from MuJoCo Menagerie in Week 1.
ARM_MJCF = """
<mujoco model="vinea_ik_check">
  <compiler angle="radian"/>
  <worldbody>
    <light pos="0 0 2"/>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.025"/>
        <site name="ee" pos="0.3 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@check("mink IK solve (MoveIt replacement)")
def _mink_ik():
    """Option A: differential IK instead of MoveIt 2, which has no Lyrical binaries."""
    import mink
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(ARM_MJCF)
    data = mujoco.MjData(model)
    config = mink.Configuration(model)
    config.update(np.zeros(model.nq))

    target = np.array([0.35, 0.0, 0.25])
    task = mink.FrameTask(
        frame_name="ee", frame_type="site",
        position_cost=1.0, orientation_cost=0.0,
    )
    tf = mink.SE3.from_translation(target)
    task.set_target(tf)

    dt = 0.02
    for _ in range(400):
        vel = mink.solve_ik(config, [task], dt, "daqp", 1e-3)
        config.integrate_inplace(vel, dt)

    data.qpos[:] = config.q
    mujoco.mj_forward(model, data)
    reached = data.site("ee").xpos.copy()
    err = float(np.linalg.norm(reached - target))
    assert err < 0.02, f"IK did not converge: error {err:.4f} m, reached {reached}"
    return f"mink converged to {err * 1000:.1f} mm of target"


@check("CV stack")
def _cv_stack():
    import cv2
    import torch
    import ultralytics

    dev = "CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    # Prove a real tensor op runs on the selected device.
    t = torch.rand(64, 64, device="cuda" if torch.cuda.is_available() else "cpu")
    _ = (t @ t).sum().item()
    return f"torch {torch.__version__} ({dev}), ultralytics {ultralytics.__version__}, cv2 {cv2.__version__}"


@check("ROS 2 Python layer")
def _ros2():
    import rclpy
    from rclpy.node import Node

    rclpy.init()
    try:
        node = Node("vinea_smoketest")
        node.create_publisher(__import__("std_msgs.msg", fromlist=["String"]).String, "smoke", 10)
        name = node.get_name()
        node.destroy_node()
    finally:
        rclpy.shutdown()
    import rclpy as _r

    return f"rclpy {_r.__name__} ok, node '{name}' created and torn down"


def main():
    print("Vinea Phase 0 toolchain smoke test")
    print("=" * 58)
    width = max(len(n) for _, n, _ in RESULTS)
    for ok, name, detail in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
    print("=" * 58)
    failed = [n for ok, n, _ in RESULTS if not ok]
    if failed:
        print(f"{len(failed)} check(s) failed: {', '.join(failed)}")
        return 1
    print(f"All {len(RESULTS)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

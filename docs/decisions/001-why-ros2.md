# ADR 001 — Why ROS 2

**Date:** 2026-06-19
**Status:** Decided

## Decision

Use ROS 2 (Humble) as the robotics middleware for all simulation and future hardware work.

## Reasons

- Industry standard for agricultural robotics — integrates with MoveIt 2, Gazebo, nav2 out of the box
- Long-term support (Humble LTS until 2027)
- Large ecosystem: pre-built packages for perception, motion planning, hardware drivers
- What potential technical co-founders will already know
- SAIA and other agri-robotics players are ROS-based — easier to hire into later

## Rejected Alternatives

- **ROS 1** — End of life 2025, no reason to start on it
- **Custom middleware** — No justification at this stage; adds complexity with no benefit
- **Pure Python with no middleware** — Fine for scripts, not for a multi-sensor robot system

## Implications

- Simulation: Gazebo Fortress (native ROS 2 integration)
- Motion planning: MoveIt 2
- All packages follow ROS 2 conventions (ament_cmake or ament_python)

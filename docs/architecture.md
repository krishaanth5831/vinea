# Vinea — System Architecture

> **Status: the original June 2026 plan, superseded.** This describes the
> ROS 2 + Gazebo + MoveIt 2 layout the repo started with. None of it was built:
> the simulation runs in MuJoCo with `mink` differential IK, as plain Python in
> `simulation/mujoco/` (see the README's Project layout). The package skeletons
> named below are in `ros2/` and are empty. Kept as the intended split for a
> hardware phase.

## High-Level Overview

```
┌─────────────────────────────────────────────────────┐
│                   vinea_bringup                      │
│           (top-level launch, orchestration)          │
└────────┬──────────────┬──────────────┬──────────────┘
         │              │              │
         ▼              ▼              ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────────┐
│vinea_descrip-│ │vinea_control│ │ vinea_perception  │
│    tion      │ │             │ │                   │
│ URDF, meshes │ │ Controllers │ │ Camera processing │
│ robot model  │ │ Trajectory  │ │ Fruit detection   │
└──────────────┘ └──────┬──────┘ └────────┬─────────┘
                        │                  │
                        ▼                  ▼
                ┌──────────────────────────────┐
                │       vinea_planning          │
                │  MoveIt2 — pick/place logic   │
                └──────────────────────────────┘
                        │
                        ▼
                ┌──────────────┐
                │vinea_scouting│
                │ Inspection   │
                │ Anomaly log  │
                └──────────────┘
```

## ROS 2 Packages (empty skeletons in `ros2/`)

| Package | Role |
|---|---|
| `vinea_bringup` | Top-level launch files — starts everything |
| `vinea_description` | URDF, meshes, robot description |
| `vinea_control` | Arm controllers, trajectory execution (ros2_control) |
| `vinea_perception` | Camera processing, fruit detection (YOLOv8) |
| `vinea_planning` | MoveIt2 integration, pick/place logic |
| `vinea_scouting` | Inspection pass logic, anomaly detection + logging |

## Simulation Stack

- **Gazebo Fortress** — physics simulation, greenhouse world
- **RViz 2** — visualisation, debugging
- **MoveIt 2** — motion planning
- World file: `simulation/gazebo/worlds/greenhouse_row.world` (never created; the Gazebo folders were removed)

## Sensing Stack

> To be defined during prototype phase. Candidates: RGB-D camera (Intel RealSense), LiDAR for navigation.

## CV Pipeline

> Planned: YOLOv8 (Ultralytics) for tomato detection. What was built instead: `simulation/mujoco/detect.py` scores an HSV colour detector against a stock, un-fine-tuned `yolo11n` checkpoint (`models/weights/yolo11n.pt`, not committed — see the README's Setup).

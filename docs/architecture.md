# Vinea — System Architecture

> **Status:** Placeholder — to be updated as simulation develops.

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

## ROS 2 Packages

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
- World file: `simulation/gazebo/worlds/greenhouse_row.world`

## Sensing Stack

> To be defined during prototype phase. Candidates: RGB-D camera (Intel RealSense), LiDAR for navigation.

## CV Pipeline

> YOLOv8 (Ultralytics) for tomato detection. Models stored in `models/tomato_detector/`. Large model files (.pt, .onnx) are not committed — use download script.

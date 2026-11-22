# ROS 2 package skeletons (unbuilt)

These six folders are empty placeholders from the repo's first day (June 2026),
when the plan was ROS 2 + Gazebo + MoveIt 2. That plan changed: the simulation
moved to MuJoCo with `mink` for inverse kinematics, and **nothing in this repo
uses ROS today**. Every demo runs as a plain Python script from
`simulation/mujoco/`.

They are kept as the intended package split for a future hardware phase, where
ROS 2 earns its place (real drivers, several processes, logging):

| Package | Intended role |
|---|---|
| `vinea_bringup` | top-level launch files |
| `vinea_description` | robot description (URDF, meshes) |
| `vinea_control` | arm controllers, trajectory execution |
| `vinea_perception` | camera processing, fruit detection |
| `vinea_planning` | pick/place logic |
| `vinea_scouting` | inspection pass, anomaly logging |

None of them contains a `package.xml`, so `colcon build` finds nothing to build
here. `scripts/setup.sh` and `scripts/run_sim.sh` belong to the same original
plan and do not run the MuJoCo simulation. See `docs/decisions/001-why-ros2.md`
and `docs/architecture.md` for the original reasoning.

#!/bin/bash
# Vinea — Launch Gazebo Simulation (original June 2026 plan, NOT the simulation)
#
# ⚠️ Gazebo was dropped for MuJoCo and `vinea_bringup` has no launch file, so
# this fails as it stands. The simulation is run with plain Python scripts in
# simulation/mujoco/ — see the README. Kept for reference; see ros2/README.md.

set -e

source /opt/ros/humble/setup.bash
source install/setup.bash

echo "Launching Vinea greenhouse simulation..."

ros2 launch vinea_bringup simulation.launch.py

#!/bin/bash
# Vinea — Launch Gazebo Simulation

set -e

source /opt/ros/humble/setup.bash
source install/setup.bash

echo "Launching Vinea greenhouse simulation..."

ros2 launch vinea_bringup simulation.launch.py

#!/bin/bash
# Vinea — ROS 2 workspace setup (original June 2026 plan, NOT the simulation)
#
# ⚠️ The MuJoCo simulation does not use ROS. To run it, follow "Setup" in the
# README instead (a Python venv + requirements.txt). This script sources ROS 2
# Humble and runs colcon from the repo root; the Vinea packages in ros2/ are
# empty skeletons, so nothing of Vinea's own gets built. Kept for the hardware
# phase; see ros2/README.md.

set -e

echo "Setting up Vinea development environment..."

# Source ROS 2
source /opt/ros/humble/setup.bash

# Install Python dependencies
pip install -r requirements.txt 2>/dev/null || echo "No requirements.txt found — skipping"

# Build ROS 2 packages
colcon build --symlink-install

# Source the workspace
source install/setup.bash

echo "Done. Run 'source install/setup.bash' to activate the workspace."

#!/bin/bash
# Vinea — Environment Setup Script
# Run once after cloning the repo

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

#!/usr/bin/env bash
# =============================================================================
# run_carotid_sim.sh
# Launches the full carotid catheterization simulation.
# Sets CUDA library paths needed by OpenCV (cv2) on this system.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_WS="$SCRIPT_DIR/ros2_ws"

# ── CUDA library resolution ───────────────────────────────────────────────────
# OpenCV (python3-opencv) compiled with CUDA needs these shared libs at runtime.
CUDA_LIBS=""
for d in $(find /home/admin1/anaconda3/lib/python3.12/site-packages/nvidia \
              -maxdepth 2 -name "lib" -type d 2>/dev/null); do
  CUDA_LIBS="$CUDA_LIBS:$d"
done
# NPP from .local
if [ -d "/home/admin1/.local/lib/python3.12/site-packages/nvidia/npp/lib" ]; then
  CUDA_LIBS="$CUDA_LIBS:/home/admin1/.local/lib/python3.12/site-packages/nvidia/npp/lib"
fi

export LD_LIBRARY_PATH="${CUDA_LIBS#:}:${LD_LIBRARY_PATH:-}"

# ── Fix for Conda libstdc++ vs System libstdc++ for ROS2 rclpy ─────────────────
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# ── ROS2 setup ───────────────────────────────────────────────────────────────
export GZ_SIM_RESOURCE_PATH="$ROS2_WS/install/carotid_robot_description/share:${GZ_SIM_RESOURCE_PATH:-}"
source /opt/ros/jazzy/setup.bash
source "$ROS2_WS/install/setup.bash"

# ── Parse args ────────────────────────────────────────────────────────────────
MODEL_BACKEND="${1:-unet}"
RATE_HZ="${2:-30.0}"
RVIZ="${3:-true}"

echo "================================================================="
echo "  Carotid Catheterization Simulation"
echo "  Backend : $MODEL_BACKEND"
echo "  Rate    : $RATE_HZ Hz"
echo "  RViz    : $RVIZ"
echo "================================================================="

exec ros2 launch carotid_bringup carotid_sim.launch.py \
  model_backend:="$MODEL_BACKEND" \
  rate_hz:="$RATE_HZ" \
  rviz:="$RVIZ"

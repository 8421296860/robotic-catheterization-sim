#!/usr/bin/env python3
"""
Planar Arm Tracker
==================
- Perception node detects vessel centroid → publishes (x_m, y_m) offset in metres
- This node maps that to a 3D target point in the world XY plane
- Publishes a bright RED SPHERE marker at that point (visible in Gazebo + RViz)
- Solves 2-link planar IK to move the arm end-effector to that point
- Result: arm rotates smoothly in the horizontal plane tracking the detected point

Geometry:
  L1 = 0.40 m  (link_1 length)
  L2 = 0.35 m  (link_2 length)
  Arm base at world origin.
  Target = (tx, ty) in world XY plane at Z = 0.12 m (arm height).
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

L1 = 0.40   # link 1 length (m)
L2 = 0.35   # link 2 length (m)
R_MIN = abs(L1 - L2) + 0.01   # min reachable radius
R_MAX = L1 + L2 - 0.01        # max reachable radius

# Map perception output (metres offset from image centre) → arm workspace
# Perception gives ±~3 cm offsets; arm workspace is 0.15–0.75 m radius.
# We map: lateral (x) → world X, depth (y) → world Y, scale to workspace.
SCALE = 5.0       # amplify small perception offsets into workspace range
TARGET_BIAS_X = 0.45   # default forward reach when no target
TARGET_BIAS_Y = 0.0

SERVO_DT_NS = 100_000_000   # 100 ms per step (10 Hz, smooth)


def solve_ik(tx: float, ty: float):
    """
    2-link planar IK.
    Returns (q1, q2) in radians, or None if out of reach.
    """
    r = math.hypot(tx, ty)
    r = max(R_MIN, min(R_MAX, r))   # clamp to reachable

    cos_q2 = (r**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    q2 = math.atan2(math.sqrt(max(0.0, 1.0 - cos_q2**2)), cos_q2)

    k1 = L1 + L2 * cos_q2
    k2 = L2 * math.sin(q2)
    q1 = math.atan2(ty, tx) - math.atan2(k2, k1)

    return q1, q2


class PlanarArmTracker(Node):

    def __init__(self):
        super().__init__('carotid_motion_planner_node')

        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')

        # Subscribe to perception centroid
        self.sub = self.create_subscription(
            PoseStamped, '/carotid/target_pose', self._target_cb, 10)

        # Publisher: visible RED sphere at detected point in Gazebo/RViz
        self.pub_marker = self.create_publisher(Marker, '/detected_point', 10)
        self.pub_status  = self.create_publisher(String, '/arm/status', 10)

        # Last known target in world frame
        self._tx = TARGET_BIAS_X
        self._ty = TARGET_BIAS_Y
        self._state = 'WAIT'

        # Poll for action server
        self._timer = self.create_timer(0.5, self._check_server)
        self.get_logger().info('Planar Arm Tracker — waiting for action server...')

    # ── Server ready ──────────────────────────────────────────────────────
    def _check_server(self):
        if not self.traj_client.server_is_ready():
            return
        self._timer.cancel()
        self.get_logger().info('Action server ready — arm tracking active.')
        self._state = 'TRACKING'
        # Send initial neutral pose so arm doesn't jerk
        self._send_joints(0.0, 0.0, self._tx, self._ty, duration_ns=2_000_000_000)

    # ── Perception callback ───────────────────────────────────────────────
    def _target_cb(self, msg: PoseStamped):
        # perception publishes (x=lateral offset m, y=depth offset m)
        lat = msg.pose.position.x   # left/right in image → world X offset
        dep = msg.pose.position.y   # up/down in image    → world Y offset

        # Map to world coordinates — arm sits at origin, target in XY plane
        self._tx = TARGET_BIAS_X + SCALE * lat
        self._ty = TARGET_BIAS_Y - SCALE * dep

        # Clamp to workspace
        r = math.hypot(self._tx, self._ty)
        if r > R_MAX:
            scale = R_MAX / r
            self._tx *= scale
            self._ty *= scale
        elif r < R_MIN:
            scale = R_MIN / r if r > 0 else 1.0
            self._tx *= scale
            self._ty *= scale

        # ── Publish RED SPHERE at detected world point ──────────────────
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'detected_point'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = self._tx
        m.pose.position.y = self._ty
        m.pose.position.z = 0.13   # same height as arm plane
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.06   # 6 cm sphere, very visible
        m.color.r = 1.0
        m.color.g = 0.1
        m.color.b = 0.0
        m.color.a = 1.0
        m.lifetime.sec = 1
        self.pub_marker.publish(m)

        if self._state != 'TRACKING':
            return

        # ── Solve IK and send to arm + Gazebo target ─────────────────────
        q1, q2 = solve_ik(self._tx, self._ty)
        self._send_joints(q1, q2, self._tx, self._ty, duration_ns=SERVO_DT_NS)
        self.pub_status.publish(String(data=f'TRACKING tx={self._tx:.3f} ty={self._ty:.3f}'))

    def _send_joints(self, q1: float, q2: float, tx: float, ty: float, duration_ns: int = SERVO_DT_NS):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['joint_1', 'joint_2', 'target_x_joint', 'target_y_joint']
        pt = JointTrajectoryPoint()
        pt.positions = [q1, q2, tx, ty]
        pt.velocities = [0.0, 0.0, 0.0, 0.0]
        pt.time_from_start = Duration(sec=0, nanosec=duration_ns)
        goal.trajectory.points.append(pt)
        self.traj_client.send_goal_async(goal)


def main(args=None):
    rclpy.init(args=args)
    node = PlanarArmTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

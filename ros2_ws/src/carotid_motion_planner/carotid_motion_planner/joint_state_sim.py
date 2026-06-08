#!/usr/bin/env python3
"""
carotid_motion_planner.joint_state_sim
========================================
Publishes simulated joint states at a configurable rate when there
is no hardware/Gazebo.  Useful for debugging with RViz2 without
the full motion planner running.

Publishes: /joint_states  sensor_msgs/JointState
"""

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

JOINT_NAMES = [
    'joint_base_rot',
    'joint_shoulder',
    'joint_elbow',
    'joint_wrist_pitch',
    'joint_wrist_roll',
]


class JointStateSimNode(Node):
    """Simulates gentle sinusoidal joint motion for RViz preview."""

    def __init__(self):
        super().__init__('joint_state_sim')
        self.declare_parameter('rate_hz', 30.0)
        rate = self.get_parameter('rate_hz').value

        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.t0  = self.get_clock().now().nanoseconds * 1e-9
        self.timer = self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info('JointStateSim: sinusoidal demo mode')

    def _publish(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        t   = now - self.t0

        # Gentle sweep to show the arm is alive
        positions = [
            0.3 * math.sin(0.2 * t),          # base yaw
            -0.3 + 0.2 * math.sin(0.15 * t),  # shoulder
            0.6 + 0.2 * math.sin(0.1 * t),    # elbow
            0.3 + 0.1 * math.sin(0.25 * t),   # wrist pitch
            0.2 * math.sin(0.3 * t),           # wrist roll
        ]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = JOINT_NAMES
        msg.position = positions
        msg.velocity = [0.0] * 5
        msg.effort   = [0.0] * 5
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
carotid_bringup/launch/display_robot.launch.py
===============================================
Minimal launch: just robot_state_publisher + joint_state_sim + RViz2.
Use this to verify the URDF loads correctly without perception.

Usage:
  ros2 launch carotid_bringup display_robot.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    desc_pkg  = get_package_share_directory('carotid_robot_description')
    urdf_path = os.path.join(desc_pkg, 'urdf', 'carotid_arm.urdf')
    rviz_path = os.path.join(desc_pkg, 'rviz', 'carotid_sim.rviz')

    with open(urdf_path, 'r') as f:
        robot_description_content = f.read()

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_content}],
        output='screen',
    )

    js_sim = Node(
        package='carotid_motion_planner',
        executable='joint_state_sim',
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_path],
        output='screen',
    )

    return LaunchDescription([rsp, js_sim, rviz])

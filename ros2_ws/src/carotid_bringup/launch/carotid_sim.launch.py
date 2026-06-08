#!/usr/bin/env python3
"""
carotid_bringup/launch/carotid_sim.launch.py
=============================================
Full system launch: perception + robot state + motion planner + RViz2.

Usage:
  ros2 launch carotid_bringup carotid_sim.launch.py

Optional args:
  model_backend:=unet|nnunet        (default: unet)
  model_path:=<path>                (pth or nnUNet results dir)
  rviz:=true|false                  (default: true)
  image_dir:=<path>                 (override dataset dir)
  rate_hz:=1.0                      (image publisher rate)
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    OpaqueFunction,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    Command,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # ── Package paths ─────────────────────────────────────────────────────────
    desc_pkg   = get_package_share_directory('carotid_robot_description')
    bringup_pkg = get_package_share_directory('carotid_bringup')

    urdf_path = os.path.join(desc_pkg, 'urdf', 'carotid_arm.urdf')
    rviz_path = os.path.join(desc_pkg, 'rviz', 'carotid_sim.rviz')

    with open(urdf_path, 'r') as f:
        robot_description_content = f.read()

    default_image_dir = (
        '/home/admin1/Music/prep/cyint/carotid-segmentation-main/src/data'
        '/Common Carotid Artery Ultrasound Images/US images'
    )

    # ── Declare launch arguments ──────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('model_backend', default_value='unet',
                              description='unet or nnunet'),
        DeclareLaunchArgument('model_path',    default_value='',
                              description='Path to model weights or nnUNet results'),
        DeclareLaunchArgument('rviz',          default_value='true',
                              description='Launch RViz2'),
        DeclareLaunchArgument('image_dir',     default_value=default_image_dir,
                              description='Dataset image directory'),
        DeclareLaunchArgument('rate_hz',       default_value='1.0',
                              description='Image publisher rate'),
        DeclareLaunchArgument('confidence_thr', default_value='0.5',
                              description='Segmentation binarisation threshold'),
    ]

    urdf_file = os.path.join(get_package_share_directory('carotid_robot_description'), 'urdf', 'catheter_sim.urdf')
    robot_description_content = Command(['xacro ', urdf_file])

    # ── Nodes ─────────────────────────────────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': ParameterValue(robot_description_content, value_type=str),
                     'use_sim_time': True}],
    )

    image_publisher = Node(
        package='carotid_perception',
        executable='image_publisher',
        name='ultrasound_image_publisher',
        output='screen',
        parameters=[{
            'image_dir': LaunchConfiguration('image_dir'),
            'rate_hz':   LaunchConfiguration('rate_hz'),
            'loop':      True,
        }],
    )

    perception_node = Node(
        package='carotid_perception',
        executable='perception_node',
        name='carotid_perception_node',
        output='screen',
        parameters=[{
            'model_backend':   LaunchConfiguration('model_backend'),
            'model_path':      LaunchConfiguration('model_path'),
            'confidence_thr':  LaunchConfiguration('confidence_thr'),
            'debug_overlay':   True,
            'image_scale_m':   0.0001,
        }],
    )

    motion_planner = Node(
        package='carotid_motion_planner',
        executable='motion_planner_node',
        name='carotid_motion_planner_node',
        output='screen',
        parameters=[{
            'control_rate_hz': 50.0,
            'approach_height': 0.20,
            'contact_height':  0.05,
        }],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_path],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    # ── Gazebo and Controllers ────────────────────────────────────────────────
    gz_pkg = get_package_share_directory('ros_gz_sim')
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r -v 2 empty.sdf'}.items(),
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description',
                   '-name', 'arm',
                   '-z', '0.0'],
        output='screen',
    )

    load_jsb = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    load_jtc = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller'],
        output='screen',
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    return LaunchDescription(
        args + [
            LogInfo(msg='=== Carotid Robotic Arm Simulation Starting ==='),
            gazebo_launch,
            robot_state_publisher,
            spawn_robot,
            clock_bridge,
            TimerAction(
                period=5.0,
                actions=[load_jsb, load_jtc, image_publisher, perception_node, motion_planner]
            ),
            rviz_node,
        ]
    )

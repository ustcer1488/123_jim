#!/usr/bin/env python3
"""
auditnav.launch.py — One-command launch for the full AuditNav system.

After building the package (colcon build), launch with:
    ros2 launch auditnav auditnav.launch.py

Optional overrides:
    ros2 launch auditnav auditnav.launch.py use_sim_time:=false
    ros2 launch auditnav auditnav.launch.py config:=/abs/path/to/params.yaml

Start order (enforced by TimerAction):
  t=0s  occupancy_map_node          — must publish /map first
  t=3s  open_vocab_perception_node  — needs camera topics
  t=3s  semantic_memory_node        — loads ChromaDB + embedding model
  t=3s  instruction_parser_node     — lightweight LLM middleware
  t=6s  topo_nav_node               — needs /map + all upstream nodes
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_PKG = 'auditnav'

_DEFAULT_CFG = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'config', 'default_params.yaml'
))


def _node(executable: str, name: str) -> Node:
    return Node(
        package=_PKG,
        executable=executable,
        name=name,
        output='screen',
        emulate_tty=True,
        additional_env={
            'AUDITNAV_CONFIG': LaunchConfiguration('config'),
            'SILICONFLOW_API_KEY': os.environ.get('SILICONFLOW_API_KEY', ''),
        },
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=_DEFAULT_CFG,
                              description='Path to YAML parameter file.'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use /clock. Set false for real robot.'),

        LogInfo(msg='╔══════════════════════════════════════════════╗'),
        LogInfo(msg='║       AuditNav  —  Full System Launch         ║'),
        LogInfo(msg='╚══════════════════════════════════════════════╝'),

        # t = 0s: LiDAR mapper (must be first — publishes /map)
        _node('occupancy_map_node', 'occupancy_map_node'),

        # t = 3s: perception + memory + commander (parallel)
        TimerAction(period=3.0, actions=[
            LogInfo(msg='[AuditNav] Starting perception / memory / commander...'),
            _node('open_vocab_perception_node', 'open_vocab_perception_node'),
            _node('semantic_memory_node',       'semantic_memory_node'),
            _node('instruction_parser_node',    'instruction_parser_node'),
        ]),

        # t = 6s: navigator (depends on /map + upstream)
        TimerAction(period=6.0, actions=[
            LogInfo(msg='[AuditNav] Starting topo_nav_node...'),
            _node('topo_nav_node', 'topo_nav_node'),
        ]),
    ])

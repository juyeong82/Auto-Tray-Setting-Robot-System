#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

def generate_launch_description():
    
    # 1. 로봇 메인 컨트롤러 노드
    robot_controller = Node(
        package='ff_robot',
        executable='robot_controller_node_gem',
        name='robot_controller',
        output='screen',
        # 파라미터가 필요하다면 여기에 추가
        # parameters=[{'use_sim_time': True}]
    )

    # 2. 키오스크 노드 (새 터미널에서 실행)
    # Ubuntu Desktop 환경이라면 gnome-terminal을 사용하여 별도 창을 띄움
    # 키오스크는 CLI 입력이 필요하기 때문입니다.
    kiosk_terminal = ExecuteProcess(
        cmd=['gnome-terminal', '--', 'ros2', 'run', 'ff_robot', 'kiosk_gem'],
        output='screen'
    )

    return LaunchDescription([
        LogInfo(msg="🍔 햄버거 로봇 시스템을 시작합니다..."),
        
        robot_controller,
        
        # 컨트롤러가 실행된 후 키오스크 창을 띄우고 싶다면 
        # (바로 실행해도 상관없음)
        kiosk_terminal
    ])
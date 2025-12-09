import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'ff_robot' 

    return LaunchDescription([
        # 1. YOLO Vision Node
        Node(
            package=package_name,
            executable='yolo_vision_node',  # setup.py의 왼쪽 이름과 일치해야 함!
            name='yolo_vision_node',
            output='screen'
        ),

        # 2. Item Placement Controller (Module 1)
        Node(
            package=package_name,
            executable='module1_item',      # setup.py의 'module1_item' 사용
            name='item_placement_controller',
            output='screen'
        ),

        # 3. Order Manager (Module 2)
        Node(
            package=package_name,
            executable='module2_manager_555', # setup.py의 'module2_manager_555' 사용
            name='order_orchestrator',
            output='screen'
        ),
    ])
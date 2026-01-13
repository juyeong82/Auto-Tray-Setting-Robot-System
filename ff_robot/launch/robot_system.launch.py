import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'ff_robot' 

    return LaunchDescription([
        # 1. YOLO Vision Node
        Node(
            package=package_name,
            executable='yolo_vision_node',
            name='yolo_vision_node',
            output='screen'
        ),

        # 2. Item Placement Controller (Module 1)
        Node(
            package=package_name,
            executable='module1_item_jy', 
            name='item_placement_controller',
            output='screen'
        ),

        # 3. Order Manager (Module 2)
        Node(
            package=package_name,
            executable='module2_manager_jy',
            name='order_manager',
            output='screen'
        ),
        
        # 4. YOLO Safety monitor
        Node(
            package=package_name,
            executable='safety_monitor',
            name='safety_monitor',
            output='screen'
        ),
        
        # 5. kiosk ui
        Node(
            package=package_name,
            executable='kiosk_final',
            name='kiosk_final',
            output='screen'
        ),
    ])
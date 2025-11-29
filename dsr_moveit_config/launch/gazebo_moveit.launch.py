#!/usr/bin/env python3

import os
import yaml
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. 패키지 경로
    moveit_config_pkg = get_package_share_directory('dsr_moveit_config')
    robot_description_pkg = get_package_share_directory('robot_description')
    
    # 2. URDF
    urdf_file_path = os.path.join(robot_description_pkg, 'urdf', 'dsr_combined_fixed3.urdf')
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ', urdf_file_path
    ])
    robot_description = {'robot_description': robot_description_content}

    # 3. SRDF
    srdf_file = os.path.join(moveit_config_pkg, 'config', 'dsr_m0609_complete.srdf')
    
    # 4. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )
    
    # 5. Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={'verbose': 'false'}.items()
    )
    
    # 6. Spawn Robot
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'dsr_robot',
            '-x', '0.0', '-y', '0.0', '-z', '0.0'
        ],
        output='screen'
    )

    # 7. Controllers
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    load_manipulator_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['manipulator_controller'],
        output='screen'
    )

    load_gripper_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller'],
        output='screen'
    )
    
    # 8. ✅ MoveIt 설정 파일 로드
    kinematics_file = os.path.join(moveit_config_pkg, 'config', 'kinematics.yaml')
    joint_limits_file = os.path.join(moveit_config_pkg, 'config', 'joint_limits.yaml')
    moveit_controllers_file = os.path.join(moveit_config_pkg, 'config', 'moveit_controllers.yaml')
    
    with open(kinematics_file, 'r') as f:
        kinematics_yaml = yaml.safe_load(f)
    with open(joint_limits_file, 'r') as f:
        joint_limits_yaml = yaml.safe_load(f)
    with open(moveit_controllers_file, 'r') as f:
        moveit_controllers_yaml = yaml.safe_load(f)
    
    # 9. ⭐ MoveIt 설정 (Trajectory Execution 활성화)
    moveit_config = {
        'robot_description': robot_description_content,
        'robot_description_semantic': open(srdf_file).read(),
        'robot_description_kinematics': kinematics_yaml,
        'robot_description_planning': joint_limits_yaml,
        
        # ⭐ Trajectory Execution 설정
        'moveit_manage_controllers': True,
        'trajectory_execution': {
            'allowed_execution_duration_scaling': 1.2,
            'allowed_goal_duration_margin': 0.5,
            'allowed_start_tolerance': 0.01,
        },
        
        # Planning pipeline
        'planning_pipelines': ['ompl'],
        'ompl': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/ResolveConstraintFrames default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
            'start_state_max_bounds_error': 0.1,
        },
    }
    
    # MoveIt controller manager 설정
    moveit_config.update(moveit_controllers_yaml)
    
    # 10. Move Group
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config]
    )
    
    # 11. RViz
    rviz_config = os.path.join(moveit_config_pkg, 'config', 'moveit.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config],
        parameters=[moveit_config]
    )
    
    return LaunchDescription([
        robot_state_publisher,
        gazebo,
        spawn_entity,
        
        # 컨트롤러는 spawn 후 실행
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[
                    load_joint_state_broadcaster,
                    load_manipulator_controller,
                    load_gripper_controller,
                ]
            )
        ),
        
        # ⭐ MoveGroup은 컨트롤러 로드 후 약간 지연
        TimerAction(
            period=5.0,  # 5초 후 실행
            actions=[move_group_node]
        ),
        
        rviz_node,
    ])
#!/usr/bin/env python3

import os
import yaml
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, 
    RegisterEventHandler, 
    TimerAction,
    DeclareLaunchArgument
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    PathJoinSubstitution, 
    Command, 
    FindExecutable, 
    LaunchConfiguration
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 시뮬레이션 시간 사용
    use_sim_time = True
    
    # 패키지 경로
    moveit_config_pkg = get_package_share_directory('dsr_moveit_config')
    robot_description_pkg = get_package_share_directory('robot_description')
    
    # World 파일 경로
    world_file_arg = DeclareLaunchArgument(
        'world',
        default_value='/home/juyeong/ros2_ws/src/dsr_moveit_config/worlds/hamburger_station.world',
        description='Path to the Gazebo world file'
    )
    
    # URDF
    urdf_file_path = os.path.join(robot_description_pkg, 'urdf', 'dsr_combined_fixed3.urdf')
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ', urdf_file_path
    ])
    robot_description = {'robot_description': robot_description_content}

    # SRDF
    srdf_file = os.path.join(moveit_config_pkg, 'config', 'dsr_m0609_complete.srdf')
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}]
    )
    
    # Gazebo with World
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'verbose': 'false'
        }.items()
    )
    
    # Spawn Robot
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'dsr_robot',
            '-x', '0.0',
            '-y', '0.0', 
            '-z', '0.82',
        ],
        output='screen'
    )

    # Controllers
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
    
    # MoveIt 설정 파일 로드
    kinematics_file = os.path.join(moveit_config_pkg, 'config', 'kinematics.yaml')
    joint_limits_file = os.path.join(moveit_config_pkg, 'config', 'joint_limits.yaml')
    moveit_controllers_file = os.path.join(moveit_config_pkg, 'config', 'moveit_controllers.yaml')
    
    with open(kinematics_file, 'r') as f:
        kinematics_yaml = yaml.safe_load(f)
    with open(joint_limits_file, 'r') as f:
        joint_limits_yaml = yaml.safe_load(f)
    with open(moveit_controllers_file, 'r') as f:
        moveit_controllers_yaml = yaml.safe_load(f)
    
    # MoveIt 설정
    moveit_config = {
        'robot_description': robot_description_content,
        'robot_description_semantic': open(srdf_file).read(),
        'robot_description_kinematics': kinematics_yaml,
        'robot_description_planning': joint_limits_yaml,
        'use_sim_time': use_sim_time,
        
        # Trajectory Execution 설정
        'moveit_manage_controllers': True,
        'trajectory_execution': {
            'allowed_execution_duration_scaling': 1.2,
            'allowed_goal_duration_margin': 0.5,
            'allowed_start_tolerance': 0.01,
        },
        
        # Planning Scene Monitor 설정
        'planning_scene_monitor_options': {
            'name': 'planning_scene_monitor',
            'robot_description': 'robot_description',
            'joint_state_topic': '/joint_states',
            'attached_collision_object_topic': '/attached_collision_object',
            'publish_planning_scene_topic': '/planning_scene',
            'monitored_planning_scene_topic': '/monitored_planning_scene',
            'wait_for_initial_state_timeout': 10.0,
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
    
    # Move Group
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config],
        remappings=[('/joint_states', '/joint_states')]
    )
    
    # RViz
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
        world_file_arg,
        robot_state_publisher,
        gazebo,
        spawn_entity,
        
        # Step 1: Spawn 후 컨트롤러 실행
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
        
        # Step 2: 컨트롤러 로드 후 5초 대기 후 MoveGroup 실행
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_gripper_controller,
                on_exit=[
                    TimerAction(
                        period=5.0,
                        actions=[move_group_node]
                    )
                ]
            )
        ),
        
        # RViz는 바로 실행
        rviz_node,
    ])
#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. 패키지 경로 설정
    moveit_config_pkg = get_package_share_directory('dsr_moveit_config')
    robot_description_pkg = get_package_share_directory('robot_description')
    
    # 2. URDF 파일 경로 (확장자는 .urdf지만 내용 해석을 위해 xacro 필수)
    urdf_file_path = os.path.join(robot_description_pkg, 'urdf', 'dsr_combined_fixed3.urdf')
    
    # 3. Xacro 명령어로 URDF 생성 (성공했던 핵심 설정)
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ', urdf_file_path
    ])
    
    robot_description = {'robot_description': robot_description_content}

    # SRDF 파일 경로
    srdf_file = os.path.join(moveit_config_pkg, 'config', 'dsr_m0609_complete.srdf')
    
    # 4. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )
    
    # 5. Gazebo 실행
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={'verbose': 'true'}.items()
    )
    
    # 6. 로봇 스폰
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

    # 7. 컨트롤러 로드 (Spawner)
    # Gazebo 플러그인과 충돌하지 않도록 'spawner'를 사용합니다.
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
    
    # 8. MoveIt 설정 (OMPL 파이프라인)
    moveit_config = {
        'robot_description': robot_description_content,
        'robot_description_semantic': open(srdf_file).read(),
        'robot_description_kinematics': PathJoinSubstitution([
            moveit_config_pkg, 'config', 'kinematics.yaml'
        ]),
        'planning_pipelines': ['ompl'],
        'ompl': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/ResolveConstraintFrames default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
            'start_state_max_bounds_error': 0.1,
        },
    }
    
    # Move Group 노드 (경로 계획)
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            moveit_config,
            {
                'trajectory_execution.allowed_execution_duration_scaling': 1.2,
                'trajectory_execution.allowed_goal_duration_margin': 0.5,
                'trajectory_execution.allowed_start_tolerance': 0.01,
            }
        ]
    )
    
    # RViz 실행 (시각화)
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
        # 로봇 스폰 후 컨트롤러 실행
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
        # 컨트롤러 로드와 동시에 MoveIt 실행
        move_group_node,
        rviz_node,
    ])
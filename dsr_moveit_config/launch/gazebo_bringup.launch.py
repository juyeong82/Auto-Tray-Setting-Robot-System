import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, Command, FindExecutable
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Gazebo 실행
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gzserver.launch.py'
            ])
        ]),
        launch_arguments={'world': '/opt/ros/humble/share/gazebo_ros/worlds/empty.world'}.items()
    )

    gazebo_client_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gzclient.launch.py'
            ])
        ])
    )

    # URDF 생성 (xacro 실행)
    # 확장자 수정: .urdf.xacro -> .urdf (실제 파일명과 일치시킴)
    # 주의: .urdf 파일이지만 $(find) 구문 해석을 위해 xacro 명령어로 실행 필수
    robot_description_content = Command([
        FindExecutable(name='xacro'),
        ' ',
        PathJoinSubstitution([
            FindPackageShare('robot_description'),
            'urdf',
            'dsr_combined_fixed3.urdf' 
        ])
    ])
    robot_description = {'robot_description': robot_description_content}

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )

    # 컨트롤러 매니저 노드 제거 (Gazebo 플러그인과 충돌 방지)
    # control_node = ... 

    # Gazebo에 로봇 스폰
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'dsr_robot'],
        output='screen'
    )

    # 컨트롤러 로드 (Spawner)
    # 로봇이 스폰된 후 실행되도록 설정
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

    return LaunchDescription([
        gazebo_launch,
        gazebo_client_launch,
        robot_state_publisher,
        spawn_entity,
        # 로봇 스폰 완료 후 컨트롤러 실행 (순서 제어)
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[
                    load_joint_state_broadcaster,
                    load_manipulator_controller,
                    load_gripper_controller,
                ]
            )
        )
    ])
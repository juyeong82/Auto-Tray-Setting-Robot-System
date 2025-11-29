import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    
    robot_desc_pkg = get_package_share_directory('robot_description')
    
    # Arguments
    mode_arg = DeclareLaunchArgument('mode', default_value='virtual')
    gripper_arg = DeclareLaunchArgument('gripper', default_value='true')
    camera_arg = DeclareLaunchArgument('camera', default_value='true')
    
    # URDF with gripper and camera
    xacro_file = os.path.join(
        robot_desc_pkg, 
        'urdf', 
        'dsr_m0609_gripper_camera.urdf.xacro'
    )
    
    robot_description_content = Command([
        'xacro ', xacro_file,
        ' gripper:=', LaunchConfiguration('gripper'),
        ' camera:=', LaunchConfiguration('camera')
    ])
    
    robot_description_param = {
        'robot_description': ParameterValue(robot_description_content, value_type=str)
    }
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[robot_description_param],
        output='screen'
    )
    
    # Joint State Publisher GUI
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )
    
    # RViz with custom config
    rviz_config = os.path.join(robot_desc_pkg, 'config', 'robot_view.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen'
    )
    
    return LaunchDescription([
        mode_arg,
        gripper_arg,
        camera_arg,
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz
    ])
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ff_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='yangsw197@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'robot_controller_node = ff_robot.robot_controller_node:main',
            'robot_controller_tray = ff_robot.robot_controller_tray:main',
            'kiosk_gem = ff_robot.kiosk_gem:main',
            'kiosk_with_voice = ff_robot.kiosk_with_voice:main',
            'dummy_vision_node = ff_robot.dummy_vision_node:main',
            'yolo_vision_node = ff_robot.yolo_vision_node:main',
            'yolo_vision_node_gazebo = ff_robot.yolo_vision_node_gazebo:main',
            'yolo_vision_node_tray = ff_robot.yolo_vision_node_tray:main',
            'kvu = ff_robot.kvu:main',
            'kiosk = ff_robot.kiosk:main',
            'kiosk_v15 = ff_robot.kiosk_v15:main',
            'module1_item = ff_robot.module1_item:main',
            'module2_manager = ff_robot.module2_manager:main',
            'module1_item_virtual = ff_robot.module1_item_virtual:main',
            'module2_manager_virtual = ff_robot.module2_manager_virtual:main',
        ],
    },
)

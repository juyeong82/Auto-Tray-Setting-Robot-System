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
            'kiosk_gem = ff_robot.kiosk_gem:main',
            'kiosk_with_voice = ff_robot.kiosk_with_voice:main',
            'dummy_vision_node = ff_robot.dummy_vision_node:main',
            'yolo_vision_node = ff_robot.yolo_vision_node:main',
            'kvu = ff_robot.kvu:main',
            'kiosk = ff_robot.kiosk:main',
        ],
    },
)

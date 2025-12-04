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
            'yolo_vision_node = ff_robot.yolo_vision_node:main',
            'kiosk_final = ff_robot.kiosk_final:main',
            'module1_item_jy = ff_robot.module1_item_jy:main',
            'module2_manager_jy = ff_robot.module2_manager_jy:main',
            'safety_monitor = ff_robot.safety_monitor:main',
        ],
    },
)

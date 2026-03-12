from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'auditnav'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Required by ROS 2 / ament
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # Install config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Anonymous Author',
    maintainer_email='anonymous@review.iros',
    description='AuditNav: VLN with Semantic Topological Memory (IROS)',
    license='MIT',
    entry_points={
        'console_scripts': [
            # Each node is registered as a ROS 2 executable:
            #   ros2 run auditnav <entry_point>
            'occupancy_map_node         = auditnav.nodes.occupancy_map_node:main',
            'open_vocab_perception_node = auditnav.nodes.open_vocab_perception_node:main',
            'semantic_memory_node       = auditnav.nodes.semantic_memory_node:main',
            'instruction_parser_node    = auditnav.nodes.instruction_parser_node:main',
            'topo_nav_node              = auditnav.nodes.topo_nav_node:main',
        ],
    },
)

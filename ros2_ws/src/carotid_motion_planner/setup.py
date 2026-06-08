from setuptools import find_packages, setup
import os

package_name = 'carotid_motion_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib', package_name),
            ['scripts/motion_planner_node', 'scripts/joint_state_sim']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gokul Soman',
    maintainer_email='gokul@example.com',
    description='Motion planner node for carotid robotic arm',
    license='MIT',
    tests_require=['pytest'],
)

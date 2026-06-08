from setuptools import find_packages, setup
import os

package_name = 'carotid_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib', package_name),
            ['scripts/perception_node', 'scripts/image_publisher']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gokul Soman',
    maintainer_email='gokul@example.com',
    description='Carotid artery perception node for ROS2',
    license='MIT',
    tests_require=['pytest'],
)

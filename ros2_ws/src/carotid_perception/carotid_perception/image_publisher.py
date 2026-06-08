#!/usr/bin/env python3
"""
carotid_perception.image_publisher
=====================================
Test/demo node: streams carotid ultrasound images from the dataset directory
to /ultrasound/image_raw at a configurable rate.

Usage:
  ros2 run carotid_perception image_publisher \
    --ros-args -p image_dir:=/path/to/US\ images -p rate_hz:=2.0
"""

import glob
import os
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header

try:
    from cv_bridge import CvBridge
    HAS_CV_BRIDGE = True
except ImportError:
    HAS_CV_BRIDGE = False

# Default dataset path relative to the carotid-segmentation project
DEFAULT_IMAGE_DIR = (
    '/home/admin1/Music/prep/cyint/carotid-segmentation-main/src/data'
    '/Common Carotid Artery Ultrasound Images/US images'
)


class ImagePublisherNode(Node):
    """Publishes carotid ultrasound images to /ultrasound/image_raw."""

    def __init__(self):
        super().__init__('ultrasound_image_publisher')

        self.declare_parameter('image_dir', DEFAULT_IMAGE_DIR)
        self.declare_parameter('rate_hz', 1.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('frame_id', 'ultrasound_frame')

        image_dir = self.get_parameter('image_dir').value
        rate_hz = self.get_parameter('rate_hz').value
        self.loop = self.get_parameter('loop').value
        self.frame_id = self.get_parameter('frame_id').value

        self.bridge = CvBridge() if HAS_CV_BRIDGE else None
        self.pub = self.create_publisher(Image, '/ultrasound/image_raw', 10)

        # Collect images
        self.images = sorted(
            glob.glob(os.path.join(image_dir, '*.png'))
            + glob.glob(os.path.join(image_dir, '*.jpg'))
        )
        if not self.images:
            self.get_logger().error(f'No images found in: {image_dir}')
            return

        self.idx = 0
        self.get_logger().info(
            f'Found {len(self.images)} images in {image_dir}. '
            f'Publishing at {rate_hz} Hz.'
        )

        period = 1.0 / rate_hz
        self.timer = self.create_timer(period, self._publish_next)

    def _publish_next(self) -> None:
        if self.idx >= len(self.images):
            if self.loop:
                self.idx = 0
            else:
                self.get_logger().info('All images published. Shutting down.')
                self.timer.cancel()
                return

        path = self.images[self.idx]
        self.idx += 1

        bgr = cv2.imread(path)
        if bgr is None:
            self.get_logger().warn(f'Could not read: {path}')
            return

        stamp = self.get_clock().now().to_msg()

        if self.bridge is not None:
            msg = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
        else:
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            msg.height, msg.width = bgr.shape[:2]
            msg.encoding = 'bgr8'
            msg.step = bgr.shape[1] * 3
            msg.data = bgr.tobytes()

        self.pub.publish(msg)
        self.get_logger().info(
            f'Published image {self.idx}/{len(self.images)}: '
            f'{Path(path).name}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

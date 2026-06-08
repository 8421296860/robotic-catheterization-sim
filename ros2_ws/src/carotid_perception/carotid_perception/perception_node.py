#!/usr/bin/env python3
"""
carotid_perception.perception_node
====================================
ROS2 node that subscribes to ultrasound images, runs the carotid artery
segmentation model (nnUNetv2 or UNet fallback), and publishes:

  /carotid/mask          sensor_msgs/Image       – binary segmentation mask
  /carotid/centroid      geometry_msgs/PointStamped – vessel centroid (px→m)
  /carotid/target_pose   geometry_msgs/PoseStamped  – 3-D target for the arm
  /carotid/markers       visualization_msgs/MarkerArray – RViz overlay

Parameters (ROS2):
  image_scale_m   (float, default 0.001) – metres per pixel (calibration)
  model_backend   (str,   default 'nnunet') – 'nnunet' or 'unet'
  model_path      (str,   default '') – path to nnUNet results dir or .pth
  confidence_thr  (float, default 0.5) – mask binarisation threshold
  debug_overlay   (bool,  default True) – publish coloured overlay on /carotid/overlay

Topics consumed:
  /ultrasound/image_raw  sensor_msgs/Image
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseStamped, Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

try:
    from cv_bridge import CvBridge
    HAS_CV_BRIDGE = True
except ImportError:
    HAS_CV_BRIDGE = False


# ──────────────────────────────────────────────────────────────────────────────
# Backend wrappers
# ──────────────────────────────────────────────────────────────────────────────

class UNetBackend:
    """Thin wrapper around the existing carotid-segmentation UNet model."""

    def __init__(self, model_name: str = 'unet_1', src_dir: Optional[str] = None):
        if src_dir is None:
            # default: one level above this package's install location or
            # fallback to the source tree alongside the ros2_ws
            candidates = [
                Path(__file__).resolve().parents[6] / 'carotid-segmentation-main' / 'src',
                Path('/home/admin1/Music/prep/cyint/carotid-segmentation-main/src'),
            ]
            src_dir = next((str(p) for p in candidates if p.is_dir()), None)
            if src_dir is None:
                raise RuntimeError(
                    'Cannot locate carotid-segmentation src directory. '
                    'Set the model_path parameter explicitly.'
                )

        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        os.chdir(src_dir)  # model.py uses relative paths for config/models

        from model import carotidSegmentation  # noqa: PLC0415
        self.model = carotidSegmentation(model=model_name)
        self.threshold = 0.5

    def predict_mask(self, bgr_image: np.ndarray) -> np.ndarray:
        """Return float32 probability map [0,1] same HxW as input."""
        import torch
        from torchvision.transforms.functional import to_tensor  # noqa: PLC0415

        # Convert BGR→RGB, resize to 512 (model expectation), float
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (512, 512))
        tensor = to_tensor(rgb_resized).float().unsqueeze(0)  # (1,3,512,512)

        with torch.no_grad():
            pred = self.model.net(tensor)  # (1,1,H,W)

        prob = pred[0, 0].cpu().numpy()  # (H, W) float32
        # resize back to original
        prob = cv2.resize(prob, (bgr_image.shape[1], bgr_image.shape[0]))
        return prob


class NNUNetBackend:
    """
    Wrapper around nnUNetv2 predictor for a trained Dataset501_CarotidArtery.
    Falls back to UNetBackend if nnUNet is not installed or results not found.
    """

    DATASET_ID = 501
    DATASET_NAME = 'Dataset501_CarotidArtery'

    def __init__(self, results_dir: Optional[str] = None):
        self._fallback: Optional[UNetBackend] = None
        self._predictor = None

        if results_dir is None:
            candidates = [
                Path('/home/admin1/Music/prep/cyint/carotid-segmentation-main/nnunet_data/nnUNet_results'),
                Path.home() / 'nnunet_data' / 'nnUNet_results',
            ]
            results_dir = next((str(p) for p in candidates if p.is_dir()), None)

        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: PLC0415
            import torch

            model_folder = (
                Path(results_dir) / self.DATASET_NAME /
                'nnUNetTrainer__nnUNetPlans__2d'
            ) if results_dir else None

            if model_folder and model_folder.is_dir():
                self._predictor = nnUNetPredictor(
                    tile_step_size=0.5,
                    use_gaussian=True,
                    use_mirroring=True,
                    perform_everything_on_device=True,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    verbose=False,
                )
                self._predictor.initialize_from_trained_model_folder(
                    str(model_folder),
                    use_folds=(0,),
                    checkpoint_name='checkpoint_best.pth',
                )
            else:
                raise FileNotFoundError(f'nnUNet model folder not found: {model_folder}')

        except Exception as exc:  # noqa: BLE001
            # Graceful fallback — use the existing UNet weights
            print(f'[NNUNetBackend] nnUNet not available ({exc}), '
                  f'falling back to UNet backend.')
            self._fallback = UNetBackend()

    def predict_mask(self, bgr_image: np.ndarray) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.predict_mask(bgr_image)

        import torch

        # The specific nnUNet model expects a 3-channel input in shape (C, Z, Y, X)
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32).transpose(2, 0, 1)  # (3, H, W)
        arr = np.expand_dims(arr, axis=1)  # (3, 1, H, W)

        pred_dict = self._predictor.predict_single_npy_array(
            input_image=arr,
            image_properties={'spacing': [999, 1, 1]},
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=True,
        )
        # pred_dict is (seg, probs); probs shape is (n_classes, Z, H, W) -> (2, 1, H, W)
        probs = pred_dict[1]
        prob = probs[1, 0, :, :].astype(np.float32)  # artery channel, Z=0
        # The returned prob matches the original input dimensions, no resize needed
        return prob


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def largest_blob_centroid(mask_binary: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Return (cx, cy, radius_px) of the largest connected component in mask_binary.
    Returns None if no foreground pixels found.
    """
    contours, _ = cv2.findContours(
        mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    area = cv2.contourArea(largest)
    radius = np.sqrt(area / np.pi)
    return float(cx), float(cy), float(radius)


def pixel_to_world(
    cx_px: float,
    cy_px: float,
    img_w: int,
    img_h: int,
    scale_m_per_px: float = 0.001,
) -> Tuple[float, float]:
    """
    Simple linear mapping from image pixels to world XY metres.
    Origin is at image centre.  Z is set by the planner.
    """
    x_m = (cx_px - img_w / 2.0) * scale_m_per_px
    y_m = (cy_px - img_h / 2.0) * scale_m_per_px
    return x_m, y_m


# ──────────────────────────────────────────────────────────────────────────────
# ROS2 node
# ──────────────────────────────────────────────────────────────────────────────

class CarotidPerceptionNode(Node):
    """
    ROS2 node: ultrasound image → segmentation → centroid → target pose.
    """

    # Frame that ties perception to the robot world
    WORLD_FRAME = 'world'
    CAMERA_FRAME = 'ultrasound_frame'

    def __init__(self):
        super().__init__('carotid_perception_node')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('image_scale_m', 0.0001)
        self.declare_parameter('model_backend', 'unet')   # 'nnunet' or 'unet'
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_thr', 0.45)
        self.declare_parameter('debug_overlay', True)
        self.declare_parameter('pub_rate_hz', 1.0)

        self.scale_m = self.get_parameter('image_scale_m').value
        self.backend_name = self.get_parameter('model_backend').value
        self.model_path = self.get_parameter('model_path').value
        self.confidence_thr = self.get_parameter('confidence_thr').value
        self.debug_overlay = self.get_parameter('debug_overlay').value

        # ── CV bridge ────────────────────────────────────────────────────────
        if HAS_CV_BRIDGE:
            self.bridge = CvBridge()
        else:
            self.bridge = None
            self.get_logger().warn('cv_bridge not available; using manual encoding')

        # ── Load model ───────────────────────────────────────────────────────
        self.get_logger().info(f'Loading segmentation backend: {self.backend_name}')
        try:
            if self.backend_name == 'nnunet':
                self.backend = NNUNetBackend(
                    results_dir=self.model_path if self.model_path else None
                )
            else:
                model_name = (
                    Path(self.model_path).stem if self.model_path else 'unet_1'
                )
                self.backend = UNetBackend(model_name=model_name)
            self.get_logger().info('Model loaded successfully ✓')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Model load failed: {exc}')
            self.get_logger().warn('Node will publish zeros until model is available.')
            self.backend = None

        # ── Publishers ───────────────────────────────────────────────────────
        self.pub_mask = self.create_publisher(Image, '/carotid/mask', 10)
        self.pub_centroid = self.create_publisher(
            PointStamped, '/carotid/centroid', 10
        )
        self.pub_target = self.create_publisher(
            PoseStamped, '/carotid/target_pose', 10
        )
        self.pub_markers = self.create_publisher(
            MarkerArray, '/carotid/markers', 10
        )
        if self.debug_overlay:
            self.pub_overlay = self.create_publisher(
                Image, '/carotid/overlay', 10
            )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.sub_image = self.create_subscription(
            Image,
            '/ultrasound/image_raw',
            self.image_callback,
            10,
        )

        # ── State ────────────────────────────────────────────────────────────
        self._last_target: Optional[Tuple[float, float, float]] = None
        self._frame_count = 0

        self.get_logger().info(
            'CarotidPerceptionNode ready. Listening on /ultrasound/image_raw'
        )

    # ─────────────────────────────────────────────────────────────────────────

    def image_callback(self, msg: Image) -> None:
        t0 = time.time()
        self._frame_count += 1

        # Decode ROS Image → OpenCV BGR
        bgr = self._ros_to_cv(msg)
        if bgr is None:
            return

        h, w = bgr.shape[:2]
        stamp = msg.header.stamp
        frame_id = msg.header.frame_id or self.CAMERA_FRAME

        # ── Run inference ────────────────────────────────────────────────────
        if self.backend is not None:
            prob = self.backend.predict_mask(bgr)
        else:
            prob = np.zeros((h, w), dtype=np.float32)

        binary_mask = (prob > self.confidence_thr).astype(np.uint8)

        # ── Publish mask ─────────────────────────────────────────────────────
        self._publish_mask(binary_mask, stamp, frame_id)

        # ── Compute centroid ─────────────────────────────────────────────────
        centroid = largest_blob_centroid(binary_mask)

        if centroid is not None:
            cx_px, cy_px, radius_px = centroid
            x_m, y_m = pixel_to_world(cx_px, cy_px, w, h, self.scale_m)
            self._last_target = (x_m, y_m, 0.05)  # Z = 5 cm above probe plane
            self.get_logger().info(
                f'[{self._frame_count}] Vessel detected  '
                f'cx={cx_px:.1f}px cy={cy_px:.1f}px  '
                f'→ ({x_m*100:.1f}, {y_m*100:.1f}) cm'
            )
        else:
            self.get_logger().warn(
                f'[{self._frame_count}] No vessel detected in frame.'
            )
            x_m, y_m = 0.0, 0.0
            cx_px = cy_px = radius_px = 0.0

        # ── Publish centroid ─────────────────────────────────────────────────
        pt = PointStamped()
        pt.header = Header(stamp=stamp, frame_id=frame_id)
        pt.point = Point(x=x_m, y=y_m, z=0.0)
        self.pub_centroid.publish(pt)

        # ── Publish target pose ──────────────────────────────────────────────
        pose = PoseStamped()
        pose.header = Header(stamp=stamp, frame_id=self.WORLD_FRAME)
        pose.pose.position.x = x_m
        pose.pose.position.y = y_m
        pose.pose.position.z = self._last_target[2] if self._last_target else 0.05
        # Orientation: end-effector pointing straight down (z-axis into patient)
        pose.pose.orientation.x = 0.7071
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.7071
        self.pub_target.publish(pose)

        # ── RViz markers ─────────────────────────────────────────────────────
        self._publish_markers(x_m, y_m, radius_px * self.scale_m, stamp)

        # ── Debug overlay ────────────────────────────────────────────────────
        if self.debug_overlay:
            self._publish_overlay(bgr, prob, binary_mask, cx_px, cy_px, radius_px, stamp, frame_id)

        elapsed_ms = (time.time() - t0) * 1000
        self.get_logger().debug(f'Inference time: {elapsed_ms:.1f} ms')

    # ─────────────────────────────────────────────────────────────────────────

    def _ros_to_cv(self, msg: Image) -> Optional[np.ndarray]:
        """Convert sensor_msgs/Image → OpenCV BGR uint8."""
        if self.bridge is not None:
            try:
                return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f'cv_bridge conversion failed: {exc}')
                return None
        # Manual fallback
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding in ('bgr8',):
                return data.reshape(msg.height, msg.width, 3)
            elif msg.encoding in ('rgb8',):
                img = data.reshape(msg.height, msg.width, 3)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding in ('mono8', '8UC1'):
                gray = data.reshape(msg.height, msg.width)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().warn(f'Unsupported encoding: {msg.encoding}')
                return None
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Manual image decode failed: {exc}')
            return None

    def _cv_to_ros(self, bgr: np.ndarray, stamp, frame_id: str,
                   encoding: str = 'bgr8') -> Image:
        """Convert OpenCV BGR → sensor_msgs/Image."""
        if self.bridge is not None:
            img_msg = self.bridge.cv2_to_imgmsg(bgr, encoding=encoding)
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = frame_id
            return img_msg
        # Manual
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height, msg.width = bgr.shape[:2]
        msg.encoding = encoding
        msg.step = bgr.shape[1] * bgr.shape[2] if bgr.ndim == 3 else bgr.shape[1]
        msg.data = bgr.tobytes()
        return msg

    def _publish_mask(self, binary: np.ndarray, stamp, frame_id: str) -> None:
        vis = (binary * 255).astype(np.uint8)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        self.pub_mask.publish(self._cv_to_ros(vis_bgr, stamp, frame_id))

    def _publish_overlay(
        self, bgr, prob, binary_mask, cx, cy, radius, stamp, frame_id
    ):
        overlay = bgr.copy()
        # Green channel = probability heatmap
        heat = (prob * 255).clip(0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(overlay, 0.6, heat_color, 0.4, 0)
        # Draw detected vessel circle
        if radius > 2:
            cv2.circle(overlay, (int(cx), int(cy)), int(radius),
                       (0, 255, 0), 2)
            cv2.circle(overlay, (int(cx), int(cy)), 4,
                       (0, 0, 255), -1)
            cv2.putText(overlay, f'Carotid ({cx:.0f},{cy:.0f})',
                        (int(cx) + 5, int(cy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        self.pub_overlay.publish(
            self._cv_to_ros(overlay, stamp, frame_id)
        )

    def _publish_markers(self, x_m, y_m, radius_m, stamp) -> None:
        arr = MarkerArray()

        # Sphere at centroid
        sphere = Marker()
        sphere.header = Header(stamp=stamp, frame_id=self.WORLD_FRAME)
        sphere.ns = 'carotid_vessel'
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = x_m
        sphere.pose.position.y = y_m
        sphere.pose.position.z = 0.0
        sphere.pose.orientation.w = 1.0
        r = max(radius_m, 0.005)
        sphere.scale.x = r * 2
        sphere.scale.y = r * 2
        sphere.scale.z = 0.005
        sphere.color.r = 1.0
        sphere.color.g = 0.2
        sphere.color.b = 0.2
        sphere.color.a = 0.7
        sphere.lifetime.sec = 2
        arr.markers.append(sphere)

        # Arrow showing target approach direction
        arrow = Marker()
        arrow.header = Header(stamp=stamp, frame_id=self.WORLD_FRAME)
        arrow.ns = 'carotid_approach'
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.position.x = x_m
        arrow.pose.position.y = y_m
        arrow.pose.position.z = 0.15
        arrow.pose.orientation.x = 0.7071
        arrow.pose.orientation.w = 0.7071
        arrow.scale.x = 0.1   # length
        arrow.scale.y = 0.008  # shaft diameter
        arrow.scale.z = 0.012  # head diameter
        arrow.color.r = 0.0
        arrow.color.g = 0.8
        arrow.color.b = 1.0
        arrow.color.a = 0.9
        arrow.lifetime.sec = 2
        arr.markers.append(arrow)

        self.pub_markers.publish(arr)


# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = CarotidPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

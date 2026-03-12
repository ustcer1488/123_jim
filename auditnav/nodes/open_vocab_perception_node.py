#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config, get
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
import tf2_ros
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf2_geometry_msgs
from tf2_geometry_msgs import do_transform_point
import os
import message_filters
import math
import re
import time
import torch

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, TransformStamped
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLOWorld

# Base directory for data output (relative to current working directory)
BASE_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(BASE_DIR, exist_ok=True)


class SimpleKalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        cv2.setIdentity(self.kf.processNoiseCov, 1e-2)
        cv2.setIdentity(self.kf.measurementNoiseCov, 1e-3)
        cv2.setIdentity(self.kf.errorCovPost, 1.0)
        self.is_initialized = False
        self.last_timestamp = 0

    def reset(self):
        self.is_initialized = False

    def update(self, measurement_x, measurement_y, current_time):
        if not self.is_initialized:
            self.kf.statePost = np.array(
                [[measurement_x], [measurement_y], [0], [0]], np.float32)
            self.kf.statePre = self.kf.statePost
            self.last_timestamp = current_time
            self.is_initialized = True
            return float(measurement_x), float(measurement_y)
        dt = current_time - self.last_timestamp
        self.last_timestamp = current_time
        if dt > 0.5:
            self.reset()
            return self.update(measurement_x, measurement_y, current_time)
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt
        self.kf.predict()
        measurement = np.array(
            [[np.float32(measurement_x)], [np.float32(measurement_y)]])
        estimated = self.kf.correct(measurement)
        return float(estimated[0, 0]), float(estimated[1, 0])


class OpenVocabPerceptionNode(Node):
    def __init__(self):
        super().__init__('open_vocab_perception_node')

        # Load centralised config
        self._cfg = load_config()

        self.get_logger().info("Loading YOLO-World model (Large)...")
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Place yolov8l-world.pt in the same directory as this script.
        # Download from: https://github.com/ultralytics/assets/releases/
        model_path = os.path.join(script_dir, 'yolov8l-world.pt')

        self.prompt_map = {
            "cup": ["cup", "mug", "glass", "coffee cup"],
            "bottle": ["plastic bottle", "water bottle", "glass bottle"],
            "person": ["person", "human", "pedestrian"],
            "chair": ["chair", "seat", "sofa", "stool"],
            "mouse": ["computer mouse"],
            "keyboard": ["computer keyboard"]
        }

        if os.path.exists(model_path):
            try:
                self.model = YOLOWorld(model_path)
                self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                self.model.to(self.device)
                self.active_prompts = ["person"]
                self.model.set_classes(self.active_prompts)
                self.get_logger().info(f"Model loaded. Device={self.device}")
            except Exception as e:
                self.get_logger().error(f"Model load failed: {e}")
        else:
            self.get_logger().error(f"Model file not found: {model_path}")

        self.current_instruction = ""
        self._last_tf_error_log = 0.0
        self.camera_info = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transforms()

        self.bridge = CvBridge()
        self.pub_debug_img = self.create_publisher(Image, get(self._cfg, "topics", "debug_image", default="/audit_nav/debug_image"), 10)
        self.pub_goal = self.create_publisher(PointStamped, get(self._cfg, "topics", "current_goal", default="/audit_nav/current_goal"), 10)
        self.pub_conf = self.create_publisher(Float32, get(self._cfg, "topics", "confidence", default="/audit_nav/confidence"), 10)

        self.create_subscription(String, get(self._cfg, "topics", "instruction", default="/audit_nav/instruction"), self.cmd_callback, 10)

        video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CameraInfo, get(self._cfg, 'topics', 'camera_info', default='/camera_info'), self.info_callback, video_qos)

        rgb_sub = message_filters.Subscriber(
            self, Image, get(self._cfg, 'topics', 'rgb_image', default='/camera/rgb/image_raw'), qos_profile=video_qos)
        depth_sub = message_filters.Subscriber(
            self, Image, get(self._cfg, 'topics', 'depth_image', default='/camera/depth/image_raw'), qos_profile=video_qos)
        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.5)
        ts.registerCallback(self.callback)

        self.kf = SimpleKalmanFilter()
        self.conf_ema = 0.0
        self.conf_ema_alpha = 0.20
        self.prev_bbox = None
        self.prev_bbox_time = 0.0
        self.iou_track_threshold = get(self._cfg, "perception", "iou_track_threshold", default=0.20)
        self.depth_patch_size = 11
        self.prev_depth = 0.0
        self.prev_depth_time = 0.0
        self.max_depth_jump = 0.80
        self.last_published_dist = 0.0
        self.last_detection_time = 0.0
        self.reset_timeout = 0.5
        self.log_counter = 0

        # Video writer (lazy init on first frame)
        self.video_writer = None
        self.video_path = None

        self.get_logger().info("VL Spatial Node ready.")

    def publish_static_transforms(self):
        current_time = self.get_clock().now().to_msg()
        ts_cam = TransformStamped()
        ts_cam.header.stamp = current_time
        ts_cam.header.frame_id = 'base_link'
        ts_cam.child_frame_id = 'Camera_01'
        ts_cam.transform.translation.x = 0.40
        ts_cam.transform.translation.y = 0.0
        ts_cam.transform.translation.z = 0.25
        ts_cam.transform.rotation.x = -0.5
        ts_cam.transform.rotation.y = 0.5
        ts_cam.transform.rotation.z = -0.5
        ts_cam.transform.rotation.w = 0.5
        self.static_broadcaster.sendTransform([ts_cam])

    def info_callback(self, msg):
        self.camera_info = msg

    def cmd_callback(self, msg):
        raw_cmd = msg.data.lower().strip()
        if not raw_cmd:
            return
        m = re.search(r"\[(.*?)\]", raw_cmd)
        target_key = (m.group(1).strip() if m else raw_cmd)
        self.current_instruction = target_key

        if hasattr(self, "model"):
            try:
                prompts = self.prompt_map.get(target_key, [target_key])
                self.model.set_classes(prompts)
                self.active_prompts = prompts
                self.get_logger().info(f"YOLO target locked: {prompts}")
            except Exception as e:
                self.get_logger().error(f"set_classes failed: {e}")

        self.kf.reset()
        self.prev_bbox = None
        self.last_published_dist = 0.0
        self.get_logger().info(f"Instruction received: find [{self.current_instruction}]")

    def get_roi_depth(self, cv_depth, x1, y1, x2, y2):
        h_img, w_img = cv_depth.shape[:2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w_box, h_box = x2 - x1, y2 - y1
        roi_w, roi_h = int(w_box * 0.3), int(h_box * 0.3)
        roi_x1 = max(0, cx - roi_w // 2)
        roi_x2 = min(w_img, cx + roi_w // 2)
        roi_y1 = max(0, cy - roi_h // 2)
        roi_y2 = min(h_img, cy + roi_h // 2)
        roi = cv_depth[roi_y1:roi_y2, roi_x1:roi_x2]
        valid_depths = roi[(roi > 0.1) & (roi < 20.0) & np.isfinite(roi)]
        return float(np.median(valid_depths)) if valid_depths.size > 0 else 0.0

    def get_patch_median_depth(self, cv_depth, u, v, patch_size=None):
        if patch_size is None:
            patch_size = self.depth_patch_size
        if cv_depth is None:
            return 0.0
        h_img, w_img = cv_depth.shape[:2]
        ps = int(patch_size)
        if ps < 3: ps = 3
        if ps % 2 == 0: ps += 1
        r = ps // 2
        x1, x2 = max(0, int(u) - r), min(w_img, int(u) + r + 1)
        y1, y2 = max(0, int(v) - r), min(h_img, int(v) + r + 1)
        roi = cv_depth[y1:y2, x1:x2]
        valid = roi[(roi > 0.1) & (roi < 20.0) & np.isfinite(roi)]
        return float(np.median(valid)) if valid.size > 0 else 0.0

    @staticmethod
    def _iou_xyxy(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = float(area_a + area_b - inter)
        return float(inter) / denom if denom > 0 else 0.0

    def callback(self, rgb_msg, depth_msg):
        if not hasattr(self, 'model') or self.camera_info is None:
            return

        current_time_sec = self.get_clock().now().nanoseconds / 1e9
        if (current_time_sec - self.last_detection_time) > self.reset_timeout:
            self.kf.reset()
            self.last_published_dist = 0.0

        try:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
                cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
                cv_depth = np.nan_to_num(cv_depth, posinf=0.0, neginf=0.0, nan=0.0)
            except CvBridgeError:
                return

            results = self.model.predict(cv_image, conf=0.05, verbose=False)

            debug_img = cv_image.copy()
            cv2.putText(debug_img, f"Target: {self.current_instruction}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            candidates = []
            best_conf = 0.0

            if hasattr(results[0], 'boxes'):
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id < len(self.active_prompts):
                        label = self.active_prompts[cls_id]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text_y = max(20, y1 - 10)
                        label_str = f"{label} {conf:.2f}"
                        cv2.putText(debug_img, label_str, (x1, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                        cv2.putText(debug_img, label_str, (x1, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        candidates.append({"bbox": (x1, y1, x2, y2), "conf": conf, "label": label})
                        if conf > best_conf:
                            best_conf = conf

            self.pub_debug_img.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

            # Lazy-init video writer
            if self.video_writer is None:
                h, w = debug_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_filename = f"yolo_debug_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                self.video_path = os.path.join(BASE_DIR, video_filename)
                self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 10.0, (w, h))
                self.get_logger().info(f"YOLO video recording started: {self.video_path}")
            if self.video_writer.isOpened():
                self.video_writer.write(debug_img)

            if not self.current_instruction:
                return
            if not candidates:
                self.conf_ema *= 0.90
                self.pub_conf.publish(Float32(data=float(self.conf_ema)))
                return

            K = self.camera_info.k
            fx, cx, fy, cy = K[0], K[2], K[4], K[5]

            if self.conf_ema <= 0.0:
                self.conf_ema = best_conf
            else:
                self.conf_ema = (1.0 - self.conf_ema_alpha) * self.conf_ema + self.conf_ema_alpha * best_conf

            chosen = None
            if self.prev_bbox is not None and (current_time_sec - self.prev_bbox_time) < 1.0:
                best_iou, best_i = 0.0, None
                for i, c in enumerate(candidates):
                    iou = self._iou_xyxy(self.prev_bbox, c["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_i = i
                chosen = (candidates[best_i] if best_i is not None and best_iou >= self.iou_track_threshold
                          else max(candidates, key=lambda d: d["conf"]))
            else:
                chosen = max(candidates, key=lambda d: d["conf"])

            if chosen:
                x1, y1, x2, y2 = chosen["bbox"]
                u, v = (x1 + x2) // 2, (y1 + y2) // 2

                depth_val = self.get_patch_median_depth(cv_depth, u, v)
                if depth_val < 0.1:
                    depth_val = self.get_roi_depth(cv_depth, x1, y1, x2, y2)

                if depth_val > 0.1:
                    if (self.prev_depth > 0.1 and
                            (current_time_sec - self.prev_depth_time) < 0.6 and
                            abs(depth_val - self.prev_depth) > float(self.max_depth_jump)):
                        depth_val = self.prev_depth
                    self.prev_depth = float(depth_val)
                    self.prev_depth_time = float(current_time_sec)

                    opt_z = float(depth_val)
                    opt_x = float((u - cx) * opt_z / fx)
                    opt_y = float((v - cy) * opt_z / fy)

                    point_source = PointStamped()
                    point_source.header = rgb_msg.header
                    point_source.point.x = opt_x
                    point_source.point.y = opt_y
                    point_source.point.z = opt_z

                    try:
                        target_frame = "base_link"
                        if self.tf_buffer.can_transform(
                                target_frame, point_source.header.frame_id,
                                rgb_msg.header.stamp,
                                rclpy.duration.Duration(seconds=0.5)):
                            transform = self.tf_buffer.lookup_transform(
                                target_frame, point_source.header.frame_id,
                                rgb_msg.header.stamp)
                            point_target = do_transform_point(point_source, transform)
                            kf_x, kf_y = self.kf.update(
                                point_target.point.x, point_target.point.y, current_time_sec)
                            point_target.point.x = kf_x
                            point_target.point.y = kf_y
                            point_target.point.z = 0.0
                            point_target.header.frame_id = target_frame
                            point_target.header.stamp = rgb_msg.header.stamp
                            self.pub_goal.publish(point_target)
                            self.last_detection_time = current_time_sec
                            self.prev_bbox = (x1, y1, x2, y2)
                            self.prev_bbox_time = float(current_time_sec)

                            self.log_counter += 1
                            if self.log_counter % 10 == 0:
                                self.get_logger().info(
                                    f"Tracking [{chosen['label']}] "
                                    f"conf={chosen['conf']:.2f} dist={depth_val:.2f}m")
                    except Exception as e:
                        if (current_time_sec - self._last_tf_error_log) > 1.0:
                            self._last_tf_error_log = current_time_sec
                            self.get_logger().warn(f"TF Error: {e}")

            self.pub_conf.publish(Float32(data=float(self.conf_ema)))

        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

    def destroy_node(self):
        if self.video_writer is not None and self.video_writer.isOpened():
            self.video_writer.release()
            self.get_logger().info(f"YOLO video saved: {self.video_path}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OpenVocabPerceptionNode()
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

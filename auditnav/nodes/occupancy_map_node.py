#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# occupancy_map_node.py
#
# Global occupancy-grid mapper based on lidar_mapper_v3_trust.py.
# Changes vs. original:
#   1) Publishes a "global map" instead of a fixed 25m×25m local window:
#      the grid auto-expands whenever the robot/observations exceed the boundary.
#   2) Uses Isaac Sim's 'world' frame as the global coordinate frame:
#      OccupancyGrid.header.frame_id = 'world'.
#   3) Core mapping parameters are unchanged (resolution, trust radius,
#      probability update magnitudes, height filter, etc.).
#
# Compatibility: simultaneously publishes /map and /local_map (same global map)
# to avoid modifying subscriber configurations in other nodes.

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.parameter import Parameter
import numpy as np
import tf2_ros
import cv2
import json
import os
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config, get



class OccupancyMapNode(Node):
    def __init__(self):
        super().__init__('occupancy_map_node')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        # Load centralised config
        self._cfg = load_config()

        # === 1. Map parameters (initial window) ===
        self.resolution = get(self._cfg, 'mapper', 'map_resolution', default=0.05)
        self.width_m = 25.0
        self.height_m = 25.0
        self.grid_w = int(self.width_m / self.resolution)
        self.grid_h = int(self.height_m / self.resolution)
        self.origin_x = -self.width_m / 2.0
        self.origin_y = -self.height_m / 2.0

        # Initialise: 50 = unknown
        self.prob_grid = np.full((self.grid_h, self.grid_w), 50.0, dtype=np.float32)
        self.display_grid = np.full((self.grid_h, self.grid_w), -1, dtype=np.int8)

        self.NUM_BINS = 720

        # === Trust radius (keep original value) ===
        self.TRUST_RADIUS_M = 6.0
        self.TRUST_RADIUS_PX = int(self.TRUST_RADIUS_M / self.resolution)

        # === 2. Auto-calibration ===
        self.calib_file = os.path.join(BASE_DIR, 'map_calibration.json')
        self.trajectory_points = []
        self.last_record_pos = None
        self.record_interval = 2.0

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_sensor_tf()

        # Subscriptions / Publishers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub_lidar = self.create_subscription(
            PointCloud2, get(self._cfg, 'topics', 'lidar', default='/point_cloud'), self.lidar_callback, qos
        )

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub_map = self.create_publisher(OccupancyGrid, get(self._cfg, 'topics', 'map', default='/map'), map_qos)
        self.pub_local_map = self.create_publisher(OccupancyGrid, get(self._cfg, 'topics', 'local_map', default='/local_map'), map_qos)

    def publish_sensor_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'sim_lidar'
        t.transform.translation.x = 0.1
        t.transform.translation.z = 0.42
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(t)

    def _expand_map_if_needed(self, min_x: int, max_x: int, min_y: int, max_y: int):
        """Auto-expand prob_grid / display_grid to accommodate out-of-bounds indices.

        Only modifies map boundary handling; core mapping rules are unchanged.
        Returns: (shift_x, shift_y) = (pad_left, pad_top) to offset old indices.
        """
        margin = max(8, int(self.TRUST_RADIUS_PX))

        pad_left = max(0, -min_x + margin)
        pad_top = max(0, -min_y + margin)
        pad_right = max(0, max_x - (self.grid_w - 1) + margin)
        pad_bottom = max(0, max_y - (self.grid_h - 1) + margin)

        if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
            return 0, 0

        self.prob_grid = np.pad(
            self.prob_grid,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=50.0,
        )
        self.display_grid = np.pad(
            self.display_grid,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=-1,
        )

        self.grid_h, self.grid_w = self.prob_grid.shape
        self.origin_x -= pad_left * self.resolution
        self.origin_y -= pad_top * self.resolution

        if self.trajectory_points:
            try:
                for e in self.trajectory_points:
                    if (isinstance(e, dict) and 'pixel' in e
                            and isinstance(e['pixel'], list) and len(e['pixel']) == 2):
                        e['pixel'][0] = int(e['pixel'][0]) + int(pad_left)
                        e['pixel'][1] = int(e['pixel'][1]) + int(pad_bottom)
                with open(self.calib_file, 'w') as f:
                    json.dump(self.trajectory_points, f, indent=4)
            except Exception:
                pass

        return pad_left, pad_top

    def lidar_callback(self, msg: PointCloud2):
        try:
            trans = self.tf_buffer.lookup_transform('world', msg.header.frame_id, Time())
            t, q = trans.transform.translation, trans.transform.rotation

            robot_world_x = t.x
            robot_world_y = t.y

            gx0 = int((t.x - self.origin_x) / self.resolution)
            gy0 = int((t.y - self.origin_y) / self.resolution)

            cloud_arr = np.frombuffer(
                msg.data,
                dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)],
                count=-1,
                offset=0,
            )
            pts_local = np.vstack((cloud_arr['x'], cloud_arr['y'], cloud_arr['z'])).T

            x, y, z, w = q.x, q.y, q.z, q.w
            R = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,   2*x*z + 2*y*w],
                [2*x*y + 2*z*w,   1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w,   2*y*z + 2*x*w,   1 - 2*x*x - 2*y*y],
            ])
            pts_global = np.dot(pts_local, R.T) + np.array([t.x, t.y, t.z])
            z_glob = pts_global[:, 2]

            dx = pts_global[:, 0] - t.x
            dy = pts_global[:, 1] - t.y
            dists = np.sqrt(dx**2 + dy**2)
            angles = np.arctan2(dy, dx)
            bin_indices = ((angles + np.pi) / (2 * np.pi) * self.NUM_BINS).astype(np.int32)
            bin_indices = np.clip(bin_indices, 0, self.NUM_BINS - 1)

            # Height filter (original thresholds kept)
            obs_mask = (z_glob > 0.2) & (z_glob < 2.5)

            horizon_dists = np.full(self.NUM_BINS, np.inf)
            if np.any(obs_mask):
                np.minimum.at(horizon_dists, bin_indices[obs_mask], dists[obs_mask])

            scan_limit_dists = np.zeros(self.NUM_BINS)
            np.maximum.at(scan_limit_dists, bin_indices, dists)
            clearing_dists = np.minimum(horizon_dists, scan_limit_dists)

            valid_bins = np.where(scan_limit_dists > 0.05)[0]
            valid_angles = valid_bins * (2 * np.pi) / self.NUM_BINS - np.pi
            valid_ranges = clearing_dists[valid_bins]

            end_x = t.x + valid_ranges * np.cos(valid_angles)
            end_y = t.y + valid_ranges * np.sin(valid_angles)
            end_gx0 = ((end_x - self.origin_x) / self.resolution).astype(np.int32)
            end_gy0 = ((end_y - self.origin_y) / self.resolution).astype(np.int32)

            obs_idx_x0 = ((pts_global[obs_mask, 0] - self.origin_x) / self.resolution).astype(np.int32)
            obs_idx_y0 = ((pts_global[obs_mask, 1] - self.origin_y) / self.resolution).astype(np.int32)

            cand_x = [gx0]
            cand_y = [gy0]
            if end_gx0.size > 0:
                cand_x += [int(end_gx0.min()), int(end_gx0.max())]
                cand_y += [int(end_gy0.min()), int(end_gy0.max())]
            if obs_idx_x0.size > 0:
                cand_x += [int(obs_idx_x0.min()), int(obs_idx_x0.max())]
                cand_y += [int(obs_idx_y0.min()), int(obs_idx_y0.max())]

            shift_x, shift_y = self._expand_map_if_needed(
                int(min(cand_x)), int(max(cand_x)),
                int(min(cand_y)), int(max(cand_y))
            )

            gx = gx0 + shift_x
            gy = gy0 + shift_y
            end_gx = end_gx0 + shift_x
            end_gy = end_gy0 + shift_y
            obs_idx_x = obs_idx_x0 + shift_x
            obs_idx_y = obs_idx_y0 + shift_y

            # Calibration record
            flipped_pixel_y = self.grid_h - 1 - gy
            current_pos = np.array([robot_world_x, robot_world_y])
            if (self.last_record_pos is None
                    or np.linalg.norm(current_pos - self.last_record_pos) > self.record_interval):
                if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                    entry = {
                        'pixel': [int(gx), int(flipped_pixel_y)],
                        'world': [float(robot_world_x), float(robot_world_y)],
                    }
                    self.trajectory_points.append(entry)
                    self.last_record_pos = current_pos
                    with open(self.calib_file, 'w') as f:
                        json.dump(self.trajectory_points, f, indent=4)

            # Free-space mask
            free_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
            for i in range(len(valid_bins)):
                x0c = int(np.clip(gx, 0, self.grid_w - 1))
                y0c = int(np.clip(gy, 0, self.grid_h - 1))
                x1c = int(np.clip(end_gx[i], 0, self.grid_w - 1))
                y1c = int(np.clip(end_gy[i], 0, self.grid_h - 1))
                cv2.line(free_mask, (x0c, y0c), (x1c, y1c), 255, 1)

            # Obstacle map
            obs_map = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
            if obs_idx_x.size > 0:
                valid_obs = (
                    (obs_idx_x >= 0) & (obs_idx_x < self.grid_w) &
                    (obs_idx_y >= 0) & (obs_idx_y < self.grid_h)
                )
                if np.any(valid_obs):
                    obs_map[obs_idx_y[valid_obs], obs_idx_x[valid_obs]] = 255

            # Trust mask
            trust_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
            cv2.circle(
                trust_mask,
                (int(np.clip(gx, 0, self.grid_w - 1)), int(np.clip(gy, 0, self.grid_h - 1))),
                self.TRUST_RADIUS_PX, 255, -1
            )

            # Update free space
            mask_far_free = (free_mask == 255) & (trust_mask == 0) & (self.prob_grid < 95.0)
            self.prob_grid[mask_far_free] -= 1.0
            mask_near_free = (free_mask == 255) & (trust_mask == 255) & (self.prob_grid < 95.0)
            self.prob_grid[mask_near_free] -= 3.0

            # Update obstacles
            mask_far_obs = (obs_map == 255) & (trust_mask == 0)
            update_far = mask_far_obs & (self.prob_grid < 70.0)
            self.prob_grid[update_far] += 3.0
            mask_near_obs = (obs_map == 255) & (trust_mask == 255) & (self.prob_grid > 5)
            self.prob_grid[mask_near_obs] += 15.0

            cv2.circle(
                self.prob_grid,
                (int(np.clip(gx, 0, self.grid_w - 1)), int(np.clip(gy, 0, self.grid_h - 1))),
                4, 0.0, -1
            )
            np.clip(self.prob_grid, 0.0, 100.0, out=self.prob_grid)

            # Publish
            self.display_grid[:] = -1
            self.display_grid[self.prob_grid > 90.0] = 100
            self.display_grid[self.prob_grid < 40.0] = 0

            occ_msg = OccupancyGrid()
            occ_msg.header.frame_id = 'world'
            occ_msg.header.stamp = self.get_clock().now().to_msg()
            occ_msg.info.resolution = self.resolution
            occ_msg.info.width = self.grid_w
            occ_msg.info.height = self.grid_h
            occ_msg.info.origin.position.x = float(self.origin_x)
            occ_msg.info.origin.position.y = float(self.origin_y)
            occ_msg.info.origin.orientation.w = 1.0
            occ_msg.data = self.display_grid.flatten().tolist()

            self.pub_map.publish(occ_msg)
            self.pub_local_map.publish(occ_msg)

        except Exception as e:
            import traceback
            self.get_logger().error(f"LiDAR callback error: {e}")
            self.get_logger().error(traceback.format_exc())


def main():
    rclpy.init()
    node = OccupancyMapNode()
    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

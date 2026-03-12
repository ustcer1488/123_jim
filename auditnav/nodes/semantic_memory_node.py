#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# semantic_memory_node.py

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config, get
import cv2
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["all_proxy"] = ""
os.environ["ALL_PROXY"] = ""
import time
import math
import re
import json
import threading
import argparse
import requests
import base64
import numpy as np
import chromadb
import torch
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Int32
from rclpy.executors import MultiThreadedExecutor

from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# Base directory for data output (relative to current working directory)
# BASE_DIR and DB_PATH resolved from config at startup (see VLMFinalNode.__init__)
_cfg_global = load_config()
BASE_DIR = _cfg_global["data"]["base_dir"]
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, "chroma_db_final")

def parse_args():
    parser = argparse.ArgumentParser(description="VLM Final Node")
    parser.add_argument("--freeze_memory", type=lambda x: str(x).lower() == "true",
                        default=False)
    parser.add_argument("--log_prefix", type=str, default="ours")
    parser.add_argument("--reset_db", action="store_true")
    parser.add_argument("--max_retry", type=int, default=3)
    args, _ = parser.parse_known_args()
    return args


CONFIG = {
    "API_KEY": _cfg_global["api"]["key"],
    "API_URL": get(_cfg_global, "api", "url", default="https://api.siliconflow.cn/v1/chat/completions"),
    "API_MODEL": get(_cfg_global, "api", "vlm_model", default="Qwen/Qwen3-VL-235B-A22B-Instruct"),
    "JSON_DATA_DIR": BASE_DIR,
    "DB_PATH": DB_PATH,
    "MODELS": {
        "EMBED": get(_cfg_global, "data", "embed_model_path", default="./bge-m3"),
    },
}


class SemanticAgent:
    def __init__(self, reset_db: bool = False):
        self.embed_model = None
        self.db_client = None
        self.collection = None
        self._init_db(reset_db=reset_db)
        self._load_models()

    def _init_db(self, reset_db: bool = False):
        print(f"Connecting to DB: {CONFIG['DB_PATH']}")
        try:
            self.db_client = chromadb.PersistentClient(
                path=CONFIG["DB_PATH"],
                settings=Settings(allow_reset=True),
            )
            if reset_db:
                self.db_client.reset()
                print("DB reset (--reset_db flag). Fresh start.")
            self.collection = self.db_client.get_or_create_collection(name="robot_memory")
        except Exception as e:
            print(f"DB Error: {e}")

    def _load_models(self):
        print("Loading BGE-M3 Embed Model...")
        try:
            self.embed_model = SentenceTransformer(
                CONFIG["MODELS"]["EMBED"],
                trust_remote_code=True,
                device="cuda:0",
            )
            self.embed_model.max_seq_length = 8192
            print(f"Embed model loaded. Planner=API: {CONFIG['API_MODEL']}")
        except Exception as e:
            print(f"Model load failed: {e}")

    def update_memory(self, node_id, json_path):
        if not os.path.exists(json_path):
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            narrative = json.dumps(data, ensure_ascii=False)
            vector = self.embed_model.encode(narrative, normalize_embeddings=True).tolist()
            doc_id = f"node_{node_id}_{int(time.time())}"
            self.collection.add(
                ids=[doc_id],
                documents=[narrative],
                embeddings=[vector],
                metadatas=[{"node_id": str(node_id), "timestamp": time.time()}],
            )
            print(f"  Node {node_id} memory stored.")
        except Exception as e:
            print(f"Update Memory Failed: {e}")

    def _call_planner_api(self, prompt_text: str) -> str:
        payload = {
            "model": CONFIG["API_MODEL"],
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.1,
            "max_tokens": 180,
        }
        try:
            resp = requests.post(
                CONFIG["API_URL"],
                json=payload,
                headers={"Authorization": f"Bearer {CONFIG['API_KEY']}"},
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return content.replace("```json", "").replace("```", "").strip()
        except Exception as e:
            return f'{{"node_id":"-1","score":0.0,"reason":"api_error:{str(e)[:60]}"}}'

    def search_object(self, query, avoid_node_ids=None, feedback=None):
        print(f"Searching for: {query}")
        try:
            avoid_set = set()
            if avoid_node_ids:
                for x in avoid_node_ids:
                    try:
                        avoid_set.add(str(int(x)))
                    except Exception:
                        if str(x).strip():
                            avoid_set.add(str(x).strip())

            query_vec = self.embed_model.encode(query, normalize_embeddings=True).tolist()
            results = self.collection.query(query_embeddings=[query_vec], n_results=8)

            candidates = []
            seen_nodes = set()

            if results.get("documents"):
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i]
                    nid = str(meta.get("node_id", "")).strip()
                    if not nid or nid in seen_nodes or nid in avoid_set:
                        continue
                    seen_nodes.add(nid)
                    try:
                        data = json.loads(doc)
                        room = data.get("place_info", {}).get("room_type", "unknown")
                        last_seen_mins = int(
                            time.time() - float(meta.get("timestamp", time.time()))
                        ) // 60
                        objs = [
                            f"{o.get('name','')}(HP:{o.get('health',3)})"
                            for o in data.get("detailed_objects", [])
                        ]
                        snippet = (f"Room: {room}, Last_seen: {last_seen_mins} mins ago, "
                                   f"Objects: {', '.join(objs)}")
                    except Exception:
                        snippet = doc[:800].replace("\n", " ")
                    candidates.append(f"[node_id={nid}] {snippet}")

            if not candidates:
                reason = "no_related_memory" if not avoid_set else "no_candidate_after_avoid"
                return json.dumps({"node_id": "-1", "score": 0.0, "reason": reason},
                                  ensure_ascii=False)

            context = "\n".join(candidates)
            avoid_note = (
                f"\n[Already attempted nodes - do NOT select these]\n"
                f"{', '.join(sorted(avoid_set))}\n"
                if avoid_set else ""
            )
            fb_note = f"\n[Latest feedback]\n{str(feedback)[:200]}\n" if feedback else ""

            prompt = f"""
You are a robot memory retrieval arbiter. Given a query target and candidate memory snippets,
select the single best-matching node_id.

[Query]
{query}
{avoid_note}{fb_note}
[Candidates] (room type, last-seen time, object HP status)
{context}

[Decision Rules]
1. Exact match wins. Among ties, prefer higher HP and more recent last_seen.
2. If no exact match, infer from room semantics (e.g., teddy bear -> bedroom).
3. Never select a node from the already-attempted list.

[Output]
Return ONLY one JSON object:
{{"node_id":"<id>","score":0.0-1.0,"reason":"brief reason"}}
""".strip()

            return self._call_planner_api(prompt)

        except Exception as e:
            return json.dumps(
                {"node_id": "-1", "score": 0.0, "reason": f"search_error:{str(e)[:60]}"},
                ensure_ascii=False,
            )


class SemanticMemoryNode(Node):
    def __init__(self, cli_args):
        super().__init__("semantic_memory_node")

        # Load centralised config
        self._cfg = load_config()
        self.cli = cli_args

        self.freeze_memory_update = self.cli.freeze_memory
        try:
            self.declare_parameter("freeze_memory_update", False)
            if not self.cli.freeze_memory:
                self.freeze_memory_update = bool(
                    self.get_parameter("freeze_memory_update").value)
        except Exception:
            pass

        self.agent = SemanticAgent(reset_db=self.cli.reset_db)
        self.bridge = CvBridge()
        self.latest_img = None
        self.pose = [0.0, 0.0, 0.0]
        self.is_working = False

        log_dir = CONFIG["JSON_DATA_DIR"]
        self.experiment_log_path = os.path.join(log_dir, f"{self.cli.log_prefix}_stats.csv")
        if not os.path.exists(self.experiment_log_path):
            with open(self.experiment_log_path, "w", encoding="utf-8") as f:
                f.write("Timestamp,NodeID_or_Query,ActionType,Metric1,Metric2\n")

        mode_tag = ("Baseline(freeze_memory=True)" if self.freeze_memory_update
                    else "Ours(dynamic audit)")
        self.get_logger().info(f"{mode_tag} | log -> {self.experiment_log_path}")

        self.search_sessions: dict = {}
        self.search_lock = threading.Lock()
        self.active_search_query = ""
        self.last_task_query = ""
        self.last_task_query_ts = 0.0
        self.task_query_ttl_sec = 0.0
        self.raw_task_query = ""
        self.abort_audit = False

        self.abs_angles_deg = [0, 45, 90, 135, 180, -135, -90, -45]
        self.compass_labels = [
            "EAST (0 deg)", "N-EAST (45 deg)", "NORTH (90 deg)", "N-WEST (135 deg)",
            "WEST (180 deg)", "S-WEST (-135 deg)", "SOUTH (-90 deg)", "S-EAST (-45 deg)",
        ]

        self._last_bbox_pub_ts = 0.0
        self._last_bbox_pub_node = -1
        self._last_bbox_pub_query = ""
        self._last_view_images: dict = {}
        self.bbox_debug_dir = os.path.join(CONFIG["JSON_DATA_DIR"], "bbox_debug")
        os.makedirs(self.bbox_debug_dir, exist_ok=True)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Odometry, "/odom", self.odom_cb, qos)
        self.create_subscription(Image, "/camera/rgb/image_raw", self.img_cb, qos)
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Int32, get(self._cfg, "topics", "vlm_trigger", default="/vlm/trigger_audit"), self.audit_trigger_cb, 10)
        self.pub_complete = self.create_publisher(String, get(self._cfg, "topics", "vlm_complete", default="/vlm/audit_complete"), 10)
        self.create_subscription(String, get(self._cfg, "topics", "instruction", default="/audit_nav/instruction"), self.search_cmd_cb, 10)
        self.create_subscription(String, "/audit_nav/raw_instruction", self.search_cmd_raw_cb, 10)
        self.create_subscription(String, get(self._cfg, "topics", "feedback", default="/audit_nav/feedback"), self.search_feedback_cb, 10)
        self.pub_res = self.create_publisher(String, get(self._cfg, "topics", "result", default="/audit_nav/result"), 10)
        self.pub_bbox = self.create_publisher(String, get(self._cfg, "topics", "object_bbox", default="/audit_nav/object_bbox"), 10)
        self.pub_bbox_debug = self.create_publisher(Image, get(self._cfg, "topics", "target_bbox_image", default="/audit_nav/target_bbox_image"), 1)

        self.get_logger().info("VLM Final Node ready.")

    # ── Basic callbacks ──────────────────────────────────────
    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

    def img_cb(self, msg):
        try:
            self.latest_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def _extract_json_payload(self, text: str):
        s = (text or "").strip()
        if not s:
            return None
        dec = json.JSONDecoder()
        for i, ch in enumerate(s):
            if ch in "[{":
                try:
                    obj, _ = dec.raw_decode(s[i:])
                    return obj
                except Exception:
                    continue
        return None

    # ── Search instruction callbacks ─────────────────────────
    def search_cmd_raw_cb(self, msg: String):
        query = (msg.data or "").strip()
        self.raw_task_query = query if query else ""

    def search_cmd_cb(self, msg: String):
        short_query = (msg.data or "").strip()
        self.abort_audit = True
        if not short_query:
            self.active_search_query = ""
            return

        query = (self.raw_task_query or "").strip() or short_query
        self.active_search_query = query
        self.last_task_query = query
        self.last_task_query_ts = time.time()

        with self.search_lock:
            self.search_sessions[query] = set()

        self.get_logger().info(f"Search Command: Find '{query}'")
        threading.Thread(
            target=self._async_search_with_context,
            args=(query, [], None),
            daemon=True,
        ).start()

    def search_feedback_cb(self, msg: String):
        payload = self._extract_json_payload(msg.data)
        if not isinstance(payload, dict):
            self.get_logger().warn("Feedback has no valid JSON payload.")
            return

        feedback_query = str(payload.get("query", "")).strip()
        if not feedback_query:
            return

        query = self.raw_task_query if getattr(self, 'raw_task_query', '') else feedback_query
        failed_nodes = payload.get("failed_nodes", [])
        last_failed = payload.get("last_failed_node", None)
        reason = str(payload.get("failure_reason", "")).strip()
        extra = payload.get("extra", None)

        with self.search_lock:
            s = self.search_sessions.setdefault(query, set())
            for x in (failed_nodes if isinstance(failed_nodes, list) else []):
                try:
                    s.add(str(int(x)))
                except Exception:
                    if str(x).strip():
                        s.add(str(x).strip())
            avoid_list = sorted(list(s))

        if len(avoid_list) >= self.cli.max_retry:
            self.get_logger().error(
                f"'{query}' not found after {self.cli.max_retry} attempts. Abort.")
            self._log_csv(query, "TASK_FAILED_LIMIT_REACHED", len(avoid_list), 0)
            out = String()
            out.data = json.dumps(
                {"node_id": "-1", "score": 0.0, "reason": "max_retry_reached"},
                ensure_ascii=False)
            self.pub_res.publish(out)
            return

        self._log_csv(query, "RECOVERY_TRIGGERED", len(avoid_list), 0)
        fb_text = (f"last_failed_node={last_failed}, reason={reason}, "
                   f"extra={str(extra)[:120] if extra else ''}")
        self.get_logger().info(f"Feedback for '{query}'. Avoid={avoid_list}")
        threading.Thread(
            target=self._async_search_with_context,
            args=(query, avoid_list, fb_text),
            daemon=True,
        ).start()

    def _async_search_with_context(self, query: str, avoid_nodes, feedback_text):
        if not rclpy.ok():
            return
        try:
            result_text = self.agent.search_object(
                query, avoid_node_ids=avoid_nodes, feedback=feedback_text)
        except Exception as e:
            result_text = json.dumps(
                {"node_id": "-1", "score": 0.0, "reason": f"search_error:{str(e)[:60]}"},
                ensure_ascii=False)
        if rclpy.ok():
            self.get_logger().info(f"Search Result: {result_text}")
            out = String()
            out.data = result_text
            self.pub_res.publish(out)

    # ── Audit pipeline ───────────────────────────────────────
    def audit_trigger_cb(self, msg):
        node_id = msg.data
        if self.is_working:
            self.get_logger().warn(f"Busy! Ignoring audit Node {node_id}")
            return
        self.get_logger().info(f"Trigger: Audit Node {node_id}")
        threading.Thread(
            target=self._execute_audit_pipeline, args=(node_id,), daemon=True
        ).start()

    def _execute_audit_pipeline(self, node_id):
        self.abort_audit = False
        self.is_working = True
        try:
            self.get_logger().info("Starting Panoramic Scan...")
            captured = []
            for i, deg in enumerate(self.abs_angles_deg):
                if not rclpy.ok() or self.abort_audit:
                    self.get_logger().warn(f"Audit Node {node_id} aborted by new instruction!")
                    return

                label = self.compass_labels[i]
                self.get_logger().info(f"Node {node_id}: -> {label}")
                self.rotate_to_absolute(math.radians(deg))
                time.sleep(0.5)
                img = (self.latest_img.copy() if self.latest_img is not None
                       else np.zeros((480, 640, 3), np.uint8))
                captured.append(img)

            if not rclpy.ok():
                return

            try:
                self._last_view_images = {
                    "E": captured[0], "NE": captured[1], "N": captured[2],
                    "NW": captured[3], "W": captured[4], "SW": captured[5],
                    "S": captured[6], "SE": captured[7],
                }
            except Exception:
                self._last_view_images = {}

            panorama = self.create_stitched_panorama(captured)
            pano_path = os.path.join(CONFIG["JSON_DATA_DIR"], f"node_{node_id}.jpg")
            cv2.imwrite(pano_path, panorama)

            json_path = os.path.join(CONFIG["JSON_DATA_DIR"], f"{node_id}.json")
            old_data = {"detailed_objects": []}
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    old_data = json.load(f)

            if rclpy.ok():
                self.get_logger().info("Calling VLM API...")
            target_query = self._get_task_query_for_audit()
            old_mem_str = json.dumps(old_data.get("detailed_objects", []), ensure_ascii=False)
            new_json_str = self.call_vlm_arbitrator(panorama, old_mem_str, target_query=target_query)

            try:
                if new_json_str and target_query:
                    self._debug_target_detection_in_audit(int(node_id), target_query, new_json_str)
            except Exception:
                pass

            try:
                if new_json_str:
                    tmp = self._extract_json_payload(new_json_str)
                    if (isinstance(tmp, dict) and
                            isinstance(tmp.get("detailed_objects"), list) and
                            len(tmp["detailed_objects"]) == 0):
                        self.get_logger().warn(
                            f"Node {node_id}: detailed_objects empty, retrying...")
                        retry = self._audit_retry_if_empty(panorama, old_mem_str, target_query)
                        if retry:
                            new_json_str = retry
            except Exception:
                pass

            if new_json_str and rclpy.ok():
                self.process_and_save(
                    node_id, new_json_str, old_data.get("detailed_objects", []), json_path)
                self.get_logger().info("Updating Vector Memory...")
                self.agent.update_memory(node_id, json_path)

                try:
                    published = False
                    if target_query:
                        published = bool(
                            self._maybe_publish_target_bbox(
                                node_id, new_json_str, target_query, panorama))
                    if target_query and not published:
                        det_str = self.call_vlm_target_bbox_panorama(panorama, target_query)
                        if det_str:
                            ok = self._publish_target_bbox_from_det_panorama(
                                node_id, det_str, target_query, panorama_bgr=panorama)
                            if ok:
                                self.get_logger().info("Fallback bbox published.")
                                published = True
                            else:
                                self.get_logger().info("Fallback bbox: not visible.")
                    if target_query and not published:
                        self._apply_bbox_veto(node_id, target_query, json_path)
                except Exception as bbox_err:
                    self.get_logger().warn(f"BBox publish error (non-fatal): {bbox_err}")

        except Exception as e:
            if rclpy.ok():
                self.get_logger().error(f"Audit Pipeline Error: {e}")
        finally:
            self.is_working = False
            if rclpy.ok():
                try:
                    msg = String()
                    msg.data = f"DONE_{node_id}"
                    self.pub_complete.publish(msg)
                    self.get_logger().info(f"Audit Complete Node {node_id}.")
                except Exception:
                    pass

    # ── Helpers ──────────────────────────────────────────────
    def _log_csv(self, query_or_node, action_type, metric1, metric2):
        try:
            with open(self.experiment_log_path, "a", encoding="utf-8") as f:
                q = str(query_or_node).replace(",", " ")
                f.write(f"{time.time()},{q},{action_type},{metric1},{metric2}\n")
        except Exception:
            pass

    def _get_task_query_for_audit(self) -> str:
        q = (self.raw_task_query or "").strip() or (self.last_task_query or "").strip()
        if not q:
            return ""
        try:
            ts = float(self.last_task_query_ts or 0.0)
            ttl = float(self.task_query_ttl_sec or 0.0)
        except Exception:
            return q
        if ttl <= 0 or (time.time() - ts) <= ttl:
            return q
        return ""

    def _debug_target_detection_in_audit(self, node_id: int, target_query: str, new_json_str: str):
        data = self._extract_json_payload(new_json_str)
        if not isinstance(data, dict):
            return
        td = data.get("target_detection", None)
        if not isinstance(td, dict):
            return
        found = bool(td.get("found", False))
        view = str(td.get("view", "")).strip()
        bbox = td.get("bbox", [])
        conf = float(td.get("confidence", 0.0) or 0.0)
        reason = str(td.get("reason", "")).strip()
        status = "FOUND" if found else "NOT_FOUND"
        self.get_logger().info(
            f"Audit Node {node_id}: {status} view={view} bbox={bbox} "
            f"conf={conf:.2f} | {reason[:80]}")

    # ── Rotation control ─────────────────────────────────────
    def rotate_to_absolute(self, target_yaw):
        timeout = time.time() + 12.0
        while rclpy.ok() and time.time() < timeout and not getattr(self, 'abort_audit', False):
            diff = target_yaw - self.pose[2]
            while diff > math.pi: diff -= 2 * math.pi
            while diff < -math.pi: diff += 2 * math.pi
            if abs(diff) < 0.04:
                break
            vel = diff * 1.5
            if abs(diff) > 0.1:
                if abs(vel) < 0.65: vel = 0.65 * (1 if vel > 0 else -1)
            else:
                if abs(vel) < 0.3: vel = 0.3 * (1 if vel > 0 else -1)
            vel = max(min(vel, 0.7), -0.7)
            cmd = Twist()
            cmd.angular.z = float(vel)
            self.pub_vel.publish(cmd)
            time.sleep(0.05)
        stop_cmd = Twist()
        for _ in range(8):
            if not rclpy.ok(): break
            self.pub_vel.publish(stop_cmd)
            time.sleep(0.05)

    # ── Panorama stitching ───────────────────────────────────
    def create_stitched_panorama(self, images):
        h, w, _ = images[0].shape
        processed = []
        for i, img in enumerate(images):
            canvas = cv2.resize(img, (w // 2, h // 2))
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
            cv2.putText(canvas, self.compass_labels[i], (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            processed.append(canvas)
        row1 = cv2.hconcat([processed[3], processed[2], processed[1], processed[0]])
        row2 = cv2.hconcat([processed[7], processed[6], processed[5], processed[4]])
        return cv2.vconcat([row1, row2])

    # ── VLM API calls ─────────────────────────────────────────
    def call_vlm_arbitrator(self, img, old_mem_str, target_query=""):
        if not old_mem_str or not str(old_mem_str).strip():
            old_mem_str = "[]"
        old_mem_str = str(old_mem_str).strip()

        _, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        b64_img = base64.b64encode(buffer).decode("utf-8")

        prompt_text = """# Role
You are a High-Precision Robot Spatial Memory Manager.

Panoramic image layout (2x4 grid):
  Top row (left->right): NW, N, NE, E
  Bottom row (left->right): SE, S, SW, W

# Task
1) Scan ALL 8 views. For TARGET_QUERY (if non-empty), search explicitly every view.
   - NEVER hallucinate. Only list objects DIRECTLY and CLEARLY visible.
   - Include tabletop objects (lamps, plants, vases on furniture surfaces).
   - Skip blank walls/ceilings/floors.
   - Aim for 8-15 high-value landmarks.
   - Each object: id, name, status, view, spatial_context, visual_description.
2) Compare with OLD_MEMORY:
   - Matched: keep id, status="visible".
   - Missing: status="gone". Blocked: status="occluded".
3) New objects: assign ids starting from 100.
4) target_detection:
   - If TARGET_QUERY empty: found=false, bbox=[].
   - If visible: found=true, best view. bbox can be [] here.
   - If not visible: found=false. DO NOT infer from room context.

# Output
Return ONLY raw JSON (no markdown).
Schema:
{
  "place_info": {"room_type": "string", "description": "string"},
  "detailed_objects": [
    {"id": 100, "name": "string", "status": "visible|gone|occluded",
     "view": "E/NE/N/NW/W/SW/S/SE", "spatial_context": "string",
     "visual_description": "string"}
  ],
  "target_detection": {
    "query": "string", "found": true/false,
    "view": "E/NE/N/NW/W/SW/S/SE",
    "bbox": [], "confidence": 0.0-1.0, "reason": "string"
  }
}"""

        text_block = (f"{prompt_text}\n\n"
                      f"### OLD_MEMORY ###\n{old_mem_str}\n\n"
                      f"### TARGET_QUERY ###\n{target_query}")
        payload = {
            "model": CONFIG["API_MODEL"],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": text_block},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
            ]}],
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        try:
            start_time = time.time()
            estimated_tokens = len(str(payload)) // 4
            resp = requests.post(
                CONFIG["API_URL"], json=payload,
                headers={"Authorization": f"Bearer {CONFIG['API_KEY']}"},
                timeout=600)
            resp.raise_for_status()
            latency = time.time() - start_time
            self._log_csv("AUDIT_API_CALL", "LLM_COST", estimated_tokens, f"{latency:.2f}")
            content = resp.json()["choices"][0]["message"]["content"]
            return (content or "").replace("```json", "").replace("```", "").strip()
        except Exception as e:
            if rclpy.ok():
                self.get_logger().error(f"API Error: {e}")
            return None

    def _audit_retry_if_empty(self, panorama_bgr, old_mem_str, target_query):
        if panorama_bgr is None:
            return None
        if not old_mem_str or not str(old_mem_str).strip():
            old_mem_str = "[]"
        try:
            _, buffer = cv2.imencode(".jpg", panorama_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            b64_img = base64.b64encode(buffer).decode("utf-8")
        except Exception:
            return None

        prompt_text = """You are an object enumerator for a robot semantic map.
Panorama (2x4 grid: Top=NW,N,NE,E | Bottom=SE,S,SW,W).
Return ONLY raw JSON with place_info, detailed_objects (8-15 significant objects),
and target_detection. If TARGET_QUERY is empty or not directly visible: found=false.
Do NOT hallucinate. No markdown."""

        text_block = (f"{prompt_text}\n\n"
                      f"### OLD_MEMORY ###\n{old_mem_str}\n\n"
                      f"### TARGET_QUERY ###\n{target_query}")
        payload = {
            "model": CONFIG["API_MODEL"],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": text_block},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
            ]}],
            "temperature": 0.0,
            "max_tokens": 1200,
        }
        try:
            resp = requests.post(
                CONFIG["API_URL"], json=payload,
                headers={"Authorization": f"Bearer {CONFIG['API_KEY']}"},
                timeout=600)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return (content or "").replace("```json", "").replace("```", "").strip()
        except Exception:
            return None

    # ── BBox utilities ───────────────────────────────────────
    def _norm_view_token(self, view_raw: str) -> str:
        v = (view_raw or "").strip().upper().replace(" ", "")
        for src, dst in [
            ("N-EAST", "NE"), ("NORTHEAST", "NE"), ("N-WEST", "NW"), ("NORTHWEST", "NW"),
            ("S-EAST", "SE"), ("SOUTHEAST", "SE"), ("S-WEST", "SW"), ("SOUTHWEST", "SW"),
            ("EAST", "E"), ("WEST", "W"), ("NORTH", "N"), ("SOUTH", "S"),
        ]:
            v = v.replace(src, dst)
        return v if v in {"E", "NE", "N", "NW", "W", "SW", "S", "SE"} else ""

    def _view_to_tile_xy(self, view_token: str, pano_w: int, pano_h: int):
        if pano_w <= 0 or pano_h <= 0:
            return None
        tw = pano_w // 4
        th = pano_h // 2
        mapping = {
            "NW": (0, 0), "N": (1, 0), "NE": (2, 0), "E": (3, 0),
            "SE": (0, 1), "S": (1, 1), "SW": (2, 1), "W": (3, 1),
        }
        if view_token not in mapping:
            return None
        col, row = mapping[view_token]
        return col * tw, row * th, tw, th

    def _bbox_1000_to_pixels(self, bbox_1000, tw: int, th: int):
        if not (isinstance(bbox_1000, (list, tuple)) and len(bbox_1000) == 4):
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox_1000]
        except Exception:
            return None
        x1 = max(0.0, min(1000.0, x1)); y1 = max(0.0, min(1000.0, y1))
        x2 = max(0.0, min(1000.0, x2)); y2 = max(0.0, min(1000.0, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        px1 = max(0, min(tw - 1, int(round(x1 / 1000.0 * (tw - 1)))))
        py1 = max(0, min(th - 1, int(round(y1 / 1000.0 * (th - 1)))))
        px2 = max(0, min(tw - 1, int(round(x2 / 1000.0 * (tw - 1)))))
        py2 = max(0, min(th - 1, int(round(y2 / 1000.0 * (th - 1)))))
        if px2 <= px1 or py2 <= py1:
            return None
        return px1, py1, px2, py2

    def _safe_filename(self, s: str) -> str:
        s = (s or "").strip()
        if not s: return "unknown"
        return re.sub(r"[^0-9A-Za-z._-]+", "_", s)[:48]

    def _bbox_area_ratio_1000(self, bbox):
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            return 0.0
        return max(0, x2 - x1) * max(0, y2 - y1) / 1_000_000.0

    def _bbox_border_touch_count(self, bbox, eps=8):
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            return 0
        return sum([x1 <= eps, y1 <= eps, x2 >= 1000 - eps, y2 >= 1000 - eps])

    def _is_large_furniture(self, query: str) -> bool:
        q = (query or "").strip().lower()
        large = {"bed", "sofa", "couch", "wardrobe", "cabinet", "dresser",
                 "table", "desk", "tv", "door", "curtain", "bookshelf"}
        return q in large

    def _bbox_plausible(self, query: str, bbox) -> bool:
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return False
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        except Exception:
            return False
        if x2 <= x1 or y2 <= y1:
            return False
        x1 = max(0, min(1000, x1)); y1 = max(0, min(1000, y1))
        x2 = max(0, min(1000, x2)); y2 = max(0, min(1000, y2))
        area = self._bbox_area_ratio_1000([x1, y1, x2, y2])
        touch = self._bbox_border_touch_count([x1, y1, x2, y2], eps=8)
        if area >= 0.92: return False
        if touch >= 3 and area >= 0.55: return False
        if (not self._is_large_furniture(query)) and area >= 0.65: return False
        if area <= 0.002: return False
        return True

    def _draw_and_publish_bbox(self, node_id, panorama_bgr, query, view_token,
                                bbox_1000, conf, reason):
        if panorama_bgr is None:
            return
        try:
            pano = panorama_bgr.copy()
        except Exception:
            return
        ph, pw = pano.shape[:2]
        tile = self._view_to_tile_xy(view_token, pw, ph)
        if tile is None:
            return
        x0, y0, tw, th = tile
        px = self._bbox_1000_to_pixels(bbox_1000, tw, th)
        if px is None:
            return
        x1, y1, x2, y2 = px
        ax1, ay1 = x0 + x1, y0 + y1
        ax2, ay2 = x0 + x2, y0 + y2
        cv2.rectangle(pano, (ax1, ay1), (ax2, ay2), (0, 255, 0), 3)
        label = f"{query} [{view_token}] conf={conf:.2f}"
        cv2.putText(pano, label, (max(0, ax1), max(15, ay1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if reason:
            cv2.putText(pano, reason[:48], (max(0, ax1), min(ph - 10, ay2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        try:
            qf = self._safe_filename(query)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(
                self.bbox_debug_dir, f"node_{node_id}_{qf}_{view_token}_{ts}.jpg")
            cv2.imwrite(out_path, pano)
        except Exception:
            pass
        try:
            msg = self.bridge.cv2_to_imgmsg(pano, encoding="bgr8")
            self.pub_bbox_debug.publish(msg)
        except Exception:
            pass
        try:
            tile_img = panorama_bgr[y0:y0 + th, x0:x0 + tw].copy()
            cv2.rectangle(tile_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(tile_img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            tile_path = os.path.join(
                self.bbox_debug_dir,
                f"node_{node_id}_{self._safe_filename(query)}_{view_token}"
                f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}_tile.jpg")
            cv2.imwrite(tile_path, tile_img)
        except Exception:
            pass

    # ── Two-stage bbox refinement ────────────────────────────
    def _neighbors_of_view(self, v: str):
        m = {
            "E": ["NE", "SE"], "NE": ["N", "E"], "N": ["NW", "NE"], "NW": ["W", "N"],
            "W": ["SW", "NW"], "SW": ["S", "W"], "S": ["SE", "SW"], "SE": ["E", "S"],
        }
        return m.get(v, [])

    def call_vlm_target_bbox_single_view(self, img_bgr, target_query: str, view_hint: str = ""):
        if img_bgr is None or not target_query:
            return None
        try:
            _, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            b64_img = base64.b64encode(buffer).decode("utf-8")
        except Exception:
            return None
        vh = f"This image is the {view_hint.strip().upper()} view." if view_hint.strip() else ""
        prompt = f"""{vh}
Precise visual grounding detector. TARGET_QUERY = single object name.
1) Decide if TARGET_QUERY is clearly and directly visible. Do NOT infer from context.
2) If visible, output tight bbox normalized [0,1000]: [x1,y1,x2,y2].
3) If not confident, output found=false, bbox=[].
Return ONLY raw JSON:
{{"query":"string","found":true/false,"bbox":[x1,y1,x2,y2]or[],"confidence":0.0-1.0,"reason":"string"}}""".strip()

        payload = {
            "model": CONFIG["API_MODEL"],
            "messages": [{"role": "user", "content": [
                {"type": "text",
                 "text": f"{prompt}\n\n### TARGET_QUERY ###\n{target_query}"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
            ]}],
            "temperature": 0.0,
            "max_tokens": 180,
        }
        try:
            resp = requests.post(
                CONFIG["API_URL"], json=payload,
                headers={"Authorization": f"Bearer {CONFIG['API_KEY']}"}, timeout=240)
            content = resp.json()["choices"][0]["message"]["content"]
            return (content or "").replace("```json", "").replace("```", "").strip()
        except Exception as e:
            if rclpy.ok():
                self.get_logger().warn(f"VLM single-view bbox API error: {str(e)[:80]}")
            return None

    def _parse_det_bbox(self, det_str: str):
        if not det_str:
            return False, None, 0.0, ""
        data = self._extract_json_payload(det_str)
        if not isinstance(data, dict):
            return False, None, 0.0, ""
        found = bool(data.get("found", False))
        bbox = data.get("bbox", [])
        if not found or not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return False, None, 0.0, str(data.get("reason", "")).strip()
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        except Exception:
            return False, None, 0.0, str(data.get("reason", "")).strip()
        x1 = max(0, min(1000, x1)); y1 = max(0, min(1000, y1))
        x2 = max(0, min(1000, x2)); y2 = max(0, min(1000, y2))
        if x2 <= x1 or y2 <= y1:
            return False, None, 0.0, str(data.get("reason", "")).strip()
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0)))
        reason = str(data.get("reason", "")).strip()[:160]
        return True, [x1, y1, x2, y2], conf, reason

    def _refine_bbox_with_single_view(self, query: str, preferred_view: str):
        view_imgs = getattr(self, "_last_view_images", {}) or {}
        order = [preferred_view] + self._neighbors_of_view(preferred_view)
        best = None
        for v in order:
            img = view_imgs.get(v)
            if img is None:
                continue
            det_str = self.call_vlm_target_bbox_single_view(img, query, view_hint=v)
            ok, bbox, conf, reason = self._parse_det_bbox(det_str)
            if not ok:
                continue
            if best is None or conf > best["confidence"]:
                best = {"view": v, "bbox": bbox, "confidence": conf, "reason": reason}
        return best

    def call_vlm_target_bbox_panorama(self, pano_bgr, target_query: str):
        if pano_bgr is None or not target_query:
            return None
        try:
            _, buffer = cv2.imencode(".jpg", pano_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_img = base64.b64encode(buffer).decode("utf-8")
        except Exception:
            return None
        prompt = """High-precision visual detector. Panorama (2x4): Top=NW,N,NE,E | Bottom=SE,S,SW,W.
Decide if TARGET_QUERY is clearly visible. Do NOT infer from context.
If visible: select ONE view, output bbox [x1,y1,x2,y2] normalized [0,1000] for that view.
If not confident: found=false, bbox=[].
Return ONLY raw JSON:
{"query":"string","found":true/false,"view":"E/NE/N/NW/W/SW/S/SE","bbox":[],"confidence":0.0-1.0,"reason":"string"}""".strip()

        payload = {
            "model": CONFIG["API_MODEL"],
            "messages": [{"role": "user", "content": [
                {"type": "text",
                 "text": f"{prompt}\n\n### TARGET_QUERY ###\n{target_query}"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
            ]}],
            "temperature": 0.0,
            "max_tokens": 180,
        }
        try:
            resp = requests.post(
                CONFIG["API_URL"], json=payload,
                headers={"Authorization": f"Bearer {CONFIG['API_KEY']}"}, timeout=240)
            content = resp.json()["choices"][0]["message"]["content"]
            return (content or "").replace("```json", "").replace("```", "").strip()
        except Exception as e:
            if rclpy.ok():
                self.get_logger().warn(f"VLM panorama bbox API error: {str(e)[:80]}")
            return None

    def _publish_bbox_payload(self, node_id, q, view, bbox, conf, reason, panorama_bgr=None):
        payload = {
            "query": q, "found": True, "node_id": int(node_id),
            "view": view, "bbox": bbox, "confidence": conf, "reason": reason[:140]
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_bbox.publish(msg)
        self._last_bbox_pub_ts = time.time()
        self._last_bbox_pub_node = int(node_id)
        self._last_bbox_pub_query = str(q)
        self.get_logger().info(f"BBox published: view={view} bbox={bbox} conf={conf:.2f}")
        try:
            if panorama_bgr is not None:
                self._draw_and_publish_bbox(int(node_id), panorama_bgr, q, view, bbox, conf, reason)
        except Exception:
            pass

    def _maybe_publish_target_bbox(self, node_id, new_json_str, target_query,
                                    panorama_bgr=None) -> bool:
        if not target_query or not new_json_str:
            return False
        data = self._extract_json_payload(new_json_str)
        if not isinstance(data, dict):
            return False
        td = data.get("target_detection", None)
        if not isinstance(td, dict) or not td.get("found", False):
            return False
        q = str(td.get("query", "")).strip() or str(target_query).strip()
        view_token = self._norm_view_token(str(td.get("view", "")).strip())
        if not view_token:
            return False

        refined = self._refine_bbox_with_single_view(q, view_token)
        if (refined and isinstance(refined.get("bbox"), (list, tuple)) and
                len(refined["bbox"]) == 4):
            rb = [int(round(float(v))) for v in refined["bbox"]]
            rconf = max(0.0, min(1.0, float(refined.get("confidence", 0.0) or 0.0)))
            if self._bbox_plausible(q, rb):
                self._publish_bbox_payload(
                    node_id, q, str(refined.get("view", view_token)).strip(),
                    rb, rconf, str(refined.get("reason", "")).strip()[:140], panorama_bgr)
                return True
            else:
                self.get_logger().warn(
                    f"Refined bbox rejected: view={view_token} bbox={rb} conf={rconf:.2f}")

        bbox = td.get("bbox", None)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                ib = [max(0, min(1000, int(round(float(v))))) for v in bbox]
            except Exception:
                return False
            conf = max(0.0, min(1.0, float(td.get("confidence", 0.0) or 0.0)))
            if ib[2] > ib[0] and ib[3] > ib[1] and self._bbox_plausible(q, ib):
                self._publish_bbox_payload(
                    node_id, q, view_token, ib, conf,
                    str(td.get("reason", "")).strip()[:140], panorama_bgr)
                return True
        return False

    def _publish_target_bbox_from_det_panorama(self, node_id, det_str, target_query,
                                                panorama_bgr=None) -> bool:
        if not (det_str and target_query):
            return False
        data = self._extract_json_payload(det_str)
        if not isinstance(data, dict) or not data.get("found", False):
            reason = str(data.get("reason", "")).strip() if isinstance(data, dict) else ""
            self.get_logger().info(
                f"Fallback bbox NOT found ('{target_query}'), reason={reason[:80]}")
            return False
        q = str(data.get("query", "")).strip() or str(target_query).strip()
        view_token = self._norm_view_token(str(data.get("view", "")).strip())
        bbox = data.get("bbox", [])
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and view_token):
            return False
        try:
            ib = [max(0, min(1000, int(round(float(v))))) for v in bbox]
        except Exception:
            return False
        if ib[2] <= ib[0] or ib[3] <= ib[1]:
            return False
        conf = max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0)))
        reason = str(data.get("reason", "")).strip()[:140]
        if not self._bbox_plausible(q, ib):
            self.get_logger().warn(
                f"Fallback bbox rejected: view={view_token} bbox={ib} conf={conf:.2f}")
            return False
        self._publish_bbox_payload(node_id, q, view_token, ib, conf, reason, panorama_bgr)
        return True

    # ── BBox veto (hallucination penalty) ────────────────────
    def _apply_bbox_veto(self, node_id, target_query, json_path):
        """If the object cannot be bbox-localized, penalize its HP in memory."""
        if self.freeze_memory_update:
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            current_query = target_query.lower()
            target_mapping = {
                "toilet": ["toilet"], "washing machine": ["washing machine", "washer"],
                "microwave": ["microwave"], "bed": ["bed"],
                "sofa": ["sofa", "couch"], "tv": ["tv", "television"],
                "refrigerator": ["refrigerator", "fridge"],
                "chair": ["chair", "stool", "armchair"],
                "table": ["table", "desk", "nightstand"],
                "plant": ["potted", "plant", "fig", "flower"],
                "door": ["door"],
            }

            target_keywords = []
            for key, en_keys in target_mapping.items():
                if key in current_query:
                    target_keywords.extend(en_keys)
            if not target_keywords:
                target_keywords = [current_query]

            pruned_count = 0
            objs = data.get("detailed_objects", [])
            alive_objs = []

            for obj in objs:
                obj_name_lower = obj.get("name", "").lower()
                is_hallucinated = any(kw in obj_name_lower for kw in target_keywords)
                if is_hallucinated:
                    obj["health"] = 0
                    pruned_count += 1
                    self.get_logger().warn(
                        f"BBox veto: penalized [{obj['name']}] HP={obj['health']}")
                if obj.get("health", 3) > 0:
                    alive_objs.append(obj)
                else:
                    self.get_logger().info(f"Pruned [{obj['name']}] HP=0 from Node {node_id}")

            data["detailed_objects"] = alive_objs
            if pruned_count > 0:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self.agent.update_memory(node_id, json_path)
                self.get_logger().info(
                    f"Hallucination penalty applied. Node {node_id} memory updated.")
        except Exception as e:
            self.get_logger().error(f"BBox Veto Error: {e}")

    # ── Memory save ──────────────────────────────────────────
    def process_and_save(self, node_id, new_data_str, old_objects_list, file_path):
        try:
            new_data = self._extract_json_payload(new_data_str)
            if not isinstance(new_data, dict):
                self.get_logger().error(
                    f"Cannot extract valid JSON from model output: {str(new_data_str)[:150]}")
                return

            final_list = []
            old_map = {
                obj["id"]: obj for obj in old_objects_list
                if isinstance(obj, dict) and "id" in obj
            }
            pruned_count = 0
            updated_count = 0

            print(f"\n{'='*20} MEMORY AUDIT (Node {node_id}) {'='*20}")
            print(f"Mode: {'Baseline(Append-Only)' if self.freeze_memory_update else 'Ours(Dynamic)'}")

            for obj in new_data.get("detailed_objects", []):
                if not isinstance(obj, dict):
                    self.get_logger().warn(f"Skipping malformed object: {obj}")
                    continue

                oid = obj.get("id")
                if oid is None:
                    import random
                    oid = -random.randint(1000, 9999)

                name = obj.get("name", "Unknown")
                health = old_map.get(oid, {}).get("health", 3)
                status = obj.get("status", "visible")

                if self.freeze_memory_update:
                    status = "visible"
                    health = 3
                else:
                    if status == "gone":
                        health -= 1
                    else:
                        health = min(health + 1, 3)

                if health > 0:
                    obj["health"] = health
                    final_list.append(obj)
                    updated_count += 1
                    print(f"  OK ID:{oid} {name} ({status}) HP:{health}")
                else:
                    pruned_count += 1
                    print(f"  PRUNED ID:{oid} {name}")

            save_json = {
                "node_id": node_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "robot_pose": {"x": self.pose[0], "y": self.pose[1], "yaw": self.pose[2]},
                "place_info": new_data.get("place_info", {}),
                "detailed_objects": final_list,
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(save_json, f, ensure_ascii=False, indent=2)
            print(f"  Saved: {file_path}")
            self._log_csv(node_id, "AUDIT_UPDATE", updated_count, pruned_count)

        except Exception as e:
            if rclpy.ok():
                self.get_logger().error(f"JSON Save Error: {e}")


def main():
    cli_args = parse_args()
    rclpy.init()
    node = SemanticMemoryNode(cli_args)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

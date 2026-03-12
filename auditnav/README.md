> *AuditNav: VLM-Audited Hierarchical Embodied Navigation Framework with Dynamic Memory and Open-Vocabulary Adaptation*

---

## Repository Structure

```
auditnav/                              ← ROS 2 package root (ament_python)
├── package.xml                        ← ROS 2 package manifest
├── setup.py                           ← Python package entry points
├── setup.cfg
├── config_loader.py                   ← Shared YAML config loader (all nodes import this)
├── requirements.txt                   ← Python pip dependencies
├── .gitignore
│
├── nodes/                             ← All ROS 2 nodes
│   ├── topo_nav_node.py               ← Central navigator: state machine, A*, topology, VLM patrol
│   ├── semantic_memory_node.py        ← VLM audit: ChromaDB episodic memory, panoramic scan
│   ├── open_vocab_perception_node.py  ← Open-vocabulary detection: YOLO-World + Kalman tracking
│   ├── instruction_parser_node.py     ← LLM middleware: NL → structured YOLO query
│   └── occupancy_map_node.py          ← LiDAR → global occupancy grid
│
├── planners/
│   └── astar_planner.py               ← Standalone A* planner (imported as a library)
│
├── launch/
│   └── auditnav.launch.py             ← One-command launch (all nodes, ordered startup)
│
└── config/
    └── default_params.yaml            ← ALL tunable parameters in one place
```

---

## Reproduction Steps

### Prerequisites

| Requirement | Version |
|---|---|
| Ubuntu | 22.04 LTS |
| ROS 2 | Humble Hawksbill |
| Python | 3.10 |
| CUDA | 11.8+ (GPU required for YOLO-World) |
| GPU VRAM | ≥ 8 GB |
| RAM | ≥ 16 GB |

---

### Step 1 — Install ROS 2 Humble

Follow the official guide: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html

```bash
# Quick summary (Ubuntu 22.04)
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
     -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
     http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
     | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop python3-colcon-common-extensions -y

# Source ROS 2 in every terminal (or add to ~/.bashrc)
source /opt/ros/humble/setup.bash
```

Install additional ROS 2 packages:
```bash
sudo apt install -y \
  ros-humble-cv-bridge \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs
```

---

### Step 2 — Clone the Repository

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/<anonymous>/auditnav.git
```

---

### Step 3 — Install Python Dependencies

```bash
cd ~/ros2_ws/src/auditnav

# (Recommended) create a virtual env or use system Python
pip install -r requirements.txt
```

If you have CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If you have CUDA 12.x:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### Step 4 — Download Model Weights

#### 4a. YOLO-World (Large) — required by `open_vocab_perception_node`
```bash
cd ~/ros2_ws/src/auditnav/nodes
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-world.pt
```

#### 4b. BGE-M3 Embedding Model — required by `semantic_memory_node`
```bash
cd ~/ros2_ws/src/auditnav
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-m3', local_dir='./bge-m3')
"
```

Then open `config/default_params.yaml` and set:
```yaml
data:
  embed_model_path: "./bge-m3"   # ← update if you downloaded to a different path
```

---

### Step 5 — Set API Key

The system uses [SiliconFlow](https://siliconflow.cn) for VLM inference.  
Register and get a free key at https://siliconflow.cn, then:

```bash
# Add to ~/.bashrc so it persists across terminals
echo 'export SILICONFLOW_API_KEY="sk-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

The key is **never stored in any source file** — it is always read from this environment variable.

---

### Step 6 — Build the ROS 2 Package

```bash
cd ~/ros2_ws

# Install any missing ROS dependencies declared in package.xml
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --packages-select auditnav --symlink-install

# Source the workspace (add to ~/.bashrc to make permanent)
source ~/ros2_ws/install/setup.bash
```

> `--symlink-install` means edits to Python files in `src/` take effect immediately without rebuilding.

---

### Step 7 — (No Setup Required) Map Calibration

**You do not need to create any calibration file.** The system automatically converts pixel coordinates to world coordinates using the standard `resolution` and `origin` fields that ROS 2 already embeds in every `/map` (`OccupancyGrid`) message.

On the **first run**, the navigator builds a topological graph, converts all node positions to world coordinates, and saves them to:
```
data/final_node_coordinates_<timestamp>.json
```

On every **subsequent run**, this file is loaded directly — no map needed, no calibration needed.

> The optional `map_calibration.json` is only relevant if you want to supply manually measured ground-truth correspondences to override the automatic conversion. For normal use, skip it entirely.

---

### Step 8 — (Optional) Tune Parameters

All parameters live in **one file**:
```bash
nano ~/ros2_ws/src/auditnav/config/default_params.yaml
```

Key parameters you may want to adjust for your robot:

```yaml
navigator:
  base_speed: 0.60        # m/s — reduce for slower/heavier robots
  arrival_dist: 0.30      # m   — goal-reached threshold

planner:
  robot_radius: 0.15      # m   — increase for wider robots

perception:
  model_path: "nodes/yolov8l-world.pt"   # path to YOLO-World weights

topics:
  lidar: "/point_cloud"           # change to match your robot's LiDAR topic
  rgb_image: "/camera/rgb/image_raw"
  depth_image: "/camera/depth/image_raw"
```

---

### Step 9 — Launch the Full System

```bash
# Make sure your simulator / robot driver is already running and publishing sensor topics

ros2 launch auditnav auditnav.launch.py
```

**With a real robot** (disable sim clock):
```bash
ros2 launch auditnav auditnav.launch.py use_sim_time:=false
```

**With a custom config file**:
```bash
ros2 launch auditnav auditnav.launch.py config:=/abs/path/to/my_params.yaml
```

The launch file starts nodes in order automatically:

| Time | Node | Role |
|---|---|---|
| t = 0 s | `occupancy_map_node` | LiDAR → `/map` |
| t = 3 s | `open_vocab_perception_node` | YOLO-World detection |
| t = 3 s | `semantic_memory_node` | VLM + ChromaDB memory |
| t = 3 s | `instruction_parser_node` | NL → YOLO query |
| t = 6 s | `topo_nav_node` | Central navigator |

---

### Step 10 — Send a Navigation Goal

Once all nodes are running:
```bash
ros2 topic pub /audit_nav/instruction std_msgs/String "data: 'sofa'" --once
```

The robot will:
1. Explore the environment (frontier-based)
2. Build a topological graph
3. Patrol graph nodes and build semantic memory via VLM
4. Retrieve the most relevant node for "sofa"
5. Navigate to it with YOLO-World visual confirmation

---

### Step 11 — Record Evaluation Events

```bash
ros2 topic echo /audit_nav/eval_event > eval_log.jsonl
```

The log contains JSON lines for each event:

```jsonc
{"event": "reach",        "kind": "final_found", "instruction": "sofa", "conf": 0.72, ...}
{"event": "collision",    "reason": "persistent_collision", "count": 1, ...}
{"event": "mission_done", "soft_found": true, "final_found": true, "collisions": 2, ...}
```

---

## Running Individual Nodes

After `colcon build`, each node is also available via `ros2 run`:

```bash
ros2 run auditnav occupancy_map_node
ros2 run auditnav open_vocab_perception_node
ros2 run auditnav semantic_memory_node
ros2 run auditnav instruction_parser_node
ros2 run auditnav topo_nav_node
```

Or run directly with Python (no colcon needed, for quick testing):
```bash
cd ~/ros2_ws/src/auditnav
python3 nodes/occupancy_map_node.py
```

---

## Topic Reference

### `/audit_nav/*` — Main Interface

| Topic | Type | Publisher → Subscriber | Description |
|---|---|---|---|
| `/audit_nav/instruction` | `String` | `instruction_parser_node` → `topo_nav_node`, `open_vocab_perception_node` | Structured target label |
| `/audit_nav/result` | `String` | `semantic_memory_node` → `topo_nav_node` | JSON: ranked node candidates from memory |
| `/audit_nav/feedback` | `String` | `topo_nav_node` → `semantic_memory_node` | JSON: failed node IDs for re-ranking |
| `/audit_nav/current_goal` | `PointStamped` | `open_vocab_perception_node` → `topo_nav_node` | Goal point in `base_link` |
| `/audit_nav/confidence` | `Float32` | `open_vocab_perception_node` → `topo_nav_node` | EMA detection confidence |
| `/audit_nav/object_bbox` | `String` | `semantic_memory_node` → `topo_nav_node` | JSON: VLM-confirmed bbox + view direction |
| `/audit_nav/debug_image` | `Image` | `open_vocab_perception_node` | YOLO annotated stream |
| `/audit_nav/target_bbox_image` | `Image` | `semantic_memory_node` | Panorama with bbox overlay |
| `/audit_nav/eval_event` | `String` | `topo_nav_node` | JSON evaluation events |

### `/vlm/*` — Internal Handshake

| Topic | Type | Description |
|---|---|---|
| `/vlm/trigger_audit` | `Int32` | Navigator → Memory: trigger VLM audit at node ID |
| `/vlm/audit_complete` | `String` | Memory → Navigator: audit finished |

### Standard ROS Topics (inputs)

| Topic | Type | Description |
|---|---|---|
| `/map` | `OccupancyGrid` | Global occupancy grid (from `occupancy_map_node`) |
| `/odom` | `Odometry` | Robot odometry |
| `/cmd_vel` | `Twist` | Velocity commands (published by navigator) |
| `/camera/rgb/image_raw` | `Image` | RGB (`bgr8`) |
| `/camera/depth/image_raw` | `Image` | Depth (`32FC1`, metres) |
| `/point_cloud` | `PointCloud2` | 3D LiDAR point cloud |

---

## Runtime Outputs (`./data/`, gitignored)

| Path | Description |
|---|---|
| `data/topology_graph_*.json` | Topological graph: node types, edges, pixel positions |
| `data/final_node_coordinates_*.json` | World-frame coordinates (reloaded across sessions) |
| `data/chroma_db_final/` | ChromaDB episodic memory store |
| `data/nav_dashboard_*.mp4` | Dashboard recording (global map + robot-centred view at 10 Hz) |

---

## Ablation / Baseline Flags

`semantic_memory_node` supports flags for the ablation study in the paper:

```bash
# Static memory baseline (no dynamic updates)
ros2 run auditnav semantic_memory_node --freeze_memory true

# Wipe memory and start fresh
ros2 run auditnav semantic_memory_node --reset_db

# Custom log prefix for CSV output
ros2 run auditnav semantic_memory_node --log_prefix baseline_static
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'config_loader'`**  
→ Run from the repo root, or ensure `~/ros2_ws/install/setup.bash` is sourced.

**`FileNotFoundError: map_calibration.json`**  
→ Place the calibration file in the working directory from which nodes are launched.

**`SILICONFLOW_API_KEY not set`**  
→ `export SILICONFLOW_API_KEY="sk-..."` and relaunch.

**YOLO-World not found**  
→ Check `config/default_params.yaml` → `perception.model_path` points to `yolov8l-world.pt`.

**`/map` topic not received by navigator**  
→ The mapper needs a few seconds to start. The launch file already adds a 6 s delay before the navigator; if your machine is slow, increase `period=6.0` in `launch/auditnav.launch.py`.

---

# AuditNav — Step-by-Step Reproduction Guide

> This guide walks you through every command needed to reproduce the system from scratch on a clean Ubuntu 22.04 machine.  
> Estimated total setup time: **30–60 minutes** (mostly model downloads).

---

## Environment Overview

| Item | Requirement |
|---|---|
| OS | Ubuntu 22.04 LTS |
| ROS 2 | Humble Hawksbill |
| Python | 3.10 (ships with Ubuntu 22.04) |
| GPU | CUDA-capable, ≥ 8 GB VRAM |
| CUDA | 11.8 or 12.x |
| RAM | ≥ 16 GB |
| Disk | ≥ 20 GB free (models + data) |

---

## Part 1 — Install ROS 2 Humble

> Skip if ROS 2 Humble is already installed. Test with `ros2 --version`.

```bash
# 1-1. Add ROS 2 apt repository
sudo apt update && sudo apt install curl gnupg lsb-release -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
     -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) \
     signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
     http://packages.ros.org/ros2/ubuntu \
     $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
     | sudo tee /etc/apt/sources.list.d/ros2.list

# 1-2. Install ROS 2 Desktop + build tools
sudo apt update
sudo apt install ros-humble-desktop python3-colcon-common-extensions python3-rosdep -y

# 1-3. Install required ROS 2 bridge packages
sudo apt install -y \
  ros-humble-cv-bridge \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  ros-humble-message-filters

# 1-4. Source ROS 2 (add to ~/.bashrc to make permanent)
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Verify
ros2 --version   # should print: ros2 cli version X.X.X
```

---

## Part 2 — Create ROS 2 Workspace and Clone the Repo

```bash
# 2-1. Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# 2-2. Clone AuditNav (or copy the zip and unzip here)
git clone https://github.com/<anonymous>/auditnav.git
# — OR — if you have the zip:
# unzip auditnav_release.zip
# mv auditnav ~/ros2_ws/src/

# 2-3. Verify structure
ls ~/ros2_ws/src/auditnav/
# Expected: nodes/  planners/  launch/  config/  package.xml  setup.py  ...
```

---

## Part 3 — Install Python Dependencies

```bash
# 3-1. Install pip packages
cd ~/ros2_ws/src/auditnav
pip install -r requirements.txt

# 3-2. Install PyTorch with the correct CUDA version
# Check your CUDA version first:
nvcc --version   # or: nvidia-smi

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3-3. Verify GPU is visible to PyTorch
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
# Should print: CUDA available: True
```

---

## Part 4 — Download Model Weights

### 4-1. YOLO-World (Large) — used by `open_vocab_perception_node`

```bash
cd ~/ros2_ws/src/auditnav/nodes

wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-world.pt

# Verify
ls -lh yolov8l-world.pt   # should be ~87 MB
```

### 4-2. BGE-M3 Embedding Model — used by `semantic_memory_node`

```bash
cd ~/ros2_ws/src/auditnav

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-m3', local_dir='./bge-m3')
print('BGE-M3 download complete')
"

# Verify
ls bge-m3/   # should contain: config.json, tokenizer files, model weights, etc.
```

> **Slow network?** You can also download manually from https://huggingface.co/BAAI/bge-m3 and place the folder at `~/ros2_ws/src/auditnav/bge-m3/`.

### 4-3. (Optional) Update model paths in config

The default config already points to the correct relative paths. Check:

```bash
grep "model_path\|embed_model_path" ~/ros2_ws/src/auditnav/config/default_params.yaml
```

Expected output:
```
  embed_model_path: "./bge-m3"
  model_path: "yolov8l-world.pt"
```

If you downloaded models to different locations, edit the paths here.

---

## Part 5 — Set the API Key

The system uses [SiliconFlow](https://siliconflow.cn) for VLM/LLM inference.  
Register a free account at https://siliconflow.cn and copy your API key.

```bash
# Add to ~/.bashrc so it persists across terminals
echo 'export SILICONFLOW_API_KEY="sk-xxxxxxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc

# Verify
echo $SILICONFLOW_API_KEY   # should print your key
```

> The key is never stored in any source file — only read from this environment variable at runtime.

---

## Part 6 — Build the ROS 2 Package

```bash
cd ~/ros2_ws

# 6-1. Initialize rosdep (first time only)
sudo rosdep init      # skip if already done
rosdep update

# 6-2. Auto-install any missing ROS dependencies
rosdep install --from-paths src --ignore-src -r -y

# 6-3. Build
colcon build --packages-select auditnav --symlink-install

# Expected output (last lines):
# Starting >>> auditnav
# Finished <<< auditnav [...]
# Summary: 1 packages finished [...]

# 6-4. Source the workspace
source ~/ros2_ws/install/setup.bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc

# 6-5. Verify all 5 executables are registered
ros2 pkg executables auditnav
# Expected:
# auditnav occupancy_map_node
# auditnav open_vocab_perception_node
# auditnav semantic_memory_node
# auditnav instruction_parser_node
# auditnav topo_nav_node
```

> `--symlink-install` means edits to Python files take effect immediately — no rebuild needed.

---

## Part 7 — Configure Topics for Your Robot / Simulator

Open the config file:

```bash
nano ~/ros2_ws/src/auditnav/config/default_params.yaml
```

Check the `topics:` section and match your robot's actual topic names:

```yaml
topics:
  lidar:       "/point_cloud"           # your LiDAR PointCloud2 topic
  rgb_image:   "/camera/rgb/image_raw"  # RGB image (bgr8)
  depth_image: "/camera/depth/image_raw"# Depth image (32FC1, in metres)
  camera_info: "/camera_info"
  odom:        "/odom"
  cmd_vel:     "/cmd_vel"
```

Check what topics your simulator/robot is actually publishing:

```bash
ros2 topic list
ros2 topic info /your_lidar_topic   # check the message type
```

---

## Part 8 — Launch the Full System

```bash
# 8-1. Open a new terminal, source environment
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 8-2. Make sure your simulator or robot driver is already running
#      (Isaac Sim / Gazebo / real robot — must be publishing sensor topics)

# 8-3. Launch all nodes
ros2 launch auditnav auditnav.launch.py
```

You should see nodes starting in sequence in the terminal:

```
[INFO] [launch]: AuditNav — Full System Launch
[occupancy_map_node]: LiDAR mapper started
... (3 seconds later)
[AuditNav] Starting perception / memory / commander...
[open_vocab_perception_node]: Loading YOLO-World model (Large)...
[semantic_memory_node]: Loading BGE-M3 Embed Model...
[instruction_parser_node]: Ready
... (3 more seconds)
[AuditNav] Starting topo_nav_node...
[topo_nav_node]: Waiting for /map ...
[topo_nav_node]: Map received. Starting exploration.
```

**For a real robot** (turn off simulator clock):
```bash
ros2 launch auditnav auditnav.launch.py use_sim_time:=false
```

---

## Part 9 — Send a Navigation Goal

Once all nodes are running and you see `[topo_nav_node]: Map received`:

```bash
# Open a new terminal
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Send a navigation instruction
ros2 topic pub /audit_nav/instruction std_msgs/String "data: 'sofa'" --once
```

The system will:
1. Parse the instruction via LLM → structured YOLO label
2. Explore the environment using frontier-based navigation
3. Build a topological graph from the occupancy map
4. Patrol graph nodes and build semantic memory via VLM panoramic scan
5. Query ChromaDB to retrieve the best candidate node for "sofa"
6. Navigate to the target node with YOLO-World visual confirmation

---

## Part 10 — Monitor the System

Open additional terminals to monitor each component:

```bash
# Watch navigation feedback
ros2 topic echo /audit_nav/feedback

# Watch detection confidence
ros2 topic echo /audit_nav/confidence

# Watch VLM memory results
ros2 topic echo /audit_nav/result

# Record evaluation events
ros2 topic echo /audit_nav/eval_event > eval_log.jsonl
```

View the annotated camera stream:
```bash
ros2 run rqt_image_view rqt_image_view /audit_nav/debug_image
```

---

## Part 11 — Check Runtime Outputs

All outputs are saved automatically to `./data/` (relative to launch directory):

```bash
ls ~/ros2_ws/data/

# Expected files after first run:
# topology_graph_<timestamp>.json        — topological graph
# final_node_coordinates_<timestamp>.json — node world coordinates (reloaded on next run)
# chroma_db_final/                       — ChromaDB memory store
# nav_dashboard_<timestamp>.mp4          — video recording of navigation
```

---

## Troubleshooting

**`ros2 pkg executables auditnav` shows nothing**
```bash
# Rebuild and re-source
cd ~/ros2_ws && colcon build --packages-select auditnav --symlink-install
source ~/ros2_ws/install/setup.bash
```

**`ModuleNotFoundError: No module named 'config_loader'`**
```bash
# Make sure workspace is sourced
source ~/ros2_ws/install/setup.bash
```

**`CUDA available: False`**
```bash
# Reinstall PyTorch with the correct CUDA version (Step 3-2)
# Check CUDA version: nvidia-smi
```

**YOLO-World model not found**
```bash
# Check the path — must be inside nodes/
ls ~/ros2_ws/src/auditnav/nodes/yolov8l-world.pt
```

**BGE-M3 fails to load**
```bash
# Check the path in config
grep embed_model_path ~/ros2_ws/src/auditnav/config/default_params.yaml
# Check the folder exists
ls ~/ros2_ws/src/auditnav/bge-m3/config.json
```

**`[topo_nav_node]: Waiting for /map ...` — stuck forever**
```bash
# Check occupancy_map_node is running and receiving LiDAR
ros2 topic hz /map              # should show ~2 Hz
ros2 topic hz /point_cloud      # should show data
```

**API errors (401 / connection refused)**
```bash
# Check key is set
echo $SILICONFLOW_API_KEY
# Test directly
curl https://api.siliconflow.cn/v1/models \
  -H "Authorization: Bearer $SILICONFLOW_API_KEY"
```

**`message_filters` not found**
```bash
sudo apt install ros-humble-message-filters
```

---

## Running Nodes Individually (for debugging)

After `colcon build`, each node can be run standalone:

```bash
# Each of these in a separate terminal (source workspace first):
ros2 run auditnav occupancy_map_node
ros2 run auditnav open_vocab_perception_node
ros2 run auditnav semantic_memory_node
ros2 run auditnav instruction_parser_node
ros2 run auditnav topo_nav_node
```

Or run directly with Python (no build needed, for quick testing):
```bash
cd ~/ros2_ws/src/auditnav
python3 nodes/occupancy_map_node.py
```

---

## Ablation / Baseline Experiments

```bash
# Freeze memory baseline (no dynamic updates — for ablation study)
ros2 run auditnav semantic_memory_node --freeze_memory true --log_prefix baseline

# Wipe memory and start fresh
ros2 run auditnav semantic_memory_node --reset_db

# Custom log prefix for CSV output
ros2 run auditnav semantic_memory_node --log_prefix exp_run1
```

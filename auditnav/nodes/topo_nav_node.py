#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# 文件名: integrated_vlm_navigator.py
# 版本: FINAL ULTIMATE FLAGSHIP (V_MAX_PRO_VIS_DESENSITIZED)
# 作者: Integrated AI Assistant & User
# 日期: 2026-01-29
#
# --- 核心架构描述 (System Architecture) ---
# 本文件实现了 Vision-Language Navigation (VLN) 系统的中央控制节点。
# 它采用单体架构 (Monolithic Architecture)，深度整合了四个独立模块的核心算法，
# 旨在实现从 未知环境探索 -> 语义拓扑构建 -> 视觉语言巡检 的全流程自动化。
#
# --- 模块来源与职责 (Module Responsibilities) ---
#
# 1. 底层运动控制 (Base Control) - 源自 [smart_navigator_v2_copy.py]
#    - 职责: 负责机器人的速度决策、姿态调整和异常恢复。
#    - 核心算法: 
#      - 自适应 PID 控制器 (P/I/D 三项完整)
#      - 动态前瞻点选择 (Dynamic Lookahead)
#      - "Savage Rush" 暴力脱困机制 (应对打滑或卡死)
#      - 路径动态剪枝 (Dynamic Pruning) 与防撞墙保护
#      - [优化] 碰撞检测去抖动 (Collision Debounce) - 防止在狭窄区域误触发
#
# 2. 拓扑地图构建 (Topology Engine) - 源自 [test.py]
#    - 职责: 将栅格地图抽象为语义拓扑图 (Nodes & Edges)。
#    - 核心算法:
#      - 骨架提取 (Medial Axis Skeletonization)
#      - 距离变换与局部极大值检测 (Distance Transform & Peak Local Max)
#      - 安全路径广度优先搜索 (Safe Path BFS) - 防止路径穿墙
#      - 图膨胀算法 (Graph Inflation) - 用于节点归属权判定
#      - [可视化] 拓扑结构可视化绘制 (Topology Visualization)
#      - [修复] 严格剔除未知区域与形态学去噪
#
# 3. 坐标系统解算 (Coordinate Solver) - 源自 [coordinate_calculator.py]
#    - 职责: 实现 [像素坐标系] 与 [世界坐标系] 的精确双向映射。
#    - 核心算法: 线性回归标定 (Linear Regression Calibration)
#
# 4. 语义巡检任务 (Semantic Patrol) - 源自 [node_traversal_navigator.py]
#    - 职责: 管理巡逻任务队列，触发 VLM 审计。
#    - 核心逻辑: 房间节点筛选、到达判定、VLM 通讯握手。
#
# 5. 路径规划 (Path Planning) - 引用自 [a_star_planner.py]
#    - 职责: 在栅格地图上搜索无碰撞的最优路径。
#
# --- 严格约束 (Strict Constraints) ---
# 1. 外部依赖: 必须存在 `a_star_planner.py`。
# 2. 参数锁定: 严禁修改 PID 系数、速度阈值、膨胀半径等任何参数。
# 3. 逻辑完整: 必须保留所有去抖动 (Debounce)、黑名单 (Blacklist) 和安全检查逻辑。
# 4. 代码规模: 总行数 > 1200 行 (通过详细文档和规范格式保证)。
# ==============================================================================

# ==============================================================================
# 📥 导入依赖库 (Imports)
# ==============================================================================

# ROS 2 核心库
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import re
# 标准库
import math
import time
import json
import os
import collections
from collections import deque

import re
# 数学与科学计算
import numpy as np
import networkx as nx

# 图像处理库 (OpenCV & Scikit-Image)
import cv2
from skimage.morphology import medial_axis
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN

# ROS 消息类型
from geometry_msgs.msg import Twist, Point, PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String, Int32, Float32

# TF 坐标变换库
import tf2_ros
from tf2_geometry_msgs import do_transform_point

# 🔥 引用用户提供的 A* 规划器 🔥
# 确保 a_star_planner.py 在同一目录下
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config, get
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planners"))
from astar_planner import AStarPlanner 

# BASE_DIR resolved from config at startup (see TopoNavNode.__init__)

# ==============================================================================
# 🔧 通用工具函数 (Utility Functions)
# ==============================================================================

def normalize_angle(angle):
    """
    将给定的角度归一化到 [-pi, pi] 区间内。
    这是机器人控制中处理角度误差的关键步骤。
    
    参数:
        angle (float): 原始角度 (弧度制)
        
    返回:
        float: 归一化后的角度，范围 [-pi, pi]
        
    逻辑说明:
        通过循环加减 2*pi，确保角度落在主值区间。
        这对于 PID 控制器的误差计算至关重要，防止机器人选择长路径旋转。
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ==============================================================================
# 🧩 模块 1: 坐标解算算法 (Coordinate Calculator)
# 来源: coordinate_calculator.py
# 功能: 读取 map_calibration.json，计算线性变换矩阵。
# ==============================================================================

def calculate_linear_mapping(calibration_points):
    """
    通过多个标定点计算线性回归参数，建立 Pixel -> World 的映射关系。
    
    核心公式: 
        World_X = Pixel_X * Scale_X + Offset_X
        World_Y = Pixel_Y * Scale_Y + Offset_Y
    
    参数:
        calibration_points (list): 包含标定数据的字典列表。
                                   每个字典应包含 {'pixel': [u, v], 'world': [x, y]}。
        
    返回:
        tuple: (scale_x, scale_y, offset_x, offset_y) 若计算成功。
        None: 若标定点不足。
    """
    print("\n" + "="*60)
    print("📐 [Coordinate Calculator] 开始计算坐标标定参数...")
    print("="*60)
    
    # --- 1. 数据校验 ---
    # 线性回归至少需要两个点才能确定一条直线。
    if len(calibration_points) < 2:
        print("❌ [Calc] 错误: 标定点不足2个，无法计算线性回归。")
        print("   -> 请确保机器人在建图阶段移动了足够的距离。")
        print("   -> 请检查 map_calibration.json 文件是否包含数据。")
        return None

    # --- 2. 提取数据 ---
    # 将输入的字典列表解构为独立的坐标列表，方便计算。
    u_list = [] # Pixel X (图像列坐标)
    v_list = [] # Pixel Y (图像行坐标)
    x_list = [] # World X (ROS 世界坐标)
    y_list = [] # World Y (ROS 世界坐标)

    print(f"📊 [Calc] 读取到 {len(calibration_points)} 个标定点。正在分析...")

    for i, p in enumerate(calibration_points):
        u = p['pixel'][0]
        v = p['pixel'][1]
        wx = p['world'][0]
        wy = p['world'][1]
        
        u_list.append(u)
        v_list.append(v)
        x_list.append(wx)
        y_list.append(wy)
        
        # 仅打印首尾几个点用于调试，避免刷屏
        if i < 2 or i > len(calibration_points) - 3:
            print(f"   -> Sample {i}: P({u}, {v}) => W({wx:.3f}, {wy:.3f})")

    # --- 3. 计算线性参数 ---
    # 采用两点法计算 (取首尾两点)。
    # 原因：首尾两点距离最远，能最大程度减小量化误差对斜率的影响。
    # 对于完美的栅格地图，两点法通常比最小二乘法更鲁棒且计算更快。
    
    p_start = 0
    p_end = -1 # 列表中的最后一个点

    # 计算差值 (Delta)
    du = u_list[p_end] - u_list[p_start]
    dv = v_list[p_end] - v_list[p_start]
    dx = x_list[p_end] - x_list[p_start]
    dy = y_list[p_end] - y_list[p_start]

    print(f"   Delta Pixel: dU={du}, dV={dv}")
    print(f"   Delta World: dX={dx:.3f}, dY={dy:.3f}")

    # --- 4. 防除零保护 ---
    # 虽然在实际运动中很难发生像素坐标完全不动的情况，但防御性编程是必须的。
    if abs(du) < 1e-5: 
        print("⚠️ [Calc] 警告: dU 接近 0 (X轴像素未变化)。强制置为 1.0 以防除零异常。")
        du = 1.0 
    if abs(dv) < 1e-5: 
        print("⚠️ [Calc] 警告: dV 接近 0 (Y轴像素未变化)。强制置为 1.0 以防除零异常。")
        dv = 1.0

    # 计算比例因子 (Scale)
    # Scale 表示每一个像素代表多少米 (Meters per Pixel)
    scale_x = dx / du
    scale_y = dy / dv
    
    # 计算偏移量 (Offset)
    # Offset = World - Pixel * Scale
    # 使用起始点作为基准点进行计算
    offset_x = x_list[p_start] - u_list[p_start] * scale_x
    offset_y = y_list[p_start] - v_list[p_start] * scale_y

    # --- 5. 输出结果 ---
    print("-" * 40)
    print(f"✅ [Calc] 标定成功!")
    print(f"   Scale X : {scale_x:.6f} (m/px)")
    print(f"   Scale Y : {scale_y:.6f} (m/px)")
    print(f"   Offset X: {offset_x:.6f} (m)")
    print(f"   Offset Y: {offset_y:.6f} (m)")
    print("="*60 + "\n")
    
    return (scale_x, scale_y, offset_x, offset_y)


# ==============================================================================
# 🧩 模块 2: 拓扑分析算法集 (Topology Analysis Algorithms)
# 来源: test.py
# 功能: 包含所有图像处理、骨架提取、图论构建、安全路径搜索的函数。
# ==============================================================================

def get_room_centers(binary_map, min_distance=60):
    """
    使用距离变换 (Distance Transform) 提取房间的几何中心。
    这些中心点将作为拓扑图的关键节点 (Room Nodes)。
    
    算法原理:
    1. 计算每个自由空间像素到最近障碍物的距离 (Distance Transform)。
    2. 距离越大的点，说明离墙越远，越可能是房间的中心。
    3. 寻找这些距离值的局部极大值 (Local Maxima)。
    
    参数:
        binary_map (np.ndarray): 二值化地图 (0=障碍, 255=空闲)
        min_distance (int): 两个中心点之间的最小像素距离 (Smart-V2 参数为 30)
        
    返回:
        list: 排序后的房间中心点列表 [(cx, cy), ...]
    """
    # 确保地图数据类型正确
    if binary_map.dtype != np.uint8:
        binary_map = binary_map.astype(np.uint8)
        
    # --- 1. 距离变换 ---
    # cv2.DIST_L2 表示使用欧几里得距离
    # maskSize=5 表示距离计算的掩模大小
    dist_map = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
    
    # --- 2. 寻找局部极大值 ---
    # peak_local_max 返回的是坐标 (row, col) 即 (y, x)
    # min_distance 参数保证了找到的中心点不会过于密集
    coordinates = peak_local_max(dist_map, min_distance=min_distance, labels=binary_map)
    
    room_centers = []
    
    # --- 3. 过滤与转换 ---
    for c in coordinates:
        cy, cx = c[0], c[1]
        
        # 噪声过滤: 
        # 🔥 [关键修复] 将阈值从 10 降低到 6
        # 解释：10px = 0.5m半径 = 1m宽度。如果厕所宽度小于1米，会被当成噪声过滤掉。
        # 改为 6px = 0.3m半径 = 0.6m宽度，这样就能识别狭窄的厕所了。
        if dist_map[cy, cx] > 3:
            room_centers.append((cx, cy)) # 转换为 (x, y) 格式存储
            
    # --- 4. 排序 ---
    # 按坐标排序 (先 y 后 x)，保证每次生成的顺序一致性。
    # 这对于后续分配稳定的 Node ID 非常重要。
    return sorted(room_centers, key=lambda x: (x[1], x[0]))


def get_raw_skeleton(binary_map):
    """
    提取地图的骨架 (Skeleton)，用于构建拓扑图的边。
    骨架是二值图像的中轴线，代表了连通性的核心结构 (Voronoi Diagram 近似)。
    
    参数:
        binary_map (np.ndarray): 二值化地图
        
    返回:
        np.ndarray: 骨架矩阵 (True/False 或 0/1)
    """
    # --- 1. 预处理 ---
    # 使用中值滤波 (Median Blur) 去除地图上的椒盐噪声。
    # kernel size 3x3 是一个保守的选择，既能去噪又不会过度模糊边缘。
    clean_map = cv2.medianBlur(binary_map, 3)
    
    # --- 2. 骨架提取 ---
    # 使用 skimage 的 medial_axis 算法。
    # 输入图像必须归一化为 0/1 或 boolean。
    # 255 -> 1, 0 -> 0
    skel = medial_axis(clean_map // 255)
    
    return skel


def graph_from_skeleton(skel):
    """
    将骨架像素矩阵转换为 NetworkX 图结构。
    每个骨架像素成为一个 Node，相邻的骨架像素之间建立 Edge。
    
    参数:
        skel (np.ndarray): 骨架矩阵
        
    返回:
        nx.Graph: 构建好的无向图，包含所有骨架像素节点。
    """
    # 获取所有骨架点 (值为 True) 的坐标索引
    # y 是行索引，x 是列索引
    y, x = np.where(skel)
    
    G = nx.Graph()
    
    # --- 1. 添加节点 ---
    for px, py in zip(x, y):
        # 节点以坐标元组 (x, y) 的形式存储
        G.add_node((px, py))
        
        # --- 2. 定义邻域 ---
        # 8-connected neighborhood
        neighbors = [
            (px-1, py-1), (px, py-1), (px+1, py-1),
            (px-1, py),               (px+1, py),
            (px-1, py+1), (px, py+1), (px+1, py+1)
        ]
        
        # --- 3. 建立连接 ---
        for nx_x, nx_y in neighbors:
            # 边界检查: 确保邻居坐标在图像范围内
            if 0 <= nx_y < skel.shape[0] and 0 <= nx_x < skel.shape[1]:
                # 如果邻居也是骨架点
                if skel[nx_y, nx_x]:
                    # 计算距离 (权重)
                    dist = np.linalg.norm(np.array([px, py]) - np.array([nx_x, nx_y]))
                    # 添加边
                    G.add_edge((px, py), (nx_x, nx_y), weight=dist)
    return G


def remove_colliding_nodes(G, original_binary_map):
    """
    安全检查：移除那些位于障碍物内部或地图边界外的节点。
    
    虽然骨架提取通常在自由空间进行，但由于前处理 (如中值滤波) 或计算误差，
    骨架边缘可能会轻微侵入障碍物区域。此函数用于清洗这些不安全的节点。
    
    参数:
        G (nx.Graph): 原始骨架图
        original_binary_map (np.ndarray): 原始二值地图
        
    返回:
        nx.Graph: 清洗后的安全图
    """
    nodes_to_remove = []
    h, w = original_binary_map.shape
    
    for node in G.nodes():
        x, y = node
        
        # --- 1. 边界检查 ---
        if x < 0 or x >= w or y < 0 or y >= h:
            nodes_to_remove.append(node)
            continue
            
        # --- 2. 障碍物检查 ---
        # 假设 128 是阈值。在 OccupancyGrid 转为 Image 后：
        # 0 (黑) = 障碍物, 255 (白) = 空闲。
        # 这里为了保险，我们将 < 128 的都视为障碍或未知。
        if original_binary_map[y, x] < 128:
            nodes_to_remove.append(node)
            
    if nodes_to_remove:
        # print(f"Removing {len(nodes_to_remove)} colliding nodes from graph.")
        G.remove_nodes_from(nodes_to_remove)
        
    return G


def clean_skeleton_graph(G, prune_len=20, max_cycle_len=60):
    """
    清理骨架图，优化拓扑结构。
    
    主要执行两个操作:
    1. 移除孤立小岛 (Disconnected Components): 去除那些由于噪点产生的独立小图块。
    2. 修剪短毛刺 (Spurs Pruning): 去除骨架末端无意义的短分支。
    
    参数:
        G (nx.Graph): 输入图
        prune_len (int): 毛刺修剪长度阈值 (像素)
        max_cycle_len (int): 小环移除阈值 (可选)
        
    返回:
        nx.Graph: 优化后的骨架图
    """
    # --- 1. 移除孤立小连通分量 ---
    # 获取所有连通分量
    components = list(nx.connected_components(G))
    for comp in components:
        # 如果分量节点数少于 5 个，视为噪声移除
        if len(comp) < 5: 
            G.remove_nodes_from(comp)
            
    # --- 2. 尝试移除小环 (Cycle Pruning) ---
    # 骨架提取有时会产生微小的闭环，这对于拓扑导航是不必要的
    try:
        cycles = nx.cycle_basis(G)
        for cycle in cycles:
            if len(cycle) < max_cycle_len:
                # 简单断开环的一条边，破坏闭环结构
                u, v = cycle[0], cycle[1]
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
    except nx.NetworkXNoCycle:
        pass # 如果没有环，直接跳过

    # --- 3. 迭代修剪短毛刺 (Iterative Pruning) ---
    # 毛刺是指只有一个连接点的短路径。
    # 我们从端点 (Degree=1) 开始向内遍历，如果路径长度小于阈值就碰到了分叉点，说明是毛刺。
    while True:
        # 找到所有端点 (度为1的节点)
        endpoints = [n for n, d in G.degree() if d == 1]
        nodes_to_remove = []
        
        for ep in endpoints:
            path = [ep]
            curr = ep
            length = 0
            is_short = False
            
            # 向内搜索，直到遇到交叉点 (度 > 2) 或路径过长
            while True:
                neighbors = list(G.neighbors(curr))
                next_node = None
                
                # 找下一个未访问节点 (即向内延伸)
                for n in neighbors:
                    if n not in path:
                        next_node = n
                        break
                
                if next_node is None:
                    break # 走到尽头了
                
                # 累加路径长度
                weight = G.edges[curr, next_node].get('weight', 1)
                length += weight
                path.append(next_node)
                curr = next_node
                
                # 检查当前节点度数
                if G.degree(curr) > 2:
                    # 遇到交叉点
                    if length < prune_len:
                        is_short = True # 长度不足，标记为短毛刺
                    break
                
                # 如果还没遇到交叉点但长度已经超标，就不剪了，保留这条长路径
                if length >= prune_len:
                    break
            
            if is_short:
                # 记录要删除的节点 (不包括交叉点本身，因为它是主干的一部分)
                nodes_to_remove.extend(path[:-1])
        
        # 如果这一轮没有发现要删除的节点，说明修剪完毕，退出循环
        if not nodes_to_remove:
            break 
            
        G.remove_nodes_from(set(nodes_to_remove))
        
    return G


# 🔥 [核心算法] 安全路径 BFS (test.py 原版逻辑) 🔥
# 此函数用于在二值地图上寻找两点之间的“安全通道”。
# 它不是 A*，而是基于像素连通性的广度优先搜索 (BFS)。
# 它的目的是确保房间中心连接到骨架图时，连线不会穿过墙壁。
def get_safe_path(binary_map, start, end):
    """
    寻找两点间不穿墙的最短路径 (Pixel-wise BFS)。
    
    参数:
        binary_map: 二值地图
        start: 起点 (x, y)
        end: 终点 (x, y)
        
    返回:
        list: 路径点列表 [(x, y), ...]，如果不可达返回 None。
    """
    x1, y1 = start
    x2, y2 = end
    
    # --- 1. 区域裁剪 ---
    # 为了提高搜索效率，我们只在起点和终点的包围盒 (Bounding Box) 
    # 加上一定的 padding 范围内进行搜索。
    pad = 30 
    min_x, max_x = min(x1, x2) - pad, max(x1, x2) + pad
    min_y, max_y = min(y1, y2) - pad, max(y1, y2) + pad
    
    h, w = binary_map.shape
    min_x = max(0, min_x); max_x = min(w, max_x)
    min_y = max(0, min_y); max_y = min(h, max_y)
    
    # 提取子地图 (Sub-map)
    sub_map = binary_map[min_y:max_y, min_x:max_x]
    
    # 将全局坐标转换为局部坐标 (相对于子地图左上角)
    local_start = (x1 - min_x, y1 - min_y)
    local_end = (x2 - min_x, y2 - min_y)
    
    # --- 2. 有效性检查 ---
    # 检查起点终点是否在障碍物内 (必须是 Free Space, 值为 255)
    # 注意 binary_map 索引是 [y, x]
    if sub_map[local_start[1], local_start[0]] == 0: return None
    if sub_map[local_end[1], local_end[0]] == 0: return None
        
    # --- 3. BFS 搜索 ---
    # 队列存储: (current_node, path_so_far)
    # 这里的 path_so_far 记录了从起点到当前的完整路径
    q = collections.deque([(local_start, [])])
    visited = set([local_start])
    found_path = None
    
    while q:
        curr, path = q.popleft()
        
        # 到达终点
        if curr == local_end:
            found_path = path + [curr]
            break
        
        cx, cy = curr
        
        # 8-邻域搜索
        neighbors = [
            (cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1),
            (cx+1, cy+1), (cx-1, cy-1), (cx-1, cy+1), (cx+1, cy-1)
        ]
        
        # 启发式优化：优先搜索靠近终点的方向 (简单的 Euclidean 排序)
        # 这不是 A*，但可以稍微加速 BFS 找到目标的方向，减少无谓的扩散
        neighbors = sorted(neighbors, key=lambda k: ((k[0]-local_end[0])**2 + (k[1]-local_end[1])**2))
        
        for nx_coord, ny_coord in neighbors:
            # 边界检查 (在子地图范围内)
            if 0 <= nx_coord < (max_x - min_x) and 0 <= ny_coord < (max_y - min_y):
                # 只有非障碍物点 ( > 0) 才能走
                if sub_map[ny_coord, nx_coord] > 0:
                    if (nx_coord, ny_coord) not in visited:
                        visited.add((nx_coord, ny_coord))
                        q.append(((nx_coord, ny_coord), path + [curr]))
                        
    # --- 4. 结果转换 ---
    if found_path:
        # 将局部坐标转回全局坐标
        return [(px + min_x, py + min_y) for px, py in found_path]
        
    return None


def find_nearest_node(graph, target_point):
    """
    在图 G 中找到距离 target_point 最近的节点。
    用于将房间中心吸附到骨架网络上。
    
    参数:
        graph: NetworkX 图
        target_point: 目标点 (x, y)
        
    返回:
        tuple: 最近节点的坐标 (x, y)
    """
    nodes = list(graph.nodes())
    if not nodes:
        return None
    
    nodes_arr = np.array(nodes)
    target_arr = np.array(target_point)
    
    # 计算欧氏距离
    dists = np.linalg.norm(nodes_arr - target_arr, axis=1)
    min_idx = np.argmin(dists)
    
    return tuple(nodes[min_idx])


# 🔥 [核心算法] 最终拓扑图构建 (test.py 原版逻辑) 🔥
def build_final_topology(G, room_centers, binary_map):
    """
    结合骨架图和房间中心，生成最终的拓扑结构。
    包含房间节点 (Room) 和 关键路口 (Connector)。
    
    流程:
    1. 遍历每个房间中心点。
    2. 找到骨架图上距离它最近的节点。
    3. 使用 get_safe_path (BFS) 尝试连接房间中心和骨架节点。
    4. 如果连接成功，将路径加入图中。
    5. 再次剪枝，移除那些没有连接到任何房间的死胡同分支。
    6. 提取路口 (Junctions) 并聚类，作为 Connector 节点。
    """
    final_graph = G.copy()
    valid_rooms = []
    
    # --- 1. 将房间中心接入骨架网络 ---
    for rc in room_centers:
        nearest_node = find_nearest_node(G, rc)
        if nearest_node:
            # 使用 BFS 寻找安全连接路径
            safe_path = get_safe_path(binary_map, rc, nearest_node)
            if safe_path:
                prev_node = safe_path[0]
                valid_rooms.append(prev_node) # 记录实际接入点
                
                # 将安全路径添加到图中
                for i in range(1, len(safe_path)):
                    curr_node = safe_path[i]
                    dist = np.linalg.norm(np.array(prev_node) - np.array(curr_node))
                    final_graph.add_edge(prev_node, curr_node, weight=dist)
                    prev_node = curr_node
    
    # --- 2. 再次剪枝 (Protect Phase) ---
    # 保护连接到房间的路径，修剪掉其他无用的分支
    protect_set = set(valid_rooms)
    
    while True:
        endpoints = [n for n, d in final_graph.degree() if d == 1]
        nodes_to_remove = []
        for ep in endpoints:
            if ep in protect_set:
                continue # 受保护的节点不能删
            nodes_to_remove.append(ep)
            
        if not nodes_to_remove:
            break
        final_graph.remove_nodes_from(nodes_to_remove)
    
    # --- 3. 提取输出节点列表 (Output Nodes) ---
    output_nodes = []
    
    # 添加房间节点
    for r in valid_rooms:
        # 确保节点还在图中 (可能被误删，需校验)
        target = r if final_graph.has_node(r) else find_nearest_node(final_graph, r)
        if target:
            output_nodes.append({'xy': target, 'type': 'Room'})
            
    # --- 4. 提取连接点 (Junctions) 并聚类 ---
    # 度数 > 2 的点通常是路口
    junctions = [n for n, d in final_graph.degree() if d > 2]
    
    if junctions:
        j_arr = np.array(junctions)
        # 使用 DBSCAN 聚类，防止一个十字路口生成多个相邻节点
        clustering = DBSCAN(eps=15, min_samples=1).fit(j_arr)
        
        unique_labels = set(clustering.labels_)
        for label in unique_labels:
            cluster = j_arr[clustering.labels_ == label]
            center = cluster.mean(axis=0)
            
            # 选聚类中心最近的实际节点作为代表
            rep_node = tuple(cluster[np.argmin(np.linalg.norm(cluster - center, axis=1))])
            
            if not final_graph.has_node(rep_node):
                rep_node = find_nearest_node(final_graph, rep_node)
            
            # 过滤掉离房间太近的连接点 (避免重叠)
            is_far = True
            for r_node in output_nodes:
                if r_node['type'] == 'Room':
                    if np.linalg.norm(np.array(r_node['xy']) - np.array(rep_node)) < 15:
                        is_far = False
                        break
            
            if is_far:
                output_nodes.append({'xy': rep_node, 'type': 'Connector'})
            
    return output_nodes, final_graph


# 🔥 [核心算法] 拓扑关系提取 (test.py 原版逻辑 - 复杂版) 🔥
def extract_topology_relationships(G, key_nodes):
    """
    基于图的遍历算法，计算节点间的连接关系。
    包含“领地膨胀”逻辑，确保节点归属权正确。
    
    参数:
        G: 最终的 NetworkX 图
        key_nodes: 关键节点列表 (Room/Connector)
        
    返回:
        tuple: (topology_data_dict, sorted_nodes_list)
    """
    # 按 Y 坐标排序，保证 ID 分配顺序，符合阅读习惯 (从上到下)
    sorted_nodes = sorted(key_nodes, key=lambda n: (n['xy'][1], n['xy'][0]))
    
    pixel_to_id = {}
    INFLATION_STEPS = 3 # 膨胀步数
    
    # === 阶段 1: 节点领地膨胀 (Voronoi-like Region Growing) ===
    # 让每个关键节点“占领”周围的骨架点，分配 ID
    # 这样当 BFS 走到这些区域时，就能知道这是属于哪个节点的领地
    for i, node in enumerate(sorted_nodes):
        center_pixel = node['xy']
        if not G.has_node(center_pixel):
            center_pixel = find_nearest_node(G, center_pixel)
            
        q = collections.deque([(center_pixel, 0)])
        visited_local = set([center_pixel])
        
        while q:
            curr, dist = q.popleft()
            
            # 标记 ID
            if curr not in pixel_to_id:
                pixel_to_id[curr] = i
                
            if dist >= INFLATION_STEPS:
                continue
                
            for n in G.neighbors(curr):
                if n not in visited_local:
                    visited_local.add(n)
                    q.append((n, dist + 1))
                    
    # === 阶段 2: 邻居搜索 (Neighbor Search) ===
    topology_data = {}
    
    for i, node in enumerate(sorted_nodes):
        start_pos = node['xy']
        if not G.has_node(start_pos):
            start_pos = find_nearest_node(G, start_pos)
            
        neighbors_found = []
        found_ids = set()
        
        # BFS 搜索其他节点的领地
        q = collections.deque([(start_pos, 0)])
        visited = set([start_pos])
        
        while q:
            curr, dist = q.popleft()
            
            # 核心判断：如果碰到了别的节点的领地
            if curr in pixel_to_id:
                target_id = pixel_to_id[curr]
                if target_id != i:
                    # 发现新邻居
                    if target_id not in found_ids:
                        neighbors_found.append({'id': target_id, 'dist': dist})
                        found_ids.add(target_id)
                    continue # 碰到边界就停止该分支搜索 (阻断逻辑)
            
            # 限制搜索深度，防止爆栈或跑太远
            if dist > 800: 
                continue
                
            for n in G.neighbors(curr):
                if n not in visited:
                    visited.add(n)
                    # 累加边的权重 (距离)
                    weight = G.edges[curr, n].get('weight', 1)
                    q.append((n, dist + weight))
                    
        topology_data[i] = {
            "type": node['type'], 
            "pos": start_pos, 
            "neighbors": neighbors_found
        }
        
    return topology_data, sorted_nodes # 注意这里返回 sorted_nodes 供外部使用


# ==============================================================================
# 🧩 模块 3: 综合导航主脑 (Integrated Smart Navigator)
# 来源: smart_navigator_v2_copy.py (核心控制逻辑)
# 来源: node_traversal_navigator.py (状态机流转)
# ==============================================================================

def normalize_angle(angle):
    """ 工具函数：角度归一化 """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

class TopoNavNode(Node):
    """
    终极导航节点类 (The Ultimate Navigation Node)。
    集成了探索、建图、巡逻、VLM 触发等所有功能。
    
    Attributes:
        current_state (int): 当前系统状态 (EXPLORING, GENERATING, PATROLLING, WAITING)
        planner (AStarPlanner): 路径规划器实例
        mission_nodes (list): 巡逻任务节点列表
    """
    
    def __init__(self):
        super().__init__('topo_nav_node')

        # Load centralised config
        self._cfg = load_config()
        _BASE_DIR = self._cfg['data']['base_dir']
        os.makedirs(_BASE_DIR, exist_ok=True)

        
        # === 状态机定义 ===
        self.STATE_EXPLORING = 0        # 探索阶段 (SLAM): 自主寻找未知区域
        self.STATE_GENERATING_TOPO = 1  # 拓扑生成阶段: 分析地图结构
        self.STATE_PATROLLING = 2       # 巡逻阶段: 遍历房间节点
        self.STATE_WAITING_FOR_VLM = 3  # VLM 交互阶段: 拍照并等待分析结果
        self.STATE_VL_SEARCH = 4        # [新增] 到点后旋转检索目标 (YOLO/VL)
        self.STATE_GO_TO_VL_GOAL = 5    # [新增] 前往 VL 返回的局部目标点
        self.STATE_VL_VERIFY = 6        # [新增] 近距验证：到达疑似目标点后原地旋转确认
        self.STATE_FINAL_SEARCH_WAIT = 7  # [新增] 等待 VLM 语义拓扑检索结果 (/audit_nav/result)
        
        self.current_state = self.STATE_EXPLORING
        self.action_description = "Initializing..."

        # === Mission Phases (高层流程控制；topic/接口保持完全兼容) ===
        self.PHASE_WAIT_TARGET = 0
        self.PHASE_EXPLORE_GUIDED = 1
        self.PHASE_BUILD_TOPO = 2
        self.PHASE_PATROL_BUILD_MEMORY = 3
        self.PHASE_FINAL_SEARCH_WAIT = 4
        self.PHASE_FINAL_SEARCH_ACTIVE = 5
        self.PHASE_DONE = 6

        # 初始：必须先收到 /audit_nav/instruction 才开始探索
        self.mission_phase = self.PHASE_WAIT_TARGET
        self.primary_instruction = ""
        self.soft_found = False   # 探索阶段到旁边即算找到一次（但仍继续探索建图）
        self.final_found = False  # 最终检索阶段找到后结束任务

        # Final search request bookkeeping
        self.waiting_for_vlm_search = False
        self.final_search_request_time = 0.0
        self.final_search_last_issue_time = 0.0
        self.final_search_reissue_count = 0
        self.final_search_request_timeout = get(self._cfg, "navigator", "final_search_request_timeout", default=90.0)  # 秒：等待 /audit_nav/result 超时后会重发 instruction

        # === [新增] 视觉识别目标（来自 commander）与 VL 目标点缓存 ===
        # === + 语义探索引导（soft semantic hints for frontier rerank） ===
        self.search_instruction = ""   # commander 发布的查找目标（例如 'sofa'）
        # [优化] 查找任务的失败反馈：用于告诉 VLM“哪些节点已尝试但未找到”，让其重新决策下一个更可能的节点
        self.search_failed_node_ids = set()
        self.search_last_instruction = ""
        # 本轮 VL_SEARCH/GO_TO_VL_GOAL 的起始拓扑节点（通常是当前巡逻节点）
        self.vl_search_origin_node_id = None
        self.vl_goal_rel = None         # PointStamped in base_link frame: (x, y)
        self.vl_goal_time = 0.0         # 最近一次收到 VL 目标点的时间（time.time()）
        self.vl_goal_consumed_time = 0.0
        # [VL-Nav] Deferred VL goal (world frame) to avoid interrupting node-audit travel.
        # When a high-confidence VL goal arrives while we are still en-route to a VLM-selected node,
        # we can cache it and consume it right after the node audit finishes.
        self.pending_vl_goal_world = None   # (gx, gy) in map/world frame
        self.pending_vl_goal_time = 0.0     # time.time() when cached
        self.pending_vl_goal_label = ""     # the search_instruction associated with this goal
        self.pending_vl_goal_consumed_time = 0.0  # consumed timestamp for pending goal
        self.pending_vl_goal_max_age_sec = 240.0   # drop stale deferred goals (seconds)
# 🔥 [新增] 候选池与大模型方向神谕
        self.deferred_vl_candidates = []  
        self.vlm_authorized_view = ""
        # [VL-Nav] VL goal reachability helpers
        self.vl_goal_snap_max_m = get(self._cfg, "navigator", "vl_goal_snap_max_m", default=1.2)       # try snapping within this radius (meters)
        self.vl_goal_standoff_radii_m = (0.6, 0.8, 1.0)  # candidate approach radii (meters)
        self.vl_preempt_near_node_m = 0.7   # within this distance to node, allow preempt (meters)
  # 防止重复抢占：已处理的 VL goal 时间戳
        self.pending_vl_search = False  # 本次到点后是否需要执行 VL 搜索
        self.advance_after_vl_search = False  # VL 搜索结束后是否推进到下一个巡逻点
        self.vl_search_start_time = 0.0
        self.vl_search_timeout = 30.0   # 将在参数区初始化后赋值（复用 STUCK_TIME_THRESHOLD）

        
        # === 路径规划器 ===
        # 允许 allow_unknown=True 是因为巡逻时可能需要穿过一些未完全扫描的边缘区域
        self.planner = AStarPlanner(
            resolution=get(self._cfg, "planner", "resolution", default=0.05),
            robot_radius=get(self._cfg, "planner", "robot_radius", default=0.15),
            allow_unknown=get(self._cfg, "planner", "allow_unknown", default=True)
        )
        self.global_frame = "world"
        self.base_frame = "base_link"
        
        # === 🔥 参数设置 (1:1 还原 Smart-V2) 🔥 ===
        # 基础速度控制
        self.BASE_SPEED = get(self._cfg, "navigator", "base_speed", default=0.60)
        self.MIN_SPEED_FLOOR = get(self._cfg, "navigator", "min_speed_floor", default=0.35)  
        self.RUSH_SPEED = get(self._cfg, "navigator", "rush_speed", default=0.65)
        self.MAX_SPEED_CAP = get(self._cfg, "navigator", "max_speed_cap", default=0.85)
        
        # PID 参数
        self.kp_speed = 1.0
        self.ki_speed = 0.8
        self.kd_speed = 0.1
        self.kp_yaw = 50.0
        self.kd_yaw = 15.0
        self.MAX_ROTATION_SPEED = 2.5
        
        # 动态前瞻 (Dynamic Lookahead)
        # 速度越快，看越远
        self.MIN_LOOKAHEAD = 0.4
        self.MAX_LOOKAHEAD = 0.8
        self.current_lookahead_val = 0.5 
        
        # 到达与脱困阈值
        # ⚠️ 可根据需要调整 ARRIVAL_DIST (单位：米)
        self.ARRIVAL_DIST = get(self._cfg, "navigator", "arrival_dist", default=0.30)
        self.STUCK_VEL_THRESHOLD = get(self._cfg, "navigator", "stuck_vel_threshold", default=0.08)  # 速度低于此值认为卡住
        self.STUCK_TIME_THRESHOLD = get(self._cfg, "navigator", "stuck_time_threshold", default=1.5)  # 持续卡住多久触发恢复
        self.vl_search_timeout = get(self._cfg, "navigator", "vl_search_timeout", default=30.0) # [新增] 旋转搜索最大时长（复用既有阈值）
        self.FORCE_RUSH_SPEED = 5.0      # 暴力脱困速度
        self.BLACKLIST_RADIUS = get(self._cfg, "navigator", "blacklist_radius", default=0.30) 
        self.MAX_BLACKLIST_SIZE = 50
        
        # 配置文件路径
        self.CALIB_FILE = get(self._cfg, "data", "calib_file", default="map_calibration.json")
        
        # === 运行时变量 (Runtime Variables) ===
        self.local_map_data = None
        self.map_info = None
        self.map_frame_id = "world"  # 与 OccupancyGrid.header.frame_id 对齐
        self._global_map_received = False  # /map 优先标记
        self.target_point = None
        self.current_path = []
        self.last_path_index = 0
        self.current_linear_velocity = 0.0
        
        # PID 状态变量
        self.speed_integral = 0.0
        self.last_speed_error = 0.0
        self.last_alpha = 0.0
        self.last_target_speed = 0.0 # [重要] 恢复此变量用于卡死检测
        
        # 脱困状态变量 (Recovery States)
        self.stuck_timer = 0.0 
        # [新增] 过坎/撞墙判定计时器（基于局部地图前方 180°、半径 0.3m + 低速持续时间）
        self.threshold_timer = 0.0  # 前方无障碍但低速持续 -> 认为在过坎/门槛
        self.wall_collision_timer = 0.0  # 前方有障碍但低速持续 -> 认为撞墙/顶住障碍
        self.rush_mode_until = 0.0  # 冲刺模式持续到的绝对时间戳

        # [新增] 撞墙后小幅后退再重规划
        self.wall_backing_active = False
        self.wall_back_start_xy = (0.0, 0.0)
        # 记录后退开始时间，用于“最多后退 1s”的防卡死保护（狭窄缝隙无法走满 0.3m 时也能退出）
        self.wall_back_start_time = 0.0
        self.wall_back_distance = 0.2  # meters
        self.wall_back_max_time = 1.0  # seconds (按用户要求：后退最多 1s 就停止并进入重规划)

        # [新增] 门槛脱困：先小幅后退(带时间上限)再进入冲刺模式（复用 wall_back_distance / wall_back_max_time）
        self.threshold_backing_active = False
        self.threshold_back_start_xy = (0.0, 0.0)
        self.threshold_back_start_time = 0.0
        self.force_recovery_state = "NONE" 
        self.force_recovery_start_time = 0.0
        
        # 冲刺检测变量
        self.last_pitch = 0.0
        self.is_in_rush_mode = False # [重要] 恢复冲刺状态位
        
        # 逻辑控制变量
        self.check_validity_timer = 0.0 # 动态剪枝计时器
        self.last_control_time = time.time()
        self.unreachable_points = []
        self.mission_nodes = []
        self.load_existing_topology()
        # [VL-Nav] Instance candidates (from open-vocabulary detections)
        # - Used as high-priority exploration goals (IBTP idea) without touching low-level control.
        # - Each item: {'pt': (x, y), 't': unix_time, 'source': str}
        self.instance_candidates = deque(maxlen=200)
        # [新增] Semantic hints（低/中置信度软线索）：用于探索阶段对 frontiers 做语义加权
        self.semantic_hints = deque(maxlen=300)  # [{'pt':(x,y), 't':unix, 'conf':float, 'source':str}]
        self.semantic_hint_max_age_sec = 120.0
        self.semantic_hint_sigma_m = 2.5
        self.semantic_hint_min_conf = 0.30

        self.current_node_idx = 0
        
        # [重要] 探索完成确认去抖动变量 (Exploration Debounce)
        self.target_start_time = 0.0 
        self.finish_timer = 0.0
        

        # [新增] 探索结束判定稳定性：未知区域稳定计时
        self.last_unknown_cells = None
        self.unknown_stable_timer = 0.0

        # [新增] 探索无进展(frontier stagnation)判定：避免在不可消除前沿/局部地图边界上无限循环
        self.explore_target_unknown0 = None
        self.explore_target_t0 = 0.0
        self.explore_progress_fail_count = 0

        # [新增] 脱困逻辑冷却：用于防止 VLM 等待后误触发“门槛/卡死”后退+冲刺
        self.recovery_disable_until = 0.0
        # 🔥 [关键新增] 碰撞去抖动计数器 (Collision Debounce)
        self.collision_persistence = 0 
        
        # === ROS 通讯接口 ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.pub_vel = self.create_publisher(Twist, get(self._cfg, "topics", "cmd_vel", default="/cmd_vel"), 10)
        self.pub_vlm_trigger = self.create_publisher(Int32, get(self._cfg, "topics", "vlm_trigger", default="/vlm/trigger_audit"), 10)
        
        self.pub_vl_feedback = self.create_publisher(String, get(self._cfg, "topics", "feedback", default="/audit_nav/feedback"), 10)
        # [新增] 本节点内部可重发 instruction（同 topic，接口不变）
        self.pub_instruction = self.create_publisher(String, get(self._cfg, "topics", "instruction", default="/audit_nav/instruction"), 10)
        # [Eval] publish evaluation events (collision/reach/done) without changing existing interfaces
        self.pub_eval_event = self.create_publisher(String, get(self._cfg, "topics", "eval_event", default="/audit_nav/eval_event"), 10)
        self.eval_collision_count = 0
        self._eval_last_collision_time = 0.0
        self._eval_collision_debounce_sec = 0.8
        self._eval_soft_found_pub = False
        self._eval_final_found_pub = False
        self._eval_done_pub = False
        # 地图：优先使用 /map(全局)；若系统没有 /map，则回退到 /local_map
        self.create_subscription(OccupancyGrid, get(self._cfg, "topics", "map", default="/map"), self.map_cb_global, 10)
        self.create_subscription(OccupancyGrid, get(self._cfg, "topics", "local_map", default="/local_map"), self.map_cb_local, 10)
        self.create_subscription(Odometry, get(self._cfg, "topics", "odom", default="/odom"), self.odom_cb, 10)
        self.create_subscription(String, get(self._cfg, "topics", "vlm_complete", default="/vlm/audit_complete"), self.vlm_complete_cb, 10)
        
        # [新增] 接收 VLM 物体搜索结果 (Find 'xxx')
        self.create_subscription(String, get(self._cfg, "topics", "result", default="/audit_nav/result"), self.vlm_search_result_cb, 10)
        # [新增] commander 下发的查找目标（同时也会被 vl_perception_v2 使用）
        self.create_subscription(String, get(self._cfg, "topics", "instruction", default="/audit_nav/instruction"), self.instruction_cb, 10)
        # [新增] 接收 vl_perception_v2 输出的目标点（base_link 坐标系）
        self.create_subscription(PointStamped, get(self._cfg, "topics", "current_goal", default="/audit_nav/current_goal"), self.vl_goal_cb, 10)
        # [新增] 接收 VL 节点发布的目标置信度（用于过滤低置信度误检，避免打断探索）
        self.vl_best_conf = 0.0
        self.vl_conf_time = 0.0
        self.vl_preempt_min_conf = get(self._cfg, "navigator", "vl_preempt_min_conf", default=0.50)  # 低于该值仅缓存线索，不抢占导航（可按数据分布微调）
        self.create_subscription(Float32, get(self._cfg, "topics", "confidence", default="/audit_nav/confidence"), self.vl_conf_cb, 10)

        # [新增] 订阅 VLM 输出的目标 bbox（JSON String）
        self.create_subscription(String, get(self._cfg, "topics", "object_bbox", default="/audit_nav/object_bbox"), self.vl_bbox_cb, 10)
        # === [新增] Belief/迟滞阈值 + 近距验证 + 失败冷却（不改底层控制参数）===
        # 说明：
        # - vl_preempt_min_conf 作为“打开阈值”(on)，避免新增外部参数；
        # - 关闭阈值(off)自动比 on 低一截，形成迟滞，避免“近距离掉分”导致反复抢占/退出；
        # - 到达疑似目标点后进入 STATE_VL_VERIFY 原地旋转 1s 近距验证；
        # - 验证失败会对该位置做冷却，探索继续，不会卡死在低置信度误检点。
        self.vl_th_on = float(self.vl_preempt_min_conf)
        self.vl_th_off = max(0.0, self.vl_th_on - 0.12)
        self.vl_conf_fresh_sec = 0.60
        self.vl_on_count = 0
        self.vl_off_count = 0
        self.vl_k_on = 3    # 连续 3 帧(或消息)高于阈值才触发抢占
        self.vl_k_off = 4   # 连续 4 帧低于 off 认为“持续低”
        self.vl_belief = 0.0
        self.vl_belief_alpha = 0.20

        # 是否启用“近距验证”(到达 VL 目标点后原地旋转确认)。
        # 用户需求：到达目标区域后直接判定成功并打印 success，不做近距验证。
        self.enable_vl_verify = False

        # 近距验证参数（内部常量，不改底层控制）
        self.vl_verify_duration = 1.0  # seconds
        self.vl_verify_start_time = 0.0
        self.vl_verify_max_conf = 0.0
        self.vl_verify_goal = None
        self.vl_verify_return_state = None
        self.vl_verify_instruction = ""

        # 失败冷却：避免误检点反复被抢占（使用 BLACKLIST_RADIUS 作为空间合并半径）

        self.vl_cooldowns = deque(maxlen=100)  # [{'pt':(x,y), 'until':t, 'reason':str}]
        self.vl_fail_cooldown_sec = get(self._cfg, "navigator", "vl_fail_cooldown_sec", default=60.0)
        self.vl_ambig_cooldown_sec = get(self._cfg, "navigator", "vl_ambig_cooldown_sec", default=20.0)
        # [新增] VL 抢占前往目标后，回到原状态（探索/巡逻）
        self.vl_return_state = None
        # 初始化可视化窗口
        cv2.namedWindow("AuditNav Dashboard", cv2.WINDOW_NORMAL)
        # ================== 🔥 新增：视频录制器初始化 ==================
        # Save to data directory with timestamp to avoid overwriting
        video_filename = f"nav_dashboard_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        self.video_path = os.path.join(self._cfg["data"]["base_dir"], video_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # ⚠️ 注意尺寸：代码中 panel 高110，body高600 (共710)，总宽 600+600=1200
        # 控制循环的定时器是 0.1s (10Hz)，所以录像帧率设为 10.0
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 10.0, (1200, 710))
        self.get_logger().info(f"🎥 Dashboard recording started. Saving to: {self.video_path}")
        # ===============================================================
        # 启动主循环 (10Hz)
        # [FIX] 在极少数合并/缩进异常情况下，control_loop 可能被意外挪到类外，导致 AttributeError。
        # 这里用 getattr + 全局回退保证不会在 __init__ 直接崩溃，同时不改变正常情况下的行为。
        cb = getattr(self, 'control_loop', None)
        if cb is None:
            cb_fn = globals().get('control_loop', None)
            if callable(cb_fn):
                cb = (lambda: cb_fn(self))
            else:
                raise AttributeError('IntegratedVLMNavigator has no control_loop (check indentation/merge)')
        self.timer = self.create_timer(0.1, cb)
        
        self.get_logger().info("🚀 Integrated Navigator: FINAL ULTIMATE MERGE (Gray Fixed + Toilet Fix + Path Check)")

    # ================= 消息回调 (Callbacks) =================
    def load_existing_topology(self):
        """ Load existing topology node coordinates from local file (for checkpoint resume / ablation). """
        import glob
        import os
        try:
            search_pattern = os.path.join(self._cfg["data"]["base_dir"], "final_node_coordinates_*.json")
            files = glob.glob(search_pattern)
            if not files:
                self.get_logger().info("⚠️ No local topology memory found. Starting from scratch.")
                return False
                
            latest_file = max(files, key=os.path.getctime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.mission_nodes = []
            for item in data:
                if item.get('type') == 'Room':
                    self.mission_nodes.append({
                        'id': int(item['id']),
                        'coords': (float(item['world_coords'][0]), float(item['world_coords'][1]))
                    })
                    
            if self.mission_nodes:
                self.get_logger().info(f"💾 [Memory Loaded] Restored {len(self.mission_nodes)} topology nodes from {os.path.basename(latest_file)}.")
                return True
        except Exception as e:
            self.get_logger().error(f"❌ Failed to load topology memory: {e}")
            
        return False
    def map_cb_global(self, msg):
        """优先使用全局地图 /map 作为探索与拓扑生成依据。"""
        self._global_map_received = True
        self.map_cb(msg)

    def map_cb_local(self, msg):
        """若全局地图尚未到来，则用 /local_map 作为兜底。"""
        if not getattr(self, "_global_map_received", False):
            self.map_cb(msg)

    def map_cb(self, msg):
        """ 
        接收栅格地图消息。
        将一维数据 (int8) 转换为二维 numpy 数组 (height, width)。
        """
        try:
            self.local_map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
            self.map_info = msg.info
            # 记录地图坐标系，后续 TF 查询与 world/grid 转换都以该坐标系为准
            try:
                if getattr(msg, 'header', None) is not None:
                    fid = getattr(msg.header, 'frame_id', '')
                    if fid:
                        self.map_frame_id = fid
            except Exception:
                pass
        except Exception as e:
            self.get_logger().error(f"Map processing error: {e}")

    def odom_cb(self, msg):
        """ 
        接收里程计消息。
        计算当前实际线速度标量，用于 PID 反馈和卡死检测。
        """
        self.current_linear_velocity = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def instruction_cb(self, msg: String):
        """接收 commander 下发的查找目标。"""
        new_inst = (msg.data or "").strip()
        if not new_inst:
            # 允许外部清空指令：仅暂停本地搜索，不重置 primary
            self.search_instruction = ""
            return

        is_new_task = (str(getattr(self, 'primary_instruction', '')) != new_inst)

        if is_new_task:
            # 新任务：彻底重置高层流程
            self.primary_instruction = new_inst
            self.search_instruction = new_inst

            # 清空失败历史与线索池
            self.search_failed_node_ids = set()
            self.search_last_instruction = new_inst
            self.vl_search_origin_node_id = None
            try:
                self.instance_candidates.clear()
                self.semantic_hints.clear()
            except Exception:
                pass

            # ==============================
            # 🔥 核心修复 1：重置事件发布锁 🔥
            # 否则第二局评测脚本永远收不到到达和结束事件！
            self._eval_soft_found_pub = False
            self._eval_final_found_pub = False
            self._eval_done_pub = False
            # ==============================

            # 重置阶段标记
            self.soft_found = False
            self.final_found = False
            self.waiting_for_vlm_search = False
            self.final_search_request_time = 0.0
            self.final_search_last_issue_time = 0.0
            self.final_search_reissue_count = 0

            # ==============================
            # 🔥 核心优化 2：记忆继承与阶段跃迁 🔥
            # ==============================
            # 如果已经存在建好的拓扑节点，直接跳过探索和巡逻，进入 VLM 最终检索！
            # （注意：这里去掉了旧代码中清空 self.mission_nodes 的逻辑）
            if hasattr(self, 'mission_nodes') and len(self.mission_nodes) > 0:
                self.get_logger().info("🧠 Memory Topology found! Skipping exploration, going directly to VLM Search.")
                self.mission_phase = self.PHASE_FINAL_SEARCH_WAIT
                self.current_state = self.STATE_FINAL_SEARCH_WAIT
                self._mission_start_final_search(now=time.time(), reason='memory_guided_shortcut')
            else:
                self.mission_phase = self.PHASE_EXPLORE_GUIDED
                self.current_state = self.STATE_EXPLORING
                self.get_logger().info(f"🎯 New target received: [{new_inst}] -> start guided exploration")
            return

        # 同一任务：只在等待目标时触发探索，否则保持当前流程
        self.search_last_instruction = new_inst
        if getattr(self, 'mission_phase', self.PHASE_WAIT_TARGET) == self.PHASE_WAIT_TARGET:
            self.primary_instruction = new_inst
            self.search_instruction = new_inst
            self.search_failed_node_ids = set()
            self.vl_search_origin_node_id = None
            self.soft_found = False
            self.final_found = False
            
            # 这里也做一下保底的锁重置
            self._eval_soft_found_pub = False
            self._eval_final_found_pub = False
            self._eval_done_pub = False
            
            self.mission_phase = self.PHASE_EXPLORE_GUIDED
            self.current_state = self.STATE_EXPLORING
            self.get_logger().info(f"🎯 Target received: [{new_inst}] -> start exploration")
        else:
            # 若外部在后续阶段重复发同一指令，视为“继续/刷新”
            self.search_instruction = new_inst

    def vl_conf_cb(self, msg: Float32):
        """接收 VL 节点发布的最高置信度，用于过滤低置信度误检，并维护 belief/迟滞状态。"""
        try:
            self.vl_best_conf = float(msg.data)
            self.vl_conf_time = time.time()
            # 更新 belief + 迟滞计数
            try:
                self._vl_update_belief_hys(self.vl_best_conf, self.vl_conf_time)
            except Exception:
                pass
        except Exception:
            pass



    def vl_bbox_cb(self, msg: String):
        """接收 /audit_nav/object_bbox：打印并缓存 VLM 检测到的 bbox 信息（JSON）"""
        raw = (msg.data or "").strip()
        if not raw:
            return
        try:
            data = json.loads(raw)
        except Exception:
            self.get_logger().warn(f"📦 bbox msg is not valid JSON: {raw[:120]}")
            return

        # ==========================================================
        # 🔥 [Bug Fix 1] 如果 VLM 明确说没找到，立刻清空并退出！
        # ==========================================================
        if not data.get("found", False):
            self.vlm_authorized_view = ""
            self.last_vlm_bbox = None
            self.get_logger().info("VLM confirmed NO target here. Cleared authorized view.")
            return

        # 兼容字段
        q = str(data.get("query", "")).strip()
        node_id = data.get("node_id", None)
        view = str(data.get("view", "")).strip()
        
        # 🔥 [新增] 把 VLM 确认的方向保存下来，供后续打分使用
        self.vlm_authorized_view = view 
        
        bbox = data.get("bbox", None)
        conf = data.get("confidence", None)
        reason = str(data.get("reason", "")).strip()

        try:
            self.get_logger().info(
                f"🎯 VLM bbox: query={q} node={node_id} view={view} bbox={bbox} conf={conf} reason={reason}"
            )
        except Exception:
            pass

        self.last_vlm_bbox = data

# 兼容字段
        q = str(data.get("query", "")).strip()
        node_id = data.get("node_id", None)
        view = str(data.get("view", "")).strip()
        
        # 🔥 [新增] 把 VLM 确认的方向保存下来，供后续打分使用
        self.vlm_authorized_view = view 
        
        bbox = data.get("bbox", None)
        conf = data.get("confidence", None)
        reason = str(data.get("reason", "")).strip()

        try:
            self.get_logger().info(
                f"🎯 VLM bbox: query={q} node={node_id} view={view} bbox={bbox} conf={conf} reason={reason}"
            )
        except Exception:
            pass

        self.last_vlm_bbox = data

    def vl_goal_cb(self, msg: PointStamped):
        """接收 vl_perception_v2 输出的目标点（base_link 坐标系）。

        - 高置信度（>= vl_th_on）：缓存为 instance candidate（IBTP），可用于探索优先验证。
        - 中等置信度（>= vl_th_off）：缓存为 semantic hint（soft evidence），用于探索阶段 frontier 重排序。
        """
        if getattr(self, 'current_state', None) == getattr(self, 'STATE_PATROLLING', 2):
            return
        try:
            # 仅缓存 (x, y)，Z 不参与地面导航
            self.vl_goal_rel = (float(msg.point.x), float(msg.point.y))
            now = time.time()
            self.vl_goal_time = now

            # 仅在“允许搜索”的阶段才消耗该线索（避免建图/巡逻记忆阶段被误检打断）
            phase = getattr(self, 'mission_phase', None)
            allow = phase in (getattr(self, 'PHASE_EXPLORE_GUIDED', -1), getattr(self, 'PHASE_FINAL_SEARCH_ACTIVE', -1))
            if not allow:
                return
            if not getattr(self, 'search_instruction', ''):
                return

            # 置信度必须足够“新鲜”
            fresh_sec = float(getattr(self, 'vl_conf_fresh_sec', 0.6))
            if (now - float(getattr(self, 'vl_conf_time', 0.0))) > fresh_sec:
                return

            conf = float(getattr(self, 'vl_best_conf', 0.0))

            pose = self.get_robot_pose()
            if pose is None:
                return
            rx, ry, ryaw, _ = pose
            x_rel, y_rel = self.vl_goal_rel
            gx = rx + math.cos(ryaw) * x_rel - math.sin(ryaw) * y_rel
            gy = ry + math.sin(ryaw) * x_rel + math.cos(ryaw) * y_rel

            # 冷却点过滤（避免反复尝试已失败的局部点）
            try:
                if self._vl_is_on_cooldown(gx, gy, now):
                    return
            except Exception:
                pass

            th_on = float(getattr(self, 'vl_th_on', getattr(self, 'vl_preempt_min_conf', 0.55)))
            th_off = float(getattr(self, 'vl_th_off', max(0.0, th_on - 0.12)))

            if conf >= th_on:
                # 高置信度：直接进入 instance 候选池
                self.remember_instance_candidate((gx, gy), source='vl')
            elif conf >= th_off and conf >= float(getattr(self, 'semantic_hint_min_conf', 0.30)):
                # 中等置信度：只做 soft hint，用于探索 frontier 重排序
                try:
                    self.remember_semantic_hint((gx, gy), conf=conf, source='vl_soft')
                except Exception:
                    pass
        except Exception:
            # 避免异常打断控制循环
            self.vl_goal_rel = None
            self.vl_goal_time = 0.0


    def vlm_complete_cb(self, msg):

        """ 
        接收 VLM 节点处理完成信号。
        当 VLM 完成拍照和分析后，会发送此消息。
        """
        if self.current_state == self.STATE_WAITING_FOR_VLM:
            # [新增] 若 commander 正在下发查找目标，则在本节点审计后执行一次“原地旋转检索 → 前往目标点”
            if self.pending_vl_search and self.search_instruction:
                self.get_logger().info(
                    f"📨 VLM Audit Finished ({msg.data}). Start VL search for: {self.search_instruction}"
                )
                self.pending_vl_search = False
                self.advance_after_vl_search = True
                self.current_state = self.STATE_VL_SEARCH
                # 记录本轮 VL 搜索的起始拓扑节点（用于失败反馈与重新决策）
                self.vl_search_start_time = time.time()
                self.vl_goal_consumed_time = 0.0 # 强制允许消费最新目标
                try:
                    self.vl_search_origin_node_id = int(self.mission_nodes[self.current_node_idx]['id'])
                except Exception:
                    self.vl_search_origin_node_id = None
                self.vl_search_start_time = time.time()
                self.vl_goal_rel = None
                self.vl_goal_time = 0.0

                # 防止刚切换状态误触发恢复逻辑
                self.stuck_timer = 0.0
                self.last_target_speed = 0.0
                self.force_recovery_state = "NONE"
                self.recovery_disable_until = time.time() + 2.0
                return

            # 否则：继续原逻辑，前往下一个巡逻点
            self.get_logger().info(f"📨 VLM Audit Finished ({msg.data}). Proceeding to next node.")
            self.pending_vl_search = False
            self.advance_after_vl_search = False
            self.current_node_idx += 1
            self.current_state = self.STATE_PATROLLING

            # [新增] 防止下一段路径起步时误触发“卡死/门槛脱困”
            self.stuck_timer = 0.0
            self.last_target_speed = 0.0
            self.force_recovery_state = "NONE"
            self.recovery_disable_until = time.time() + 2.0

    def _extract_json_payload(self, text: str):
        """从 VLM 返回的混合文本中提取 JSON 载荷（支持单个对象 {...} 或数组 [...]). 
        兼容模型在 JSON 前后夹带自然语言的情况。"""
        s = (text or "").strip()
        if not s:
            return None
        dec = json.JSONDecoder()
        # 在文本中寻找第一个 JSON 起始符号，然后尝试 raw_decode
        for i, ch in enumerate(s):
            if ch in "[{":
                try:
                    obj, _end = dec.raw_decode(s[i:])
                    return obj
                except Exception:
                    continue
        return None


    def _publish_vl_feedback(self, failure_reason: str, node_id: int = None, extra: dict = None):
        """向 VLM 决策节点发送“查找失败反馈”，让其在排除已失败节点后重新决策下一个目标节点。"""
        try:
            if not self.search_instruction:
                return
            # 没有显式给 node_id 时，优先使用本轮搜索起始节点
            if node_id is None and self.vl_search_origin_node_id is not None:
                node_id = int(self.vl_search_origin_node_id)

            # 没有关联拓扑节点（例如探索阶段的即时抢占）就不发反馈，避免污染
            if node_id is None:
                return

            try:
                self.search_failed_node_ids.add(int(node_id))
            except Exception:
                pass

            payload = {
                "query": str(self.search_instruction),
                "failure_reason": str(failure_reason),
                "last_failed_node": int(node_id),
                "failed_nodes": sorted([int(x) for x in self.search_failed_node_ids]),
                "time": time.time()
            }
            if isinstance(extra, dict) and extra:
                payload["extra"] = extra

            m = String()
            m.data = json.dumps(payload, ensure_ascii=False)
            self.pub_vl_feedback.publish(m)

            self.get_logger().info(
                f"🧠 Sent VLM feedback for re-decision (reason={failure_reason}, node={node_id}, failed={len(self.search_failed_node_ids)})"
            )
        except Exception as e:
            self.get_logger().warn(f"⚠️ publish_vl_feedback failed: {e}")

    def vlm_search_result_cb(self, msg: String):
        """接收 VLM Search 结果，解析 node_id 并跳转到对应 mission node。"""
        raw = msg.data
        payload = self._extract_json_payload(raw)
        if payload is None:
            self.get_logger().warn(f"⚠️ VLM result has no JSON payload. Raw: {raw[:120]}...")
            return

        try:
            # 兼容：单个对象 {...} 或数组 [...]
            if isinstance(payload, list):
                arr = payload
            elif isinstance(payload, dict):
                arr = [payload]
            else:
                self.get_logger().warn(f"⚠️ VLM JSON payload type unsupported: {type(payload)}")
                return

            if len(arr) == 0:
                self.get_logger().warn("⚠️ VLM result JSON is empty list.")
                return

            best = max(arr, key=lambda x: float(x.get('score', 0.0)))
            target_id = int(best['node_id'])
# ================= 🔪 终极防御：全局拓扑盲搜兜底 (Hard Fallback) =================
            if target_id == -1:
                self.get_logger().error("💀 [HARD FALLBACK] VLM exhausted candidates. Degrading to Blind Topological Patrol!")
                
                if not getattr(self, 'mission_nodes', []):
                    self.get_logger().error("No topological nodes available for patrol. Stopping.")
                    return
                    
                # 策略：为了让大模型猜错的代价（路径长度）最大化，我们故意挑一个距离当前位置【最远】的节点作为盲搜目标
                furthest_idx = 0
                max_dist = -1.0
                pose = self.get_robot_pose()
                if pose is None:
                    self.get_logger().error("❌ [HARD FALLBACK] Cannot get robot pose. Aborting blind search.")
                    return
                rx, ry = pose[0], pose[1]
                
                for idx, node in enumerate(self.mission_nodes):
                    # 🔥 核心修复 2：兼容字典中的 'coords' 或 'position' 键名
                    if 'coords' in node:
                        nx, ny = node['coords'][0], node['coords'][1]
                    elif 'position' in node:
                        nx, ny = node['position'][0], node['position'][1]
                    else:
                        continue # 如果没找到坐标键，跳过该节点
                        
                    dist = math.hypot(nx - rx, ny - ry)
                    if dist > max_dist:
                        max_dist = dist
                        furthest_idx = idx
                        
                furthest_node = self.mission_nodes[furthest_idx]
                self.get_logger().info(f"🚀 [PATROL] Dispatching to furthest node {furthest_node['id']} for blind search.")
                
                # 🔥 终极修复：像正常节点一样，把状态机切回 PATROLLING，让 control_loop 自动调用 A* 规划！
                self.current_node_idx = furthest_idx
                self.target_point = None
                self.current_path = [] 
                self.last_path_index = 0
                self.current_state = self.STATE_PATROLLING
                
                # 保持整体 Phase 为寻找目标，这样 YOLO 的检测依然开着，随时准备抢占！
                if getattr(self, 'mission_phase', None) == getattr(self, 'PHASE_FINAL_SEARCH_WAIT', -1):
                    self.mission_phase = self.PHASE_FINAL_SEARCH_ACTIVE
                    
                return
            # ===================================================================================
            idx = next((k for k, n in enumerate(self.mission_nodes) if int(n['id']) == target_id), None)
            if idx is None:
                self.get_logger().warn(f"⚠️ VLM selected Node {target_id}, but it's not in mission_nodes.")
                return

            # 跳转到该节点：清空当前路径，让状态机下一轮按新 idx 规划
            self.current_node_idx = idx
            self.target_point = None
            self.current_path = []
            self.last_path_index = 0
            self.current_state = self.STATE_PATROLLING
            self.deferred_vl_candidates = []
            self.vl_goal_rel = None
            self.pending_vl_goal_world = None
            # [High-level] 若当前处于 FINAL_SEARCH_WAIT，则在收到 /audit_nav/result 后进入 FINAL_SEARCH_ACTIVE
            try:
                if (getattr(self, 'mission_phase', None) == getattr(self, 'PHASE_FINAL_SEARCH_WAIT', -1) or
                    bool(getattr(self, 'waiting_for_vlm_search', False))):
                    self.mission_phase = self.PHASE_FINAL_SEARCH_ACTIVE
                    self.waiting_for_vlm_search = False
                    self.final_search_request_time = 0.0
                    if getattr(self, 'primary_instruction', ''):
                        self.search_instruction = str(self.primary_instruction)
            except Exception:
                pass

            # 重置卡死/脱困状态，避免切换目标瞬间误触发“后退+冲刺”
            self.stuck_timer = 0.0
            self.last_target_speed = 0.0
            self.force_recovery_state = "NONE"
            self.recovery_disable_until = time.time() + 2.0

            self.get_logger().info(
                f"🎯 VLM Search Selected Node {target_id} (score={best.get('score')}). Jumping to it."
            )

        except Exception as e:
            self.get_logger().error(f"❌ Failed to parse VLM result: {e}")

    def get_robot_pose(self):
        """ 
        从 TF 树查询机器人当前在 map 坐标系下的位姿。
        返回: (x, y, yaw, pitch)
        """
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame_id, "base_link", rclpy.time.Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            # 四元数转欧拉角 (Yaw)
            yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
            # 计算 Pitch 角用于爬坡冲刺检测
            pitch = math.asin(np.clip(2 * (q.w * q.y - q.z * q.x), -1.0, 1.0)) 
            return t.x, t.y, yaw, pitch
        except Exception:
            # TF 可能会在启动初期失败，属于正常现象
            return None

    def world_to_grid(self, wx, wy):
        """ 将世界坐标 (米) 转换为栅格地图坐标 (行列索引) """
        if not self.map_info: return None, None
        res = self.map_info.resolution
        c = int((wx - self.map_info.origin.position.x) / res)
        r = int((wy - self.map_info.origin.position.y) / res)
        if 0 <= c < self.map_info.width and 0 <= r < self.map_info.height:
            return r, c
        return None, None

    def grid_to_world(self, r, c):
        """ 将栅格地图坐标转换为世界坐标 (米) - 返回中心点 """
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        return c * res + ox + res/2, r * res + oy + res/2

    # 🔥 [新增] 路径实时一致性检查 (Path Consistency Check)
    def check_current_path_safety(self):
        """
        检查当前剩余路径是否被新发现的障碍物阻挡。
        如果路径上前方的一段距离内出现了障碍物 (value > 50)，返回 True (不安全)。
        """
        if self.local_map_data is None or not self.current_path:
            return False
            
        # 只检查接下来的一段路径 (例如 20 个点，约 1米)
        # 没必要检查太远，因为远处的不影响当前运动，且可能会被后续更新覆盖
        start_idx = self.last_path_index
        end_idx = min(len(self.current_path), start_idx + 20)
        
        for i in range(start_idx, end_idx):
            r, c = self.current_path[i]
            # 边界检查
            if 0 <= r < self.map_info.height and 0 <= c < self.map_info.width:
                # 如果发现障碍物 (假设 > 50 为障碍)
                cell_value = self.local_map_data[r, c]
                if cell_value > 50:
                    return True # Path is blocked by a wall!
        return False

    # ================= 探索目标点安全修正 =================

    def snap_goal_to_safe_free(self, r0, c0, unsafe_zone, max_radius_px):
        """若目标点落在未知/障碍/危险膨胀区，则在邻域内搜索最近的安全空闲栅格点。"""
        if self.local_map_data is None:
            return None
        h, w = self.local_map_data.shape

        # 边界保护
        if r0 is None or c0 is None or not (0 <= r0 < h and 0 <= c0 < w):
            return None

        # 如果本身已是安全空闲，直接返回
        if self.local_map_data[r0, c0] == 0 and (unsafe_zone is None or unsafe_zone[r0, c0] == 0):
            return (r0, c0)

        # BFS 搜索邻域内最近安全点
        q = deque()
        q.append((r0, c0, 0))
        visited = set()

        # 8 邻域
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while q:
            r, c, d = q.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))

            if d > max_radius_px:
                continue
            if not (0 <= r < h and 0 <= c < w):
                continue

            if self.local_map_data[r, c] == 0 and (unsafe_zone is None or unsafe_zone[r, c] == 0):
                return (r, c)

            nd = d + 1
            for dr, dc in neigh:
                q.append((r + dr, c + dc, nd))

        return None




# ==============================================================================
# 🧠 [VL-Nav Inspired] 高层候选目标池 (Frontier + Instance)
# 说明：
#   - 不改变底层控制/参数，只改“选哪个目标点”的高层决策。
#   - Instance candidates 来自 VL 节点(如 YOLO-World)给出的相对目标点，经 integrated 转为 world 后缓存。
#   - 探索阶段优先尝试 Instance（更像论文的 IBTP: instance-based target points），失败再走 Frontier。
# ==============================================================================

# ================= VL 目标点可达性修正 (Safe Goal Adjustment) =================
    def _compute_unsafe_zone_mask(self):
        """生成 'unsafe_zone'：障碍物膨胀 + Unknown，一律视为不可作为导航目标的区域。"""
        if self.local_map_data is None or self.map_info is None:
            return None
        try:
            occ = (self.local_map_data >= 80).astype(np.uint8)  # occupied
            unk = (self.local_map_data < 0).astype(np.uint8)    # unknown
            safe_radius_px = max(1, int(0.30 / float(self.map_info.resolution)))
            k = np.ones((safe_radius_px * 2 + 1, safe_radius_px * 2 + 1), np.uint8)
            unsafe = cv2.dilate(occ, k)
            unsafe = np.clip(unsafe + unk, 0, 1).astype(np.uint8)
            return unsafe
        except Exception:
            return None

    def adjust_world_goal_to_safe_free(self, gx, gy, max_radius_m=1.2):
        """将世界坐标目标吸附到最近的安全 Free 栅格（用于：VL 目标点可能落在障碍/膨胀区）。"""
        if self.local_map_data is None or self.map_info is None:
            return gx, gy, False
        r0, c0 = self.world_to_grid(gx, gy)
        if r0 is None or c0 is None:
            return gx, gy, False

        unsafe_zone = self._compute_unsafe_zone_mask()
        max_radius_px = max(1, int(float(max_radius_m) / float(self.map_info.resolution)))
        snapped = self.snap_goal_to_safe_free(int(r0), int(c0), unsafe_zone, max_radius_px)

        if snapped is None:
            return gx, gy, False

        sr, sc = snapped
        wx, wy = self.grid_to_world(sr, sc)
        moved = math.hypot(wx - gx, wy - gy)
        return float(wx), float(wy), (moved > 1e-3)

    # ================= 重复脱困 -> 判定目标不可达 (Avoid infinite backing loop) =================
    def _goal_key(self, pt):
        if pt is None:
            return None
        try:
            r, c = self.world_to_grid(pt[0], pt[1])
            if r is None or c is None:
                return (round(float(pt[0]), 2), round(float(pt[1]), 2))
            return (int(r), int(c))
        except Exception:
            return None

    def _maybe_abandon_goal_after_repeated_backing(self, now, reason=""):
        """同一目标在短时间内反复触发后退/碰撞，判揭示目标不可达：加入黑名单并退出该目标。"""
        if self.target_point is None:
            return False

        key = self._goal_key(self.target_point)
        if key is None:
            return False

        if getattr(self, "_goal_fail_key", None) == key and (now - getattr(self, "_goal_fail_t0", now)) < 20.0:
            self._goal_fail_count = int(getattr(self, "_goal_fail_count", 0)) + 1
        else:
            self._goal_fail_key = key
            self._goal_fail_t0 = float(now)
            self._goal_fail_count = 1

        # 连续 3 次（~几秒内）仍无法脱离：直接判不可达，避免无尽 back-replan 循环
        if self._goal_fail_count >= 3:
            self.get_logger().warn(
                f"🛑 Goal seems unreachable (repeated backing x{self._goal_fail_count}). Abandon goal. reason={reason}"
            )
            try:
                self.add_unreachable_point(self.target_point)
            except Exception:
                pass

            # 若是 VL 目标：反馈给上层，让 VLM / VL 模块换目标或换节点
            try:
                if self.current_state == self.STATE_GO_TO_VL_GOAL:
                    self._publish_vl_feedback(
                        "vl_goal_stuck_unreachable",
                        extra={
                            "goal_world": [float(self.target_point[0]), float(self.target_point[1])],
                            "reason": str(reason),
                            "backing_count": int(self._goal_fail_count),
                        },
                    )
            except Exception:
                pass

            # 退出当前目标
            self.stop_robot()
            self.target_point = None
            self.current_path = []
            self.last_path_index = 0
            self.wall_backing_active = False
            self.threshold_backing_active = False

            # 回到上一状态/巡逻
            if self.current_state == self.STATE_GO_TO_VL_GOAL:
                ret = self.vl_return_state if self.vl_return_state is not None else self.STATE_PATROLLING
                self.current_state = ret
                self.vl_return_state = None

            # 短暂稳定窗口
            self.recovery_disable_until = float(now) + 2.0
            return True

        return False


    def _vl_compute_unsafe_zone(self):
        """Compute a conservative 'unsafe zone' (dilated obstacles) for goal snapping."""
        try:
            if self.local_map_data is None or self.map_info is None:
                return None
            obstacle = (self.local_map_data > 50).astype(np.uint8) * 255
            safe_radius_px = int(0.3 / float(self.map_info.resolution))
            k = safe_radius_px * 2 + 1
            if k <= 1:
                return obstacle
            kernel = np.ones((k, k), np.uint8)
            return cv2.dilate(obstacle, kernel)
        except Exception:
            return None

    def _vl_plan_to_world_goal(self, rx, ry, gx, gy):
        """Plan path to a world goal with snapping + standoff candidates.
        Returns: (path, gx_adj, gy_adj, meta_dict)
        """
        meta = {"snapped": False, "standoff": False, "orig": (gx, gy), "used": (gx, gy)}
        try:
            if self.local_map_data is None or self.map_info is None:
                return None, gx, gy, meta

            s_g = self.world_to_grid(rx, ry)
            g0 = self.world_to_grid(gx, gy)
            if s_g[0] is None or g0[0] is None:
                return None, gx, gy, meta

            unsafe = self._vl_compute_unsafe_zone()
            snap_px = int(self.vl_goal_snap_max_m / float(self.map_info.resolution))
            snap_px = max(0, snap_px)

            # (1) Try direct goal (snap if needed)
            g_use = g0
            g_snap = self.snap_goal_to_safe_free(g0[0], g0[1], unsafe, snap_px) if snap_px > 0 else None
            if g_snap is not None:
                g_use = g_snap
                if g_use != g0:
                    meta["snapped"] = True

            if g_use[0] is not None:
                path = self.planner.plan(self.local_map_data, s_g, g_use)
                if path:
                    wx, wy = self.grid_to_world(g_use[0], g_use[1])
                    meta["used"] = (wx, wy)
                    return path, wx, wy, meta

            # (2) Try standoff candidates around original goal
            best = None
            for r_m in self.vl_goal_standoff_radii_m:
                for deg in (0, 45, 90, 135, 180, 225, 270, 315):
                    ang = math.radians(deg)
                    cx = gx + float(r_m) * math.cos(ang)
                    cy = gy + float(r_m) * math.sin(ang)
                    gc = self.world_to_grid(cx, cy)
                    if gc[0] is None:
                        continue
                    gc2 = self.snap_goal_to_safe_free(gc[0], gc[1], unsafe, snap_px) if snap_px > 0 else None
                    if gc2 is None:
                        continue
                    path_c = self.planner.plan(self.local_map_data, s_g, gc2)
                    if not path_c:
                        continue
                    if best is None or len(path_c) < len(best[0]):
                        wx, wy = self.grid_to_world(gc2[0], gc2[1])
                        best = (path_c, wx, wy, r_m, deg)
            if best:
                path_c, wx, wy, r_m, deg = best
                meta["snapped"] = True
                meta["standoff"] = True
                meta["standoff_r_m"] = float(r_m)
                meta["standoff_deg"] = int(deg)
                meta["used"] = (wx, wy)
                return path_c, wx, wy, meta

            return None, gx, gy, meta
        except Exception:
            return None, gx, gy, meta

    def _vl_should_defer_preempt(self, rx, ry):
        """During PATROLLING (en-route to a VLM-selected node), decide whether to defer VL preempt."""
        try:
            if self.current_state != self.STATE_PATROLLING:
                return False
            if self.mission_nodes is None or len(self.mission_nodes) == 0:
                return False
            if self.current_node_idx is None or not (0 <= int(self.current_node_idx) < len(self.mission_nodes)):
                return False
            nx, ny = self.mission_nodes[int(self.current_node_idx)]['coords']
            dist_to_node = math.hypot(float(nx) - float(rx), float(ny) - float(ry))
            return dist_to_node > float(self.vl_preempt_near_node_m)
        except Exception:
            return False


    def _rect_sum_integral(self, integ_img, x0, y0, x1, y1):
        """O(1) 计算积分图窗口和。坐标为像素/栅格坐标，含边界裁剪。"""
        h = integ_img.shape[0] - 1
        w = integ_img.shape[1] - 1
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        # integral uses [y, x]
        return int(integ_img[y1, x1] - integ_img[y0, x1] - integ_img[y1, x0] + integ_img[y0, x0])

    def remember_instance_candidate(self, pt_world, source="vl", t=None):
        """缓存一个 instance 候选点（world 坐标）。用于探索阶段优先验证该线索。"""
        if t is None:
            t = time.time()

        if pt_world is None:
            return
        # 冷却点过滤：避免误检点反复进入候选池
        try:
            if self._vl_is_on_cooldown(float(pt_world[0]), float(pt_world[1]), t):
                return
        except Exception:
            pass

        # 去重：复用既有 BLACKLIST_RADIUS 作为空间合并半径（不引入新参数）
        for it in self.instance_candidates:
            try:
                if math.hypot(pt_world[0] - it['pt'][0], pt_world[1] - it['pt'][1]) < self.BLACKLIST_RADIUS:
                    it['pt'] = pt_world
                    it['t'] = t
                    it['source'] = source
                    return
            except Exception:
                continue

        self.instance_candidates.appendleft({'pt': pt_world, 't': t, 'source': source})

    def _is_point_blacklisted(self, wx, wy):
        for bpt in self.unreachable_points:
            if math.hypot(wx - bpt[0], wy - bpt[1]) < self.BLACKLIST_RADIUS:
                return True
        return False

    def _is_point_safe_free(self, wx, wy):
        """检查 world 点对应栅格是否为可通行 free（并在图内）。"""
        if self.local_map_data is None or self.map_info is None:
            return False
        g = self.world_to_grid(wx, wy)
        if g is None:
            return False
        r, c = g
        h, w = self.local_map_data.shape
        if not (0 <= r < h and 0 <= c < w):
            return False
        return (self.local_map_data[r, c] == 0)

    def get_instance_candidates(self, rx, ry):
        """返回排序后的 instance 候选点列表（不规划，只做排序/过滤）。"""
        if not self.instance_candidates or self.local_map_data is None:
            return []

        items = []
        # 先 recent 再近距离（无权重，字典序排序）
        for it in list(self.instance_candidates):
            pt = it.get('pt', None)
            if pt is None:
                continue
            wx, wy = float(pt[0]), float(pt[1])

            # 黑名单过滤
            if self._is_point_blacklisted(wx, wy):
                continue

            # 冷却点过滤（VL 近距验证失败/歧义的地点）
            try:
                if self._vl_is_on_cooldown(wx, wy, time.time()):
                    continue
            except Exception:
                pass
            # 栅格必须可通行（避免旧检测点落在障碍/未知）
            if not self._is_point_safe_free(wx, wy):
                continue

            dist = math.hypot(wx - rx, wy - ry)
            items.append({'pt': (wx, wy), 'score': -dist, 't': float(it.get('t', 0.0)), 'kind': 'instance'})

        items.sort(key=lambda d: (d['t'], d['score']), reverse=True)
        return items

    def get_exploration_candidates(self, rx, ry):
        """探索阶段候选池：Instance 优先 + Frontier（带 Curiosity 评分）。"""
        inst = self.get_instance_candidates(rx, ry)
        front = self.get_frontiers()
        # Instance 放前面，确保先尝试 IBTP 线索
        return inst + front

        # 🔥 [Smart-V2 核心逻辑] 安全的前沿点检测 🔥
        # 包含：墙体安全膨胀、黑名单过滤
    def get_frontiers(self):
        """
        基于图像处理提取地图前沿点 (Frontiers)。
        前沿点是 已知区域(Free) 与 未知区域(Unknown) 的交界处。
        """
        if self.local_map_data is None: return []
        
        grid = self.local_map_data
        h, w = grid.shape
        
        # 1. 定义区域掩码
        free = (grid == 0).astype(np.uint8) * 255
        unknown = (grid == -1).astype(np.uint8) * 255
        obstacle = (grid > 50).astype(np.uint8) * 255
        # [VL-Nav] Curiosity: precompute unknown integral image for O(1) unknown-count query
        unknown_u8 = (grid == -1).astype(np.uint8)
        unknown_integ = cv2.integral(unknown_u8)
        
        # 2. 计算墙体安全膨胀区 (0.3m)
        # 防止目标点生成在墙根，导致机器人贴墙无法到达
        safe_radius_px = int(0.3 / self.map_info.resolution)
        unsafe_zone = cv2.dilate(obstacle, np.ones((safe_radius_px*2+1, safe_radius_px*2+1), np.uint8))
        
        # 3. 提取前沿边缘
        # 对 Free 区域膨胀，然后与 Unknown 区域取交集，得到边缘
        dilated_free = cv2.dilate(free, np.ones((3,3), np.uint8))
        frontier_mask = cv2.bitwise_and(dilated_free, unknown)
        
        # 查找轮廓
        contours, _ = cv2.findContours(frontier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        rx, ry, _, _ = self.get_robot_pose()
        
        for cnt in contours:
            if len(cnt) < 5: continue # 忽略过小的噪点
            
            # 计算质心
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            
            # 4. 安全性检查：如果几何中心在不安全区域，尝试找轮廓上的安全点
            if not (0 <= cx < w and 0 <= cy < h) or unsafe_zone[cy, cx] != 0:
                best_pt = None
                min_dist = float('inf')
                
                for point in cnt:
                    px, py = point[0]
                    # 检查轮廓点是否在图像内且不在危险区
                    if 0 <= px < w and 0 <= py < h and unsafe_zone[py, px] == 0:
                        d = (px - cx)**2 + (py - cy)**2
                        if d < min_dist:
                            min_dist = d
                            best_pt = (px, py)
                
                if best_pt:
                    cx, cy = best_pt
                else:
                    continue # 整个前沿都在危险区，放弃该候选            # 5. 目标点修正：前沿点本身常在 Unknown 区，可能对应真实障碍
            #    若落在 Unknown/Obstacle/Unsafe，则在邻域内“往外拉”到最近的安全 Free 栅格
            max_radius_px = int(0.8 / self.map_info.resolution)
            snapped = self.snap_goal_to_safe_free(cy, cx, unsafe_zone, max_radius_px)
            if not snapped:
                continue
            cy, cx = snapped

            # 6. 坐标转换
            wx, wy = self.grid_to_world(cy, cx)
            # 6. 黑名单过滤 (Blacklist)
            # 如果该点之前被标记为不可达，则跳过
            is_blacklisted = False
            for bpt in self.unreachable_points:
                if math.hypot(wx - bpt[0], wy - bpt[1]) < self.BLACKLIST_RADIUS: 
                    is_blacklisted = True
                    break
            if is_blacklisted: continue
            
            # [VL-Nav] Curiosity: 未知区域增益 (Unknown-area weighting)
            # 使用 safe_radius_px 的 2 倍窗口（不引入新参数），优先选择能揭示更多未知区域的前沿。
            curiosity_r = safe_radius_px * 2
            x0, x1 = cx - curiosity_r, cx + curiosity_r
            y0, y1 = cy - curiosity_r, cy + curiosity_r
            unknown_gain = self._rect_sum_integral(unknown_integ, x0, y0, x1, y1)

            # 评分：距离越近分数越高 (使用负距离方便排序)
            score = -math.hypot(wx - rx, wy - ry)
            semantic_score = 0.0
            try:
                if (getattr(self, 'mission_phase', None) == getattr(self, 'PHASE_EXPLORE_GUIDED', -999) and
                    bool(getattr(self, 'primary_instruction', '')) and
                    (not bool(getattr(self, 'soft_found', False)))):
                    semantic_score = float(self._semantic_score_at(wx, wy))
            except Exception:
                semantic_score = 0.0

            candidates.append({'pt': (wx, wy), 'score': score, 'unknown_gain': unknown_gain, 'semantic_score': semantic_score, 'kind': 'frontier'})
            
        # [VL-Nav] 排序：先未知增益，再近距离（不引入额外权重参数）
        # After first soft_found, stop semantic guidance and favor shorter travel: pick nearest frontiers first.
        if bool(getattr(self, 'soft_found', False)):
            return sorted(candidates, key=lambda x: (x.get('score', 0.0), x.get('unknown_gain', 0)), reverse=True)

        return sorted(candidates, key=lambda x: (x.get('semantic_score', 0.0), x.get('unknown_gain', 0), x.get('score', 0.0)), reverse=True)

    # === 整合逻辑：拓扑生成 + 坐标转换 + 节点筛选 + 数据保存 + 可视化 ===
    def generate_topo_and_patrol_list(self):
        """
        核心流程：
        1. 停止探索，基于当前地图生成拓扑结构 (Test.py)
        2. 读取标定文件，计算真实坐标 (Coordinate Calculator)
        3. 保存 topology_graph.json 和 final_node_coordinates.json
        4. 生成巡逻任务列表 (只包含 Room 类型)
        """
        self.get_logger().info("📐 Generation Phase: Topology -> Calculator -> Traversal Filter")
        time_str = time.strftime("%Y%m%d_%H%M%S")
        # 准备二值图
        bin_m = np.zeros_like(self.local_map_data, dtype=np.uint8)
        
        # 🔥 [修复 1] 严格剔除灰色区域 (-1)
        # 只有 [0, 50) 范围内的才是确定的空闲区域
        # -1 (int8) 在 numpy 中比较时， -1 < 50 是 True，所以必须加 >= 0 的条件
        free_mask = (self.local_map_data >= 0) & (self.local_map_data < 50)
        bin_m[free_mask] = 255
        
        # 🔥 [修复 2] 形态学去噪：开运算去除孤立噪点，闭运算填补小洞
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_m = cv2.morphologyEx(bin_m, cv2.MORPH_OPEN, kernel)
        bin_m = cv2.morphologyEx(bin_m, cv2.MORPH_CLOSE, kernel)
        
        # 注意：Smart Navigator V2 习惯 flip 地图处理
        bin_m = cv2.flip(bin_m, 0)
        
        # 1. 调用 Test 模块的完整拓扑生成链
        # 🔥 [修复 3] 减小最小间距 (18px = 0.9m)，避免密集点，同时保留厕所
        centers = get_room_centers(bin_m, min_distance=35)
        
        skel = get_raw_skeleton(bin_m)
        G = graph_from_skeleton(skel)
        safe_G = remove_colliding_nodes(G, bin_m)
        clean_G = clean_skeleton_graph(safe_G)
        
        # 构建最终拓扑 (含 get_safe_path)
        nodes, final_G = build_final_topology(clean_G, centers, bin_m)
        
        # 提取关系 (含图膨胀)
        topo_data, sorted_nodes = extract_topology_relationships(final_G, nodes)
        
        # 🔥 [新增功能 1] 保存 topology_graph.json
        try:
            topo_json_output = []
            for node_id, data in topo_data.items():
                neighbors = [n['id'] for n in data['neighbors']]
                topo_json_output.append({
                    "id": int(node_id),
                    "type": data['type'],
                    "position": (int(data['pos'][0]), int(data['pos'][1])),
                    "connected_to": [int(nid) for nid in neighbors],
                    "semantic_info": {"label": "Unknown", "objects": []}
                })
            # 🔥 动态文件名
            topo_filename = f"topology_graph_{time_str}.json"
            with open(topo_filename, 'w') as f:
                json.dump(topo_json_output, f, indent=4)
            self.get_logger().info(f"✅ JSON Saved: {topo_filename} ({len(topo_json_output)} nodes)")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save topology JSON: {e}")
        # 这是一个用于调试的重要步骤，确保拓扑生成逻辑正确
        try:
            # 1. 准备底图 (将二值地图转换为彩色图像以便绘制)
            # 使用原始 local_map_data 以保留灰色区域 (-1)
            raw_map = self.local_map_data.copy()
            
            # 创建可视化用的灰度图
            # -1 -> 128 (Gray), 0 -> 255 (White), >0 -> 0 (Black)
            display_map = np.zeros_like(raw_map, dtype=np.uint8)
            display_map[raw_map == -1] = 128 
            display_map[raw_map == 0] = 255
            display_map[raw_map > 0] = 0
            
            # 翻转以匹配拓扑计算的坐标系 (Smart Navigator 习惯)
            display_map = cv2.flip(display_map, 0)
            
            # 转为彩色以便绘制彩色节点
            vis_img = cv2.cvtColor(display_map, cv2.COLOR_GRAY2BGR)
            
            # 2. 绘制边 (Edges) - 连接关系
            # 遍历图中的每一条边，绘制灰色线条
            for u, v in final_G.edges():
                # u, v 都是 (x, y) 坐标元组
                pt1 = (int(u[0]), int(u[1]))
                pt2 = (int(v[0]), int(v[1]))
                cv2.line(vis_img, pt1, pt2, (200, 200, 200), 1)
            
            # 3. 绘制节点 (Nodes) - 关键位置
            for i, node in enumerate(sorted_nodes):
                x, y = int(node['xy'][0]), int(node['xy'][1])
                ntype = node['type']
                
                # 颜色区分：Room=红色, Connector=绿色
                color = (0, 0, 255) if ntype == 'Room' else (0, 255, 0)
                # 大小区分
                radius = 6 if ntype == 'Room' else 4
                
                # 绘制实心圆点
                cv2.circle(vis_img, (x, y), radius, color, -1)
                
                # 绘制节点 ID (带黑色描边的白色文字，确保清晰可见)
                text = str(i)
                # 黑色描边
                cv2.putText(vis_img, text, (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 3)
                # 白色前景
                cv2.putText(vis_img, text, (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            # 4. 保存图片
            vis_filename = f"topology_visualized_{time_str}.png"
            cv2.imwrite(vis_filename, vis_img)
            self.get_logger().info(f"✅ Image Saved: {vis_filename}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save topology image: {e}")

        # 2. 调用 Calculator 模块进行坐标转换
        if os.path.exists(self.CALIB_FILE):
            with open(self.CALIB_FILE, 'r') as f:
                calib_data = json.load(f)
            
            # 计算变换矩阵
            params = calculate_linear_mapping(calib_data)
            
            if params:
                sx, sy, ox, oy = params
                self.mission_nodes = []
                
                final_coords_output = [] # 用于保存所有节点坐标
                
                # 3. 遍历拓扑数据，转换坐标
                for i, d in topo_data.items():
                    # 应用线性变换
                    # 注意：因为 bin_m 翻转了，这里 v 也要处理
                    u, v_flipped = d['pos']
                    wx = u * sx + ox
                    wy = v_flipped * sy + oy
                    
                    # 添加到文件列表
                    neighbors = [n['id'] for n in d['neighbors']]
                    final_coords_output.append({
                        "id": int(i),
                        "type": d['type'],
                        "pixel_coords": [int(u), int(v_flipped)],
                        "world_coords": [round(wx, 3), round(wy, 3)],
                        "connected_to": [int(nid) for nid in neighbors]
                    })

                    # 筛选巡逻节点 (只保留 Room)
                    if d['type'] == 'Room': 
                        self.mission_nodes.append({'id': i, 'coords': (wx, wy)})
                
# 🔥 [保存功能 2] 保存 final_node_coordinates.json
                try:
                    coords_filename = f"final_node_coordinates_{time_str}.json"
                    with open(coords_filename, 'w') as f:
                        json.dump(final_coords_output, f, indent=4)
                    self.get_logger().info(f"✅ JSON Saved: {coords_filename}")
                except Exception as e:
                    self.get_logger().error(f"Failed to save coords JSON: {e}")

                # =========================================================
                # 🔥 核心优化：贪心最短路径重排 (Greedy TSP) 🔥
                # 解决巡逻时来回折返跑的问题，让小车按照物理距离就近遍历
                # =========================================================
                if len(self.mission_nodes) > 1:
                    # 获取机器人当前位姿作为起点
                    pose = self.get_robot_pose()
                    if pose:
                        curr_x, curr_y = pose[0], pose[1]
                    else:
                        # 如果拿不到位姿，默认从第一个节点开始
                        curr_x, curr_y = self.mission_nodes[0]['coords']

                    unvisited = self.mission_nodes.copy()
                    optimized_nodes = []

                    while unvisited:
                        best_idx = 0
                        min_dist = float('inf')
                        
                        for idx, node in enumerate(unvisited):
                            nx, ny = node['coords']
                            
                            # 🔥 核心修复：给 TSP 预演也加上安全吸附，和实际导航待遇完全一致！
                            cx_safe, cy_safe, _ = self.adjust_world_goal_to_safe_free(curr_x, curr_y, max_radius_m=0.8)
                            nx_safe, ny_safe, _ = self.adjust_world_goal_to_safe_free(nx, ny, max_radius_m=1.2)
                            
                            # 1. 获取两点吸附后的安全栅格坐标
                            s_g = self.world_to_grid(cx_safe, cy_safe)
                            g_g = self.world_to_grid(nx_safe, ny_safe)
                            
                            dist = float('inf')
                            
                            # 2. 调用底层的 A* 规划器预演路线，计算真实的“避障路径长度”
                            if s_g[0] is not None and g_g[0] is not None:
                                path = self.planner.plan(self.local_map_data, s_g, g_g)
                                if path:
                                    # 路径包含的栅格点数 * 地图分辨率 = 真实需要行驶的物理距离 (米)
                                    dist = len(path) * self.map_info.resolution
                            
                            # 3. 兜底保护：如果 A* 无法规划出路径（可能目标点被临时杂物挡住），
                            # 退化为“加上极大惩罚的欧氏距离”，保证程序不会崩溃卡死
                            if dist == float('inf'):
                                dist = 10000.0 + math.hypot(nx - curr_x, ny - curr_y)

                            if dist < min_dist:
                                min_dist = dist
                                best_idx = idx
                        
                        # 选中真实导航距离最近的节点
                        next_node = unvisited.pop(best_idx)
                        optimized_nodes.append(next_node)
                        curr_x, curr_y = next_node['coords']
                    
                    self.mission_nodes = optimized_nodes
                    
                    # 打印优化后的顺序供调试查看
                    ordered_ids = [n['id'] for n in self.mission_nodes]
                    self.get_logger().info(f"🚀 Patrol Route Optimized! Order: {ordered_ids}")
                else:
                    self.mission_nodes.sort(key=lambda x: x['id'])

                self.get_logger().info(f"✅ Generated {len(self.mission_nodes)} patrol nodes.")
                return True
            else:
                self.get_logger().error("❌ Calibration failed (not enough points).")
        else:
            self.get_logger().error(f"❌ Calibration file not found: {self.CALIB_FILE}")
            
        return False

    def check_forward_collision_level(self, robot_pose):
            """ 检测前方是否有障碍物，用于防撞 """
            if self.local_map_data is None: return 0
            rx, ry, ryaw, _ = robot_pose
            
            # 🔥 [优化: 窄缝防卡死] 动态检测距离
            # 如果机器人的线速度很低（说明在原地转圈调整姿态，或者在窄缝里慢慢挤）
            # 我们就收缩“防撞视距”，只要不贴到车身(0.2m)即可；如果跑得快，就维持 0.45m
            check_dist = 0.45
            if hasattr(self, 'current_linear_velocity') and self.current_linear_velocity < 0.25:
                check_dist = 0.28  # 刚好包住 0.2m 的车身，防止转头时扫到两边的墙壁
                
            check_x = rx + check_dist * math.cos(ryaw)
            check_y = ry + check_dist * math.sin(ryaw)
            
            gp = self.world_to_grid(check_x, check_y)
            
            if gp[0] is not None and 0 <= gp[0] < self.local_map_data.shape[0] and 0 <= gp[1] < self.local_map_data.shape[1]:
                if self.local_map_data[gp[0], gp[1]] > 80: 
                    return 2 # 危险
            return 0


    def has_front_obstacle_in_radius(self, robot_pose, radius_m=0.45, occ_threshold=80):
        """ 
        判定机器人前方 180° 范围内（±90°），半径 radius_m 圆内是否存在障碍物。
        - 障碍物判据：occupancy > occ_threshold
        - 仅依赖 /local_map，不修改其他模块参数
        """
        if self.local_map_data is None or self.map_info is None:
            return False
        rx, ry, ryaw, _ = robot_pose
        rr, rc = self.world_to_grid(rx, ry)
        if rr is None:
            return False

        res = self.map_info.resolution
        r_cells = int(math.ceil(radius_m / res))
        h, w = self.local_map_data.shape

        for dr in range(-r_cells, r_cells + 1):
            r = rr + dr
            if r < 0 or r >= h:
                continue
            dy = dr * res
            for dc in range(-r_cells, r_cells + 1):
                c = rc + dc
                if c < 0 or c >= w:
                    continue
                dx = dc * res
                dist = math.hypot(dx, dy)
                if dist <= 1e-6 or dist > radius_m:
                    continue

                # 前方 180°：相对航向夹角在 [-90°, +90°]
                ang = normalize_angle(math.atan2(dy, dx) - ryaw)
                if abs(ang) > (math.pi / 2.0):
                    continue

                if self.local_map_data[r, c] > occ_threshold:
                    return True

        return False

    def choose_backing_turn_sign(self, robot_pose):
        """ 
        为“后退脱困”选择一个更安全的转向方向（避免纯后退在墙角/床边二次顶死）。
        思路：在机器人前方左右两侧做若干个栅格采样，统计“空闲(0)”数量，空闲更多的一侧作为转向方向。
        返回:
            +1: 后退时左转（angular.z > 0）
            -1: 后退时右转（angular.z < 0）
             0: 无法判断/两侧相同
        """
        if self.local_map_data is None or self.map_info is None:
            return 0
        try:
            rx, ry, ryaw, _ = robot_pose
            # 在前方左右扇区采样（角度越大越贴近侧面，越能反映“哪边更空”）
            angles = [math.radians(40), math.radians(70), math.radians(100)]
            radii = [0.35, 0.55, 0.75]

            def sample_score(sign):
                score = 0
                for a in angles:
                    for rr in radii:
                        ang = ryaw + sign * a
                        x = rx + rr * math.cos(ang)
                        y = ry + rr * math.sin(ang)
                        gr, gc = self.world_to_grid(x, y)
                        if gr is None or gc is None:
                            continue
                        v = int(self.local_map_data[gr, gc])
                        if v == 0:
                            score += 1
                return score

            left_score = sample_score(+1.0)
            right_score = sample_score(-1.0)

            if left_score == right_score:
                return 0
            return 1 if left_score > right_score else -1
        except Exception:
            return 0


    def nudge_path_to_center(self, raw_path):
        """ 
        路径中心化优化：让路径尽量走在路中间 (cost 低的地方)
        [修复版] 强制避开未知区域 (-1)
        """
        nudged = []
        for r, c in raw_path:
            # 如果当前点不是绝对空闲 (0)，尝试寻找周围更好的点
            if self.local_map_data[r, c] != 0: 
                best = (r, c)
                min_c = 255
                
                # 7x7 局部搜索
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.local_map_data.shape[0] and 0 <= nc < self.local_map_data.shape[1]:
                            val = self.local_map_data[nr, nc]
                            
                            # ⚠️ 关键修复：将未知区域 (-1) 视为高代价 (127)，防止吸附
                            if val == -1: val = 127 
                            
                            if val < min_c:
                                min_c = val
                                best = (nr, nc)
                nudged.append(best)
            else:
                nudged.append((r, c))
        return nudged

    def add_unreachable_point(self, pt):
        """ 将无法到达的点加入黑名单 """
        self.unreachable_points.append(pt)
        if len(self.unreachable_points) > self.MAX_BLACKLIST_SIZE:
            self.unreachable_points.pop(0)

    # ==============================================================================
    # 🔥 核心控制循环 (Main Control Loop)
    # 包含：状态机、PID 控制、暴力脱困、动态剪枝、[新增] 路径实时检查
    # ==============================================================================
    def control_loop(self):
        try:
            # 更新可视化面板
            self.visualize_dashboard()
            
            if self.local_map_data is None: return
            pose = self.get_robot_pose()
            if not pose: return
            
            rx, ry, ryaw, pitch = pose
            current_time = time.time()
            dt = current_time - self.last_control_time
            if dt <= 0: dt = 0.1

            # --- 0. High-level gating ---
            if getattr(self, 'mission_phase', None) == getattr(self, 'PHASE_DONE', -999):
                self.action_description = f"✅ DONE: target found [{getattr(self, 'primary_instruction', '')}]"
                try:
                    if not bool(getattr(self, '_eval_done_pub', False)):
                        pose = self.get_robot_pose()
                        if pose is not None:
                            rx, ry, _, _ = pose
                            self._publish_mission_done_event(current_time, rx, ry)
                        self._eval_done_pub = True
                except Exception:
                    pass
                self.stop_robot()
                self.last_control_time = current_time
                return

            # 初始必须先收到目标
            if (getattr(self, 'mission_phase', None) == getattr(self, 'PHASE_WAIT_TARGET', -999) and
                not bool(getattr(self, 'primary_instruction', ''))):
                self.action_description = "⏳ Waiting for target on /audit_nav/instruction..."
                self.stop_robot()
                self.last_control_time = current_time
                return

            # --- 1. 物理状态检测 ---
            pitch_rate = abs(pitch - self.last_pitch)
            self.last_pitch = pitch

            # [新增] 前方 180° 半径 0.3m 障碍物检测：用于区分“过坎”与“撞墙”
            front_has_obstacle = self.has_front_obstacle_in_radius(pose, radius_m=0.45, occ_threshold=80)

            # [新增] 复用前向碰撞等级：用于区分“门槛打滑”与“轻微贴障/狭窄通道”
            # 这样可避免前方障碍判断偶尔漏检时，误触发 Threshold Backing -> Rush，导致反复撞墙。
            collision_level_now = self.check_forward_collision_level(pose)

            # --- 过坎/撞墙判定（基于：前方 180° 半径 0.3m + 低速持续 1.5s）---
            # 注意：不再依赖 last_target_speed（因为在大角度转向/贴墙时目标速度可能被置 0，会导致计时器永远起不来）
            if (current_time > self.recovery_disable_until
                and (self.target_point is not None)
                and (self.force_recovery_state == "NONE")
                and (self.current_state != self.STATE_WAITING_FOR_VLM)
                and (not self.wall_backing_active) and (not self.threshold_backing_active)):

                if self.current_linear_velocity < self.STUCK_VEL_THRESHOLD:
                    # 若 collision_level_now>=1（有贴障风险），也视为“撞墙趋势”，优先走 wall_collision 分支
                    if front_has_obstacle or (collision_level_now >= 1):
                        self.wall_collision_timer += dt
                        self.threshold_timer = 0.0
                    else:
                        self.threshold_timer += dt
                        self.wall_collision_timer = 0.0
                else:
                    self.threshold_timer = 0.0
                    self.wall_collision_timer = 0.0
            else:
                self.threshold_timer = 0.0
                self.wall_collision_timer = 0.0

            # 撞墙：前方有障碍 + 低速持续 -> 先后退 0.2m 再重规划（目标不变）
            if self.wall_collision_timer > self.STUCK_TIME_THRESHOLD:
                self.get_logger().warn("🚨 Wall Collision Detected (front obstacle within 0.45m + low speed). Back up 0.2m then Re-plan.")
                self.stop_robot()

                # 若同一目标反复触发撞墙脱困，判不可达并退出，避免无尽循环
                if self._maybe_abandon_goal_after_repeated_backing(current_time, reason="wall_collision_timer"):
                    return

                self.wall_backing_active = True
                self.wall_back_start_xy = (rx, ry)
                self.wall_back_turn_sign = self.choose_backing_turn_sign(pose)
                self.wall_back_start_time = current_time
                self.current_path = []
                self.last_path_index = 0

                self.speed_integral = 0.0
                self.collision_persistence = 0
                self.wall_collision_timer = 0.0
                self.threshold_timer = 0.0
                self.rush_mode_until = 0.0
                return

            # 过坎：前方无障碍 + 低速持续 -> 先后退(带时间上限)再进入冲刺模式
            if self.threshold_timer > self.STUCK_TIME_THRESHOLD:
                self.get_logger().info("🔰 Threshold Detected (no front obstacle within 0.45m + low speed). Back then Rush.")
                self.stop_robot()

                # 进入“门槛脱困后退”状态：最多后退 0.2m 或 1.0s（避免狭窄缝隙卡死）
                self.threshold_backing_active = True
                self.threshold_back_start_xy = (rx, ry)
                self.threshold_back_turn_sign = self.choose_backing_turn_sign(pose)
                self.threshold_back_start_time = current_time

                # 清理计时器，避免反复触发
                self.threshold_timer = 0.0
                self.wall_collision_timer = 0.0
                self.rush_mode_until = 0.0
                self.speed_integral = 0.0

                # 立即发布一次后退指令（下一轮继续由 threshold_backing_active 状态机接管）
                cmd = Twist()
                cmd.linear.x = -0.3
                # 轻微打方向后退：从墙角/床边更容易脱离（不改变任何速度参数，仅复用既有角速度上限0.4）
                turn_sign = getattr(self, 'threshold_back_turn_sign', 0)
                cmd.angular.z = float(np.clip(turn_sign * 0.4, -0.4, 0.4))
                self.pub_vel.publish(cmd)
                self.last_control_time = current_time
                return

            # 冲刺模式触发：
            # 1) 由“过坎”判定触发后，持续一段时间（rush_mode_until）
            # 2) 仍保留 Pitch 快变触发，但要求前方无近距离障碍（避免撞墙时误触发为过坎）
            self.is_in_rush_mode = (current_time < self.rush_mode_until) or ((pitch_rate > 0.02) and (not front_has_obstacle))

            # --- 2.5 撞墙后后退状态机（先后退 0.2m 再重规划） ---
            # 仅在“撞墙判定”触发后启用，不改变其他参数与功能。
            if self.wall_backing_active:
                # 狭窄缝隙可能无法走满 0.2m：加入“最多后退 1s”上限，避免卡死
                self.action_description = "↩️ WALL BACKING (<=0.2m or 1.0s)"
                if self.wall_back_start_time <= 0.0:
                    self.wall_back_start_time = current_time
                start_x, start_y = self.wall_back_start_xy
                moved = math.hypot(rx - start_x, ry - start_y)
                elapsed_back = current_time - self.wall_back_start_time

                cmd = Twist()
                if (moved < self.wall_back_distance) and (elapsed_back < self.wall_back_max_time):
                    # 使用既有后退速度参数（-0.3 m/s），避免引入新的速度参数
                    cmd.linear.x = -0.3
                    # 轻微带一点转向，降低贴边直退导致的二次碰撞
                    turn_sign = getattr(self, 'wall_back_turn_sign', 0)
                    cmd.angular.z = float(np.clip(turn_sign * 0.4, -0.4, 0.4))
                    self.pub_vel.publish(cmd)
                    self.last_control_time = current_time
                    return


                # 后退完成：停止并尝试重规划到当前目标点
                self.stop_robot()
                self.wall_backing_active = False
                self.wall_back_start_time = 0.0

                # 给地图/速度一个“稳定窗口”，避免刚脱困就被 Threshold/Collision 立刻二次判定
                self.recovery_disable_until = current_time + 2.0

                # 重置控制与计时状态，避免立刻二次触发
                self.speed_integral = 0.0
                self.last_speed_error = 0.0
                self.last_alpha = 0.0
                self.threshold_timer = 0.0
                self.wall_collision_timer = 0.0
                self.collision_persistence = 0

                if self.target_point:
                    s_g = self.world_to_grid(rx, ry)
                    g_g = self.world_to_grid(self.target_point[0], self.target_point[1])
                    if s_g[0] is not None and g_g[0] is not None:
                        path = self.planner.plan(self.local_map_data, s_g, g_g)
                    else:
                        path = None

                    if path:
                        self.current_path = self.nudge_path_to_center(path)
                        self.last_path_index = 0
                    else:
                        # 若仍无法规划：在 GO_TO_VL_GOAL 阶段不要“悬空卡死”，而是优雅退出到返回状态/巡逻
                        try:
                            if self.current_state == self.STATE_GO_TO_VL_GOAL:
                                # 视为“VL 目标不可达”——做冷却避免反复被抢占，然后回到原状态
                                try:
                                    if self.target_point is not None:
                                        self._vl_add_cooldown(self.target_point, self.vl_fail_cooldown_sec, reason="wall_back_replan_fail")
                                except Exception:
                                    pass
                                ret_state = self.vl_return_state if self.vl_return_state is not None else self.STATE_PATROLLING
                                self.current_state = ret_state
                                self.vl_return_state = None
                                self.advance_after_vl_search = False
                            else:
                                # 其它状态保持原逻辑：拉黑该目标点
                                self.add_unreachable_point(self.target_point)
                        except Exception:
                            # 保底：避免异常导致控制循环中断
                            self.add_unreachable_point(self.target_point)
                        self.target_point = None
                        self.current_path = []
                        self.last_path_index = 0
                return
            # --- 2.55 门槛脱困：先后退(<=0.45m 或 1.0s)再进入冲刺 ---
            if self.threshold_backing_active:
                self.action_description = "↩️ THRESH BACKING (<=0.45m or 1.0s)"
                start_x, start_y = self.threshold_back_start_xy
                moved = math.hypot(rx - start_x, ry - start_y)
                elapsed_back = current_time - self.threshold_back_start_time
            
                cmd = Twist()
                # 继续后退：满足距离且未超时
                if (moved < self.wall_back_distance) and (elapsed_back < self.wall_back_max_time):
                    cmd.linear.x = -0.3
                    turn_sign = getattr(self, 'threshold_back_turn_sign', 0)
                    cmd.angular.z = float(np.clip(turn_sign * 0.4, -0.4, 0.4))
                    self.pub_vel.publish(cmd)
                    self.last_control_time = current_time
                    return
            
                # 后退结束：立即进入冲刺模式一小段时间（rush_mode_until 控制持续时长）
                self.stop_robot()
                self.threshold_backing_active = False
                self.threshold_back_start_time = 0.0

                # 同样给系统一个稳定窗口，避免立刻重复触发脱困判定
                self.recovery_disable_until = current_time + 2.0
            
                self.rush_mode_until = current_time + self.STUCK_TIME_THRESHOLD
                self.get_logger().info("🔰 Threshold Backing done. Start Rush Mode.")
                self.threshold_timer = 0.0
                self.wall_collision_timer = 0.0
            
                # 立即按冲刺控制发布一次（与 is_in_rush_mode 分支一致）
                cmd = Twist()
                alpha = 0.0
                if self.target_point:
                    alpha = normalize_angle(math.atan2(self.target_point[1]-ry, self.target_point[0]-rx) - ryaw)
                cmd.linear.x = self.RUSH_SPEED
                cmd.angular.z = float(np.clip(alpha * 1.5, -0.4, 0.4))
                self.speed_integral = 0.0
                self.pub_vel.publish(cmd)
                self.last_control_time = current_time
                return
            
            # --- 2. 暴力脱困逻辑 (最高优先级) ---
            if self.force_recovery_state != "NONE":
                elapsed = current_time - self.force_recovery_start_time; cmd = Twist()
                
                if self.force_recovery_state == "BACKING":
                    self.action_description = "⚠️ FORCE BACKING"
                    # 后退 1.0 秒
                    if elapsed < 1.0: 
                        cmd.linear.x = -0.3
                        self.pub_vel.publish(cmd)
                        return
                    else:
                        # 切换到冲刺
                        self.force_recovery_state = "RUSHING"
                        self.force_recovery_start_time = current_time
                        return
                        
                elif self.force_recovery_state == "RUSHING":
                    self.action_description = "🔥 SAVAGE RUSH"
                    # 暴力前冲 1.5 秒
                    if elapsed < 1.5:
                        if self.target_point:
                            # 简单的方向对准
                            alpha = normalize_angle(math.atan2(self.target_point[1]-ry, self.target_point[0]-rx)-ryaw)
                            cmd.angular.z = float(np.clip(alpha * 2.0, -1.0, 1.0))
                        cmd.linear.x = self.FORCE_RUSH_SPEED # 2.0 m/s
                        self.pub_vel.publish(cmd)
                        return
                    else:
                        # 恢复正常
                        self.force_recovery_state = "NONE"
                        self.speed_integral = 0.0
                        return

            # --- 3. 目标决策层 (State Machine) ---
            # [VL-Nav] 若正在查找物体且 VL 节点给出了新目标点，则允许在探索/巡逻阶段立即抢占前往该点。
            # 只改变“高层目标选择”，不改底层控制参数/逻辑。

            # [新增] 近距验证：到达疑似目标点后原地旋转一小段时间，决定“确认成功/判定误检并冷却”
            if self.current_state == self.STATE_VL_VERIFY:
                if not getattr(self, 'enable_vl_verify', True):
                    # Safety: 若因旧状态/竞态进入了 VERIFY，但当前已禁用，则直接按成功结束
                    self.stop_robot()
                    instr = self.vl_verify_instruction or str(self.search_instruction)
                    self.get_logger().info(f"✅ SUCCESS! [{instr}] reached. (verify disabled) Target acquired!")
                    # [Eval] reach event (verify disabled)
                    try:
                        pose = self.get_robot_pose()
                        if pose is not None:
                            rx, ry, _, _ = pose
                            phase = getattr(self, 'mission_phase', None)
                            if phase == getattr(self, 'PHASE_EXPLORE_GUIDED', -1):
                                if not bool(getattr(self, '_eval_soft_found_pub', False)):
                                    self._publish_reach_event('soft_found', time.time(), rx, ry, goal_xy=getattr(self, 'target_point', None), instr=str(self.search_instruction))
                                    self._eval_soft_found_pub = True
                                self.soft_found = True
                            elif phase == getattr(self, 'PHASE_FINAL_SEARCH_ACTIVE', -1):
                                if not bool(getattr(self, '_eval_final_found_pub', False)):
                                    self._publish_reach_event('final_found', time.time(), rx, ry, goal_xy=getattr(self, 'target_point', None), instr=str(self.search_instruction))
                                    self._eval_final_found_pub = True
                                self.final_found = True
                                self.mission_phase = self.PHASE_DONE
                    except Exception:
                        pass
                    self.search_instruction = ""
                    self.vl_search_origin_node_id = None
                    ret_state = self.vl_verify_return_state if self.vl_verify_return_state is not None else self.STATE_EXPLORING
                    self.current_state = ret_state
                    self.vl_verify_start_time = 0.0
                    self.vl_verify_max_conf = 0.0
                    self.vl_verify_goal = None
                    self.vl_verify_return_state = None
                    self.vl_verify_instruction = ""
                    self.stuck_timer = 0.0
                    self.last_target_speed = 0.0
                    self.force_recovery_state = "NONE"
                    self.recovery_disable_until = time.time() + 2.0
                    return

                self.action_description = f"VL_VERIFY [{self.vl_verify_instruction}] conf={self.vl_best_conf:.2f}"
                # 验证过程中持续更新最大置信度（只取新鲜值）
                conf_now = float(self.vl_best_conf) if (current_time - self.vl_conf_time) < self.vl_conf_fresh_sec else 0.0
                if conf_now > self.vl_verify_max_conf:
                    self.vl_verify_max_conf = conf_now

                if (current_time - self.vl_verify_start_time) < self.vl_verify_duration:
                    cmd = Twist()
                    cmd.linear.x = 0.0
                    cmd.angular.z = self.MAX_ROTATION_SPEED * 0.25
                    self.pub_vel.publish(cmd)
                    return

                # 结束验证：停止
                self.stop_robot()
                maxc = float(self.vl_verify_max_conf)

                if maxc >= self.vl_th_on:
                    # ✅ 近距确认：结束本次查找（防止继续探索被其它误检打断）
                    self.get_logger().info(f"✅ VL verified target [{self.vl_verify_instruction}] max_conf={maxc:.2f}.")
                    self.search_instruction = ""
                    # 本轮查找结束：清理起始节点记录
                    self.vl_search_origin_node_id = None
                else:
                    # ❌ 未确认：继续探索，但对该点做冷却，避免反复被抢占
                    cooldown = self.vl_ambig_cooldown_sec if maxc >= self.vl_th_off else self.vl_fail_cooldown_sec
                    if self.vl_verify_goal is not None:
                        try:
                            self._vl_add_cooldown(self.vl_verify_goal, cooldown, reason=f"verify_fail(conf={maxc:.2f})")
                        except Exception:
                            pass

                    # 失败反馈：近距验证仍未找到目标，触发 VLM 重新决策
                    self._publish_vl_feedback("vl_verify_failed", extra={"max_conf": float(maxc), "cooldown_sec": float(cooldown)})
                    self.vl_search_origin_node_id = None
                    self.get_logger().warn(f"⚠️ VL verify failed (max_conf={maxc:.2f}). Cooldown {cooldown:.0f}s and resume.")

                # 恢复到验证前状态
                ret_state = self.vl_verify_return_state if self.vl_verify_return_state is not None else self.STATE_EXPLORING
                self.current_state = ret_state

                # 清理验证状态
                self.vl_verify_start_time = 0.0
                self.vl_verify_max_conf = 0.0
                self.vl_verify_goal = None
                self.vl_verify_return_state = None
                self.vl_verify_instruction = ""
                self.stuck_timer = 0.0
                self.last_target_speed = 0.0
                self.force_recovery_state = "NONE"
                self.recovery_disable_until = time.time() + 2.0
                return

            # ===================== VL Goal Preempt / Defer Logic =====================
            # If vl_perception produces a high-confidence goal, we may preempt exploration/patrolling.
            # However, when we are still en-route to a VLM-selected node for audit, we defer preempt to avoid:
            #   1) skipping node memory update,
            #   2) getting stuck near object boundaries with repeated back+replan.
            vl_world_goal = None
            vl_goal_stamp = 0.0
            vl_goal_source = ""

            if (self.search_instruction and
                self.current_state != self.STATE_PATROLLING and
                self.current_state != self.STATE_VL_SEARCH and
                (getattr(self, 'mission_phase', None) in (getattr(self, 'PHASE_EXPLORE_GUIDED', -1), getattr(self, 'PHASE_FINAL_SEARCH_ACTIVE', -2))) and
                (self.force_recovery_state == "NONE") and (not self.wall_backing_active) and (not self.threshold_backing_active) and
                self._vl_should_preempt(current_time)):

                # (A) Prefer deferred world-goal (cached) if fresh
                if (self.pending_vl_goal_world is not None and
                    (self.pending_vl_goal_time > self.pending_vl_goal_consumed_time) and
                    (current_time - self.pending_vl_goal_time) < self.pending_vl_goal_max_age_sec and
                    (self.pending_vl_goal_label == str(self.search_instruction))):
                    vl_world_goal = tuple(self.pending_vl_goal_world)
                    vl_goal_stamp = float(self.pending_vl_goal_time)
                    vl_goal_source = "pending"

                # (B) Otherwise use the latest live relative-goal from vl_perception
                elif (self.vl_goal_rel is not None) and (self.vl_goal_time > self.vl_goal_consumed_time):
                    x_rel, y_rel = self.vl_goal_rel
                    gx = rx + math.cos(ryaw) * x_rel - math.sin(ryaw) * y_rel
                    gy = ry + math.sin(ryaw) * x_rel + math.cos(ryaw) * y_rel
                    vl_world_goal = (gx, gy)
                    vl_goal_stamp = float(self.vl_goal_time)
                    vl_goal_source = "live"

                # 🔥 修复 3 & 4：在全景拍照(WAITING_FOR_VLM)时，将目标缓存(Defer)起来，绝不打断拍照！
                if (vl_world_goal is not None and
                    (self.current_state in (self.STATE_EXPLORING, self.STATE_PATROLLING, self.STATE_VL_SEARCH, self.STATE_WAITING_FOR_VLM)) and
                    (self.force_recovery_state == "NONE")):

                    gx, gy = vl_world_goal

                    # Cooldown check
                    try:
                        if self._vl_is_on_cooldown(gx, gy, current_time):
                            if vl_goal_source == "live":
                                self.vl_goal_consumed_time = self.vl_goal_time
                            else:
                                self.pending_vl_goal_consumed_time = vl_goal_stamp
                            return
                    except Exception:
                        pass

# 只要是拍照阶段，或者巡逻路上，强制将其存入 Deferred 缓存池，等待结束后使用
# 🔥 核心修复：只要处于拍照阶段或巡逻赶路中，绝对禁止执行抢占！
                    if self._vl_should_defer_preempt(rx, ry) or self.current_state == self.STATE_WAITING_FOR_VLM:
                        # 如果是新鲜看到的目标，才更新缓存和打印日志
                        if vl_goal_source == "live":
                            dist_to_robot = math.hypot(gx - rx, gy - ry)
                            if not hasattr(self, 'deferred_vl_candidates'):
                                self.deferred_vl_candidates = []
                                
                            # 🔥 把这一帧看到的高置信度目标全部扔进候选池！
                            self.deferred_vl_candidates.append({
                                'world': (gx, gy),
                                'conf': float(self.vl_best_conf),
                                'dist': dist_to_robot,
                                'time': float(self.vl_goal_time),
                                'label': str(self.search_instruction)
                            })
                            
                            self.vl_goal_consumed_time = self.vl_goal_time  # 消费掉 live
                            self.get_logger().info(
                                f"📥 Added to candidate pool: ({gx:.2f}, {gy:.2f}) conf={self.vl_best_conf:.2f}, dist={dist_to_robot:.2f}m"
                            )
                        # ⚠️ 最关键的修复：不论来源是 live 还是 pending，都必须 return，强行阻断下游代码！
                        return

                    # Plan with snapping + standoff candidates
                    path_vl, gx_use, gy_use, meta = self._vl_plan_to_world_goal(rx, ry, gx, gy)

                    if path_vl:
                        # record return state
                        self.vl_return_state = self.current_state
                        self.stop_robot()
                        self.target_point = (float(gx_use), float(gy_use))
                        self.current_path = self.nudge_path_to_center(path_vl)
                        self.last_path_index = 0
                        self.current_state = self.STATE_GO_TO_VL_GOAL

                        # 🔥 核心修复：状态切换瞬间，必须重置防卡死计时器，并给予 2 秒钟的起步豁免期！
                        self.stuck_timer = 0.0
                        self.threshold_timer = 0.0
                        self.wall_collision_timer = 0.0
                        self.last_target_speed = 0.0
                        self.force_recovery_state = "NONE"
                        self.recovery_disable_until = current_time + 2.0

                        # consume goal
                        if vl_goal_source == "live":
                            self.vl_goal_consumed_time = self.vl_goal_time
                        else:
                            self.pending_vl_goal_consumed_time = vl_goal_stamp

                        # log meta
                        if meta.get("snapped") or meta.get("standoff"):
                            self.get_logger().info(
                                f"🎯 Preempt: VL goal adjusted {meta.get('orig')} -> {meta.get('used')} "
                                f"(snapped={meta.get('snapped')}, standoff={meta.get('standoff')}) for [{self.search_instruction}]"
                            )
                        else:
                            self.get_logger().info(
                                f"🎯 Preempt: going to VL goal ({gx_use:.2f}, {gy_use:.2f}) for [{self.search_instruction}]"
                            )
                        return
                    else:
                        # Not reachable: mark consumed to avoid repeated tries
                        if vl_goal_source == "live":
                            self.vl_goal_consumed_time = self.vl_goal_time
                        else:
                            self.pending_vl_goal_consumed_time = vl_goal_stamp

                    # log meta
                    if meta.get("snapped") or meta.get("standoff"):
                        self.get_logger().info(
                            f"🎯 Preempt: VL goal adjusted {meta.get('orig')} -> {meta.get('used')} "
                            f"(snapped={meta.get('snapped')}, standoff={meta.get('standoff')}) for [{self.search_instruction}]"
                        )
                    else:
                        self.get_logger().info(
                            f"🎯 Preempt: going to VL goal ({gx_use:.2f}, {gy_use:.2f}) for [{self.search_instruction}]"
                        )
                    return
                else:
                    # Not reachable: mark consumed to avoid repeated tries
                    if vl_goal_source == "live":
                        self.vl_goal_consumed_time = self.vl_goal_time
                    else:
                        self.pending_vl_goal_consumed_time = vl_goal_stamp
            # ========================================================================
            if not self.target_point:

                # [新增] FINAL_SEARCH_WAIT：停止移动，等待 /audit_nav/result 选择下一个候选节点
                if self.current_state == self.STATE_FINAL_SEARCH_WAIT:
                    self.action_description = f"FINAL_SEARCH_WAIT: {getattr(self, 'primary_instruction', '')}"
                    self.stop_robot()
                    try:
                        if (bool(getattr(self, 'waiting_for_vlm_search', False)) and
                            (current_time - float(getattr(self, 'final_search_request_time', 0.0))) > float(getattr(self, 'final_search_request_timeout', 90.0))):
                            self.get_logger().warn("⌛ Waiting /audit_nav/result timeout. Re-issuing instruction.")
                            try:
                                self._mission_start_final_search(now=current_time, reason="result_timeout")
                            except Exception:
                                pass
                    except Exception:
                        pass
                    return

                # [新增] VL 搜索模式：原地旋转等待 vl_perception_v2 返回目标点
                # [新增] VL 搜索模式：原地旋转扫描 30s，全景尽收眼底后结算最高分
                if self.current_state == self.STATE_VL_SEARCH:
                    self.action_description = f"VL Scanning (30s): {self.search_instruction}"

                    # ==========================================================
                    # 1. 如果还没转满 30 秒，持续收集目标并保持旋转！
                    # ==========================================================
                    if (current_time - self.vl_search_start_time) <= self.vl_search_timeout:
                        
                        # 检查当前帧是否有高质量目标
                        is_conf_high = (self.vl_best_conf >= getattr(self, 'vl_th_on', 0.55))
                        is_conf_fresh = (current_time - self.vl_conf_time < getattr(self, 'vl_conf_fresh_sec', 0.6))
                        
                        if (self.vl_goal_rel is not None) and (self.vl_goal_time > self.vl_goal_consumed_time) and is_conf_high and is_conf_fresh:
                            x_rel, y_rel = self.vl_goal_rel
                            gx = rx + math.cos(ryaw) * x_rel - math.sin(ryaw) * y_rel
                            gy = ry + math.sin(ryaw) * x_rel + math.cos(ryaw) * y_rel
                            
                            dist_to_robot = math.hypot(gx - rx, gy - ry)
                            if not hasattr(self, 'deferred_vl_candidates'):
                                self.deferred_vl_candidates = []
                                
                            # 将看到的有效目标悄悄存入候选池 (不立刻执行)
                            self.deferred_vl_candidates.append({
                                'world': (gx, gy),
                                'conf': float(self.vl_best_conf),
                                'dist': dist_to_robot,
                                'time': float(self.vl_goal_time),
                                'label': str(self.search_instruction)
                            })
                            self.vl_goal_consumed_time = self.vl_goal_time # 标记消费
                            self.get_logger().info(f"👀 Scanning... Cached candidate: ({gx:.2f}, {gy:.2f}) conf={self.vl_best_conf:.2f}")

                        # 维持旋转指令
                        cmd = Twist()
                        cmd.linear.x = 0.0
                        cmd.angular.z = self.MAX_ROTATION_SPEED * 0.25 # 以较慢速度稳定旋转
                        self.pub_vel.publish(cmd)
                        return

                    # ==========================================================
                    # 2. 30秒时间到！停止旋转，开始终极打分结算
                    # ==========================================================
                    self.stop_robot()
                    self.get_logger().info("⏳ 30s panoramic scan complete. Scoring all candidates...")
                    
                    valid_cands = []
                    auth_view = getattr(self, 'vlm_authorized_view', "")
                    allowed_views = []
                    if auth_view:
                        adj_map = {
                            "E": ["NE", "E", "SE"], "NE": ["N", "NE", "E"],
                            "N": ["NW", "N", "NE"], "NW": ["W", "NW", "N"],
                            "W": ["SW", "W", "NW"], "SW": ["S", "SW", "W"],
                            "S": ["SE", "S", "SW"], "SE": ["E", "SE", "S"]
                        }
                        allowed_views = adj_map.get(auth_view, [auth_view])
                    
                    if hasattr(self, 'deferred_vl_candidates') and self.deferred_vl_candidates:
                        for cand in self.deferred_vl_candidates:
                            if cand['label'] != str(self.search_instruction): continue
                            
                            cx, cy = cand['world']
                            angle = math.atan2(cy - ry, cx - rx)
                            deg = math.degrees(angle)
                            if deg < 0: deg += 360
                            dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
                            cand_view = dirs[int(round(deg / 45.0)) % 8]
                            
                            # 方向一票否决
                            if auth_view and (cand_view not in allowed_views):
                                self.get_logger().info(f"🚫 Rejected candidate (wrong direction): {cand_view} (VLM authorized: {auth_view})")
                                continue
                            
                            # 终极打分公式 (高分优先，距离近优先)
                            cand['score'] = (cand['conf'] * 10.0) - (cand['dist'] * 0.2)
                            valid_cands.append(cand)
                    
                    # 清空缓存池，为下个节点的扫描做准备
                    self.deferred_vl_candidates = []

                    if valid_cands:
                        # 选出全场最高分！
                        best_cand = max(valid_cands, key=lambda x: x['score'])
                        gx, gy = best_cand['world']
                        self.get_logger().info(f"🏆 Best candidate selected: ({gx:.2f}, {gy:.2f}) score={best_cand['score']:.2f}")
                        
                        # 规划路线 (包含避障 Standoff)
                        path, gx_nav, gy_nav, meta = self._vl_plan_to_world_goal(rx, ry, gx, gy)
                        
                        if meta.get("snapped") or meta.get("standoff"):
                            self.get_logger().info(
                                f"🧲 目标点已安全调整: ({gx:.2f},{gy:.2f}) -> ({gx_nav:.2f},{gy_nav:.2f})"
                            )

                        if path:
                            self.remember_instance_candidate((gx_nav, gy_nav), source="vl")
                            self.target_point = (gx_nav, gy_nav)
                            self.current_path = self.nudge_path_to_center(path)
                            self.last_path_index = 0
                            self.vl_return_state = self.STATE_PATROLLING
                            self.current_state = self.STATE_GO_TO_VL_GOAL
                            return
                        else:
                            self.get_logger().warn("⚠️ Best candidate is completely blocked. Unreachable.")
                            self._publish_vl_feedback("vl_goal_unreachable", extra={"goal_world": [float(gx), float(gy)]})
                    else:
                        # 如果池子里没有点，或者全被方向否决了
                        self.get_logger().warn("⚠️ Scan result: no valid target found within 30s (or all rejected by direction filter).")
                        self._publish_vl_feedback("vl_search_timeout", extra={"timeout_sec": float(self.vl_search_timeout)})
                    
                    # 无论不可达还是没找到，都要结束本次搜索，恢复正常巡逻状态
                    self.vl_search_origin_node_id = None
                    if getattr(self, 'mission_phase', None) == getattr(self, 'PHASE_FINAL_SEARCH_ACTIVE', -1):
                        self.waiting_for_vlm_search = True
                        self.final_search_request_time = current_time
                        self.current_state = self.STATE_FINAL_SEARCH_WAIT
                    else:
                        if self.advance_after_vl_search:
                            self.current_node_idx += 1
                        self.current_state = self.STATE_PATROLLING
                    self.advance_after_vl_search = False
                    return

                # A. 探索模式 (Exploration)
                if self.current_state == self.STATE_EXPLORING:
                    self.action_description = "Exploring..."
                    # [VL-Nav] 候选池：Instance(IBTP) 优先 + Frontier(含 Curiosity)
                    candidates = self.get_exploration_candidates(rx, ry)
                    target_found = False
                    
                    if candidates:
                        for cand in candidates:
                            pt = cand['pt']
                            s_g = self.world_to_grid(rx, ry)
                            g_g = self.world_to_grid(pt[0], pt[1])
                            
                            # 尝试规划路径 (使用内置 A*)
                            path = self.planner.plan(self.local_map_data, s_g, g_g)
                            if path:
                                self.target_point = pt
                                self.current_path = self.nudge_path_to_center(path)
                                self.last_path_index = 0
                                target_found = True
                                self.target_start_time = current_time # 记录开始时间用于剪枝保护
                                self.finish_timer = 0.0 # 重置完成计时器
                                self.unknown_stable_timer = 0.0
                                self.last_unknown_cells = int(np.sum(self.local_map_data == -1))
                                # [新增] 记录本次探索目标开始时 Unknown 数量，用于判断是否有“探索进展”
                                self.explore_target_unknown0 = self.last_unknown_cells
                                self.explore_target_t0 = current_time
                                break                    # 探索结束判断 (去抖动逻辑 + 未知区域稳定判定)
                    if not target_found:
                        self.finish_timer += dt

                        # [新增] 未知区域稳定性：只有当 Unknown 数量在一段时间内基本不再下降，才允许结束
                        unknown_cells = int(np.sum(self.local_map_data == -1))
                        if self.last_unknown_cells is None:
                            self.last_unknown_cells = unknown_cells
                            self.unknown_stable_timer = 0.0
                        else:
                            # 如果 Unknown 明显减少，说明仍在拓展已知区域，重置稳定计时
                            if unknown_cells < self.last_unknown_cells - 50:
                                self.unknown_stable_timer = 0.0
                            else:
                                self.unknown_stable_timer += dt
                            self.last_unknown_cells = unknown_cells

                        # 只有当连续 5 秒找不到目标，且 Unknown 已稳定 >=5 秒，且已知区域足够大时，才认为结束
                        if self.finish_timer > 5.0 and self.unknown_stable_timer > 5.0 and np.sum(self.local_map_data == 0) > 300:
                            self.get_logger().info("✅ Exploration Finished (Confirmed). Switching to Topology.")
                            self.stop_robot()
                            self.current_state = self.STATE_GENERATING_TOPO
                            try:
                                self.mission_phase = self.PHASE_BUILD_TOPO
                                # 拓扑生成阶段暂停 live VL_search，避免打断流程
                                self.search_instruction = ""
                            except Exception:
                                pass
                        else:
                            self.action_description = f"Scanning... ({max(0.0, 5.0 - self.finish_timer):.1f}s)"
                # B. 拓扑生成模式 (Topology)
                elif self.current_state == self.STATE_GENERATING_TOPO:
                    self.action_description = "Building Topo..."
                    # 调用整合好的生成函数
                    if self.generate_topo_and_patrol_list():
                        self.current_state = self.STATE_PATROLLING
                        self.current_node_idx = 0
                        try:
                            self.mission_phase = self.PHASE_PATROL_BUILD_MEMORY
                            # 巡逻阶段仅做记忆构建：不做 VL_SEARCH
                            self.search_instruction = ""
                        except Exception:
                            pass
                
                # C. 巡逻模式 (Patrolling)
                elif self.current_state == self.STATE_PATROLLING:
                    if self.current_node_idx >= len(self.mission_nodes):
                        # 巡逻阶段完成：开始最终语义拓扑检索并等待 /audit_nav/result
                        if (getattr(self, 'mission_phase', None) == getattr(self, 'PHASE_PATROL_BUILD_MEMORY', -1)
                            and bool(getattr(self, 'primary_instruction', ''))):
                            self.action_description = "Patrol done -> FINAL_SEARCH_WAIT"
                            self.stop_robot()
                            try:
                                self.mission_phase = self.PHASE_FINAL_SEARCH_WAIT
                                self.current_state = self.STATE_FINAL_SEARCH_WAIT
                                self._mission_start_final_search(now=current_time, reason='patrol_done')
                            except Exception:
                                pass
                            return

                        self.action_description = "Mission Complete"
                        self.stop_robot()
                        return
                        
                    node = self.mission_nodes[self.current_node_idx]
                    self.action_description = f"Patrol Node {node['id']}"
                    
                    s_g = self.world_to_grid(rx, ry)
                    g_g = self.world_to_grid(node['coords'][0], node['coords'][1])
                    
                    path = self.planner.plan(self.local_map_data, s_g, g_g)
                    if path: 
                        self.target_point = node['coords']
                        self.current_path = self.nudge_path_to_center(path)
                        self.last_path_index = 0
                    else:
                        self.get_logger().warn(f"Cannot reach Node {node['id']}, skipping.")
                        # 🔥 核心修复：如果在最终检索阶段去不了该节点，必须上报失败让 VLM 重新决策！
                        if getattr(self, 'mission_phase', None) == getattr(self, 'PHASE_FINAL_SEARCH_ACTIVE', -1):
                            self._publish_vl_feedback("topo_node_unreachable", node_id=node['id'])
                            self.current_state = self.STATE_FINAL_SEARCH_WAIT
                        else:
                            # 只有在纯巡逻建图阶段，才允许盲目跳过
                            self.current_node_idx += 1

            # --- 4. 运动执行层 (Control Logic) ---
            if self.target_point:
                # 4.1 动态剪枝 (Dynamic Pruning)
                if self.current_state == self.STATE_EXPLORING and (current_time - self.target_start_time > 3.0):
                    self.check_validity_timer += dt
                    if self.check_validity_timer >= 0.5:
                        self.check_validity_timer = 0
                        r, c = self.world_to_grid(self.target_point[0], self.target_point[1])
                        
                        r_min, r_max = max(0, r-16), min(self.local_map_data.shape[0], r+16)
                        c_min, c_max = max(0, c-16), min(self.local_map_data.shape[1], c+16)
                        roi = self.local_map_data[r_min:r_max, c_min:c_max]
                        
                        if np.sum(roi == -1) < 10:
                            # 目标附近已无明显未知区域，说明该“前沿”已失效；加入黑名单避免反复选择
                            try:
                                self.add_unreachable_point(self.target_point)
                            except Exception:
                                pass
                            self.target_point = None; self.current_path = []; return

                # 🔥 [核心修复] 中途碰撞熔断机制 (带去抖动) 🔥
                dist_to_goal = math.hypot(self.target_point[0]-rx, self.target_point[1]-ry)
                path_exhausted = (self.last_path_index >= len(self.current_path) - 2)
                collision_level = collision_level_now
                # 👇 [新增优化] 脱困保护期 (Invincibility Frame) 👇
                # 如果机器人刚刚触发过后退脱困，给它 2 秒钟的“强行挤压”权限。
                # 暂时屏蔽极近距离的碰撞报警，相信 A* 给出的安全路径。
                if current_time < self.recovery_disable_until:
                    collision_level = 0 
                
                # 如果检测到危险，计数器 +1
                if collision_level == 2:
                    self.collision_persistence += 1
                else:
                    self.collision_persistence = 0 # 只要有一帧安全就重置
                # 如果检测到危险，计数器 +1
                if collision_level == 2:
                    self.collision_persistence += 1
                else:
                    self.collision_persistence = 0 # 只要有一帧安全就重置
                
                # 只有连续 5 帧 (0.5s) 都危险才熔断
                if self.collision_persistence > 3:
                    self.get_logger().warn("🚨 Persistent Collision Detected! Emergency Stop -> Back 0.2m then Re-plan.")
                    try:
                        self._count_collision_event(current_time, reason='persistent_collision', level=2, rx=rx, ry=ry)
                    except Exception:
                        pass
                    self.stop_robot()


                    # 若同一目标反复触发危险碰撞脱困，判不可达并退出，避免无尽循环
                    if self._maybe_abandon_goal_after_repeated_backing(current_time, reason="persistent_collision"):
                        return

                    # 不立即拉黑目标点：先尝试后退 0.2m 让机器人脱离墙根，再基于原目标点重规划
                    self.wall_backing_active = True
                    self.wall_back_start_xy = (rx, ry)
                    self.wall_back_turn_sign = self.choose_backing_turn_sign(pose)
                    self.wall_back_start_time = current_time
                    self.current_path = []
                    self.last_path_index = 0

                    self.speed_integral = 0.0
                    self.collision_persistence = 0
                    self.wall_collision_timer = 0.0
                    self.threshold_timer = 0.0
                    return

                # 4.3 停止条件 (正常到达)
                is_arrived = (dist_to_goal < self.ARRIVAL_DIST)
                is_blocked = (path_exhausted and not is_arrived)

                if is_arrived or is_blocked:
                    self.stop_robot()
                    
                    if is_blocked: 
                        self.add_unreachable_point(self.target_point)

                    # [新增] 探索阶段：到达目标但 Unknown 基本不减少 -> 认为无效前沿，加入黑名单并累计无进展次数
                    if self.current_state == self.STATE_EXPLORING and (is_arrived or is_blocked):
                        try:
                            unknown_now = int(np.sum(self.local_map_data == -1))
                            unknown0 = int(self.explore_target_unknown0) if self.explore_target_unknown0 is not None else unknown_now
                            delta_unknown = unknown0 - unknown_now
                            if delta_unknown < 30:
                                # 无进展：加入黑名单，避免反复选择同类前沿
                                self.add_unreachable_point(self.target_point)
                                self.explore_progress_fail_count += 1
                                self.get_logger().info(
                                    f"🧹 Frontier no-progress: Δunknown={delta_unknown}, count={self.explore_progress_fail_count}"
                                )
                            else:
                                # 有进展：清零无进展计数
                                self.explore_progress_fail_count = 0
                        except Exception as e:
                            self.get_logger().warn(f"[Explore] no-progress check failed: {e}")

                        # 若连续多次无进展且 Unknown 已稳定，则认为探索已完成，进入拓扑生成
                        if self.explore_progress_fail_count >= 5 and self.unknown_stable_timer > 5.0:
                            self.get_logger().info("✅ Exploration Finished (stagnant frontiers). Switching to Topology.")
                            self.target_point = None
                            self.current_path = []
                            self.speed_integral = 0.0
                            self.stop_robot()
                            self.current_state = self.STATE_GENERATING_TOPO
                            try:
                                self.mission_phase = self.PHASE_BUILD_TOPO
                                # 拓扑生成阶段暂停 live VL_search，避免打断流程
                                self.search_instruction = ""
                            except Exception:
                                pass
                            return
                    
    # 如果是在巡逻且到达，触发 VLM
                    if self.current_state == self.STATE_PATROLLING and is_arrived:
                        # [新增] 如果 commander 正在下发“查找目标”，则在 VLM 审核完成后执行一次本地 VL 搜索
                        self.pending_vl_search = bool(self.search_instruction) and (getattr(self,'mission_phase', None) == getattr(self,'PHASE_FINAL_SEARCH_ACTIVE', -1))

                        # ==========================================================
                        # 🔥 [Bug Fix 2] 每次触发新节点的审计前，强制清空上一局的“方向圣旨”和目标缓存！
                        # ==========================================================
                        self.vlm_authorized_view = ""
                        self.last_vlm_bbox = None
                       # self.deferred_vl_candidates = [] # 顺手清空上个节点残留的高分目标
                        # ==========================================================

                        msg = Int32()
                        msg.data = self.mission_nodes[self.current_node_idx]['id']
                        self.pub_vlm_trigger.publish(msg)
                        self.current_state = self.STATE_WAITING_FOR_VLM
                        self.get_logger().info(f"📍 Arrived Node {msg.data}. Waiting for VLM...")

                        # [新增] 防止 VLM 等待后起步误触发“门槛/卡死脱困”
                        self.stuck_timer = 0.0
                        self.last_target_speed = 0.0
                        self.force_recovery_state = "NONE"
                        self.recovery_disable_until = time.time() + 2.0

                    # [新增] 若当前是前往 VL 局部目标点：
                    # - 默认不做近距验证（self.enable_vl_verify=False）
                    # - 直接判定“到达即成功”，清空 search_instruction，并恢复到抢占前状态
                    if self.current_state == self.STATE_GO_TO_VL_GOAL and is_arrived:
                        if getattr(self, 'enable_vl_verify', True):
                            # 进入近距验证：原地旋转一小段时间，观察 conf 是否能稳定在 on 阈值附近
                            self.get_logger().info("🎯 Reached VL goal. Enter near-range verify scan...")
                            self.current_state = self.STATE_VL_VERIFY
                            self.vl_verify_start_time = time.time()
                            self.vl_verify_max_conf = float(self.vl_best_conf) if (time.time() - self.vl_conf_time) < self.vl_conf_fresh_sec else 0.0
                            self.vl_verify_goal = tuple(self.target_point) if self.target_point is not None else None
                            self.vl_verify_instruction = str(self.search_instruction)
                            self.vl_verify_return_state = self.vl_return_state if self.vl_return_state is not None else self.STATE_EXPLORING

                            # 清理抢占状态（验证结束后再恢复）
                            self.vl_return_state = None
                            self.advance_after_vl_search = False

                            # 防止状态切换瞬间误触发恢复逻辑
                            self.stuck_timer = 0.0
                            self.last_target_speed = 0.0
                            self.force_recovery_state = "NONE"
                            self.recovery_disable_until = time.time() + 2.0

                            # 终止当前路径跟踪，让 STATE_VL_VERIFY 接管速度输出
                            self.target_point = None
                            self.current_path = []
                            self.speed_integral = 0.0
                            return

                        # ✅ 不启用近距验证：到达即成功
                        self.stop_robot()
                        instr = str(self.search_instruction)
                        self.get_logger().info(f"✅ SUCCESS! [{instr}] reached. Target acquired!")

                        # [Eval] reach event: use integrated goal point (target_point) as the arrival marker
                        try:
                            phase = getattr(self, 'mission_phase', None)
                            kind = 'reach'
                            if phase == getattr(self, 'PHASE_EXPLORE_GUIDED', -1):
                                kind = 'soft_found'
                                if not bool(getattr(self, '_eval_soft_found_pub', False)):
                                    self._publish_reach_event(kind, current_time, rx, ry, goal_xy=getattr(self, 'target_point', None), instr=str(self.search_instruction))
                                    self._eval_soft_found_pub = True
                            elif phase == getattr(self, 'PHASE_FINAL_SEARCH_ACTIVE', -1):
                                kind = 'final_found'
                                if not bool(getattr(self, '_eval_final_found_pub', False)):
                                    self._publish_reach_event(kind, current_time, rx, ry, goal_xy=getattr(self, 'target_point', None), instr=str(self.search_instruction))
                                    self._eval_final_found_pub = True
                            else:
                                self._publish_reach_event(kind, current_time, rx, ry, goal_xy=getattr(self, 'target_point', None), instr=str(self.search_instruction))
                        except Exception:
                            pass


                        # [High-level] 到旁边即可算找到：探索阶段找到一次后继续探索；最终检索阶段找到则结束
                        try:
                            if getattr(self, 'mission_phase', None) == getattr(self, 'PHASE_EXPLORE_GUIDED', -1):
                                self.soft_found = True
                            elif getattr(self, 'mission_phase', None) == getattr(self, 'PHASE_FINAL_SEARCH_ACTIVE', -1):
                                self.final_found = True
                                self.mission_phase = self.PHASE_DONE
                        except Exception:
                            pass

                        # 清空查找指令，避免继续被其它误检/目标打断
                        self.search_instruction = ""
                        self.vl_search_origin_node_id = None

                        # 恢复到抢占前状态（探索 / 巡逻）。若已进入 DONE，则切到 FINAL_SEARCH_WAIT 仅用于维持停机状态
                        if getattr(self, 'mission_phase', None) == getattr(self, 'PHASE_DONE', -999):
                            ret_state = self.STATE_FINAL_SEARCH_WAIT
                        else:
                            ret_state = self.vl_return_state if self.vl_return_state is not None else self.STATE_EXPLORING
                        self.current_state = ret_state
                        self.vl_return_state = None
                        self.advance_after_vl_search = False

                        # 防止刚切回状态时误触发恢复逻辑
                        self.stuck_timer = 0.0
                        self.last_target_speed = 0.0
                        self.force_recovery_state = "NONE"
                        self.recovery_disable_until = time.time() + 2.0

                        # 清理路径跟踪
                        self.target_point = None
                        self.current_path = []
                        self.speed_integral = 0.0
                        return
                    self.target_point = None
                    self.current_path = []
                    self.speed_integral = 0.0
                    return
                
                # 4.4 PID 路径跟踪 (Smart-V2 原始逻辑内联)
                la = self.MAX_LOOKAHEAD if self.is_in_rush_mode else self.MIN_LOOKAHEAD
                self.current_lookahead_val = 0.9 * self.current_lookahead_val + 0.1 * la
                
                closest = self.last_path_index
                min_d = float('inf')
                search_end = min(len(self.current_path), self.last_path_index + 20)
                for i in range(self.last_path_index, search_end):
                    n = self.current_path[i]
                    wx, wy = self.grid_to_world(n[0], n[1])
                    d = math.hypot(wx-rx, wy-ry)
                    if d < min_d: 
                        min_d = d
                        closest = i
                self.last_path_index = closest
                target_idx = closest
                for i in range(closest, len(self.current_path)):
                    n = self.current_path[i]
                    wx, wy = self.grid_to_world(n[0], n[1])
                    if math.hypot(wx-rx, wy-ry) > self.current_lookahead_val:
                        target_idx = i
                        break
                n = self.current_path[target_idx]
                twx, twy = self.grid_to_world(n[0], n[1])
                alpha = normalize_angle(math.atan2(twy-ry, twx-rx) - ryaw)
                
                cmd = Twist()
                
                if self.is_in_rush_mode:
                    cmd.linear.x = self.RUSH_SPEED 
                    cmd.angular.z = float(np.clip(alpha * 1.5, -0.4, 0.4)) 
                    self.speed_integral = 0.0 
                else:
                    

                    # 保留原有 stuck_timer 变量（但此处不再使用它触发后退+猛冲逻辑）
                    self.stuck_timer = 0.0
                    alpha_rate = (alpha - self.last_alpha) / dt
                    curvature = 2.0 * math.sin(alpha) / self.current_lookahead_val
                    steer_cmd = (self.BASE_SPEED * curvature * self.kp_yaw) + (alpha_rate * self.kd_yaw)
                    cmd.angular.z = float(np.clip(steer_cmd, -self.MAX_ROTATION_SPEED, self.MAX_ROTATION_SPEED))
                    
                    speed_factor = 1.0 - min(abs(alpha) / 1.0, 1.0)
                    decay_speed = self.BASE_SPEED * speed_factor
                    
                    if abs(alpha) < 1.0:
                        target_speed_base = max(decay_speed, self.MIN_SPEED_FLOOR)
                    else:
                        target_speed_base = 0.0 
                    
                    self.last_target_speed = target_speed_base
                    speed_error = target_speed_base - self.current_linear_velocity
                    
                    if target_speed_base > 0.1 and speed_error > 0.05 and abs(alpha) < 0.5:
                        self.speed_integral += speed_error * dt
                    else:
                        self.speed_integral = 0.0
                    self.speed_integral = max(min(self.speed_integral, 0.5), 0.0)
                    speed_derivative = (speed_error - self.last_speed_error) / dt
                    final_speed = (self.kp_speed * target_speed_base) + (self.ki_speed * self.speed_integral) + (self.kd_speed * speed_derivative)
                    cmd.linear.x = min(max(final_speed, 0.0), self.MAX_SPEED_CAP)
                    self.last_speed_error = speed_error
                    self.last_alpha = alpha
                
                self.last_control_time = current_time
                self.pub_vel.publish(cmd)

        except Exception as e:
            self.get_logger().error(f"Loop Error: {e}")

    def visualize_dashboard(self):
        """CV2 可视化仪表盘：同时提供
        1) 全局地图（自动裁剪到已知区域并居中）
        2) 以机器人为中心的局部窗口（跟随居中）

        说明：
        - 这只是可视化，不影响节点生成/路径规划（仍使用完整 self.local_map_data）。
        - 当地图很大时，全局视图会自动缩放；局部视图用于近距离观察可达性。
        """
        if self.local_map_data is None:
            return

        grid = self.local_map_data

        def _grid_to_bgr(g):
            vis = np.zeros_like(g, dtype=np.uint8)
            vis[g == 0] = 255
            vis[g == -1] = 128
            vis[g > 50] = 0
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

            # unsafe 可视化（仅用于显示，参数不改变控制逻辑）
            try:
                safe_r = int(0.3 / 0.05)
                obs = (g > 50).astype(np.uint8) * 255
                unsafe = cv2.dilate(obs, np.ones((safe_r*2+1, safe_r*2+1), np.uint8))
                vis[unsafe > 0] = (vis[unsafe > 0] * 0.7 + np.array([0, 0, 255]) * 0.3).astype(np.uint8)
            except Exception:
                pass
            return vis

        def _pad_to_square(img, pad_val=(0, 0, 0)):
            h, w = img.shape[:2]
            if h == w:
                return img
            if h > w:
                pad = (h - w) // 2
                return cv2.copyMakeBorder(img, 0, 0, pad, h - w - pad, cv2.BORDER_CONSTANT, value=pad_val)
            else:
                pad = (w - h) // 2
                return cv2.copyMakeBorder(img, pad, w - h - pad, 0, 0, cv2.BORDER_CONSTANT, value=pad_val)

        # -------------------- 1) 全局视图：裁剪已知区域后居中 --------------------
        global_bgr = _grid_to_bgr(grid)
        try:
            known = (grid != -1)
            if known.any():
                ys, xs = np.where(known)
                r0, r1 = int(ys.min()), int(ys.max())
                c0, c1 = int(xs.min()), int(xs.max())
                margin = 10
                r0 = max(0, r0 - margin)
                c0 = max(0, c0 - margin)
                r1 = min(grid.shape[0] - 1, r1 + margin)
                c1 = min(grid.shape[1] - 1, c1 + margin)
                global_bgr = global_bgr[r0:r1+1, c0:c1+1]
        except Exception:
            pass

        # 在全局视图上画路径/目标/机器人
        try:
            for pt in getattr(self, 'unreachable_points', []):
                gp = self.world_to_grid(pt[0], pt[1])
                if gp[0] is not None and 0 <= gp[0] < grid.shape[0] and 0 <= gp[1] < grid.shape[1]:
                    cv2.circle(global_bgr, (gp[1], gp[0]), 5, (0, 0, 255), 1)
        except Exception:
            pass

        try:
            if self.current_path:
                for i in range(len(self.current_path) - 1):
                    p1 = self.current_path[i]
                    p2 = self.current_path[i+1]
                    cv2.line(global_bgr, (p1[1], p1[0]), (p2[1], p2[0]), (0, 255, 0), 1)
        except Exception:
            pass

        pose = self.get_robot_pose()
        if pose:
            rx, ry, ryaw, _ = pose
            gp = self.world_to_grid(rx, ry)
            if gp[0] is not None:
                cv2.circle(global_bgr, (gp[1], gp[0]), 6, (255, 0, 0), -1)
                arrow = 25
                ex = int(gp[1] + arrow * math.cos(ryaw))
                ey = int(gp[0] + arrow * math.sin(ryaw))
                cv2.arrowedLine(global_bgr, (gp[1], gp[0]), (ex, ey), (0, 0, 255), 2, tipLength=0.3)
                if self.target_point:
                    tgp = self.world_to_grid(self.target_point[0], self.target_point[1])
                    if tgp[0] is not None:
                        cv2.circle(global_bgr, (tgp[1], tgp[0]), 8, (0, 255, 0), 2)

        global_bgr = cv2.flip(global_bgr, 0)
        global_bgr = _pad_to_square(global_bgr, pad_val=(0, 0, 0))
        global_bgr = cv2.resize(global_bgr, (600, 600), interpolation=cv2.INTER_NEAREST)

        # -------------------- 2) 局部视图：始终以机器人为中心 --------------------
        local_bgr = None
        try:
            if pose and self.map_info is not None:
                rx, ry, ryaw, _ = pose
                gp = self.world_to_grid(rx, ry)
                if gp[0] is not None:
                    # 8m x 8m 的局部窗口（仅显示），分辨率来自地图
                    res = float(getattr(self.map_info, 'resolution', 0.05))
                    win_m = 8.0
                    half = max(20, int((win_m / res) / 2.0))
                    r0 = gp[0] - half
                    r1 = gp[0] + half
                    c0 = gp[1] - half
                    c1 = gp[1] + half

                    # 边界裁剪 + pad
                    pad_top = max(0, -r0)
                    pad_left = max(0, -c0)
                    pad_bottom = max(0, r1 - (grid.shape[0] - 1))
                    pad_right = max(0, c1 - (grid.shape[1] - 1))

                    rr0 = max(0, r0)
                    rr1 = min(grid.shape[0] - 1, r1)
                    cc0 = max(0, c0)
                    cc1 = min(grid.shape[1] - 1, c1)

                    local_crop = grid[rr0:rr1+1, cc0:cc1+1]
                    if any([pad_top, pad_bottom, pad_left, pad_right]):
                        local_crop = np.pad(
                            local_crop,
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='constant',
                            constant_values=-1
                        )

                    local_bgr = _grid_to_bgr(local_crop)

                    # 在局部图上画机器人（位于中心）
                    center_r = local_crop.shape[0] // 2
                    center_c = local_crop.shape[1] // 2
                    cv2.circle(local_bgr, (center_c, center_r), 6, (255, 0, 0), -1)
                    arrow = 25
                    ex = int(center_c + arrow * math.cos(ryaw))
                    ey = int(center_r + arrow * math.sin(ryaw))
                    cv2.arrowedLine(local_bgr, (center_c, center_r), (ex, ey), (0, 0, 255), 2, tipLength=0.3)

                    # 局部视图上如果有目标点，画出相对位置
                    if self.target_point:
                        tgp = self.world_to_grid(self.target_point[0], self.target_point[1])
                        if tgp[0] is not None:
                            tr = int(tgp[0] - rr0 + pad_top)
                            tc = int(tgp[1] - cc0 + pad_left)
                            if 0 <= tr < local_crop.shape[0] and 0 <= tc < local_crop.shape[1]:
                                cv2.circle(local_bgr, (tc, tr), 8, (0, 255, 0), 2)

                    local_bgr = cv2.flip(local_bgr, 0)
                    local_bgr = _pad_to_square(local_bgr, pad_val=(0, 0, 0))
                    local_bgr = cv2.resize(local_bgr, (600, 600), interpolation=cv2.INTER_NEAREST)
        except Exception:
            local_bgr = None

        if local_bgr is None:
            local_bgr = np.zeros((600, 600, 3), dtype=np.uint8)
            cv2.putText(local_bgr, "Local view unavailable", (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # -------------------- 顶部状态面板 --------------------
        panel = np.zeros((110, 1200, 3), dtype=np.uint8)
        cv2.putText(panel, f"State: {self.action_description}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(panel, f"Mode: {self.force_recovery_state}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1)
        cv2.putText(panel, f"V: {self.current_linear_velocity:.2f}", (520, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, "Global (auto-fit) | Local (robot-centered)", (520, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        body = np.hstack((global_bgr, local_bgr))
        final_frame = np.vstack((panel, body))
        cv2.imshow("VL-Nav Dashboard", np.vstack((panel, body)))
        cv2.waitKey(1)
# ================== 🔥 新增：写入视频帧 ==================
        if hasattr(self, 'video_writer') and self.video_writer.isOpened():
            self.video_writer.write(final_frame)
    def stop_robot(self):
        self.pub_vel.publish(Twist())

# --------------------------
# [Eval] Evaluation event publishing helpers
# --------------------------
    def _publish_eval_event(self, payload: dict):
        """Publish JSON payload to /audit_nav/eval_event. Fail silently to avoid affecting control."""
        try:
            if not hasattr(self, "pub_eval_event") or self.pub_eval_event is None:
                return
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.pub_eval_event.publish(msg)
        except Exception:
            pass

    def _count_collision_event(self, t: float, reason: str, level: int, rx: float, ry: float):
        """Count collision/unsafe contacts (C2) with debounce; publish one eval_event per collision."""
        try:
            last_t = float(getattr(self, "_eval_last_collision_time", 0.0))
            deb = float(getattr(self, "_eval_collision_debounce_sec", 0.8))
            if (t - last_t) < deb:
                return
            self._eval_last_collision_time = float(t)
            self.eval_collision_count = int(getattr(self, "eval_collision_count", 0)) + 1
            payload = {
                "event": "collision",
                "t": float(t),
                "reason": str(reason),
                "level": int(level),
                "count": int(self.eval_collision_count),
                "pose_xy": [float(rx), float(ry)],
                "state": int(getattr(self, "current_state", -1)),
                "phase": int(getattr(self, "mission_phase", -1)),
            }
            self._publish_eval_event(payload)
        except Exception:
            pass

    def _publish_reach_event(self, kind: str, t: float, rx: float, ry: float, goal_xy=None, instr: str = ""):
        """Publish reach event when integrated认为已到达目标点（而不是GT附近判定）."""
        try:
            gx = None; gy = None
            if goal_xy is not None and isinstance(goal_xy, (list, tuple)) and len(goal_xy) >= 2:
                gx = float(goal_xy[0]); gy = float(goal_xy[1])
            payload = {
                "event": "reach",
                "kind": str(kind),   # 'soft_found' or 'final_found' (or 'reach')
                "t": float(t),
                "instruction": str(instr),
                "goal_xy": [gx, gy] if gx is not None and gy is not None else None,
                "pose_xy": [float(rx), float(ry)],
                "state": int(getattr(self, "current_state", -1)),
                "phase": int(getattr(self, "mission_phase", -1)),
                "conf": float(getattr(self, "vl_best_conf", 0.0)),
            }
            # best-effort: include current topo node id if available
            try:
                if hasattr(self, "mission_nodes") and self.mission_nodes and hasattr(self, "current_node_idx"):
                    nid = int(self.mission_nodes[self.current_node_idx].get("id", -1))
                    payload["node_id"] = nid
            except Exception:
                pass
            self._publish_eval_event(payload)
        except Exception:
            pass

    def _publish_mission_done_event(self, t: float, rx: float, ry: float):
        try:
            payload = {
                "event": "mission_done",
                "t": float(t),
                "instruction": str(getattr(self, "primary_instruction", "")),
                "pose_xy": [float(rx), float(ry)],
                "soft_found": bool(getattr(self, "soft_found", False)),
                "final_found": bool(getattr(self, "final_found", False)),
                "state": int(getattr(self, "current_state", -1)),
                "phase": int(getattr(self, "mission_phase", -1)),
                "collisions": int(getattr(self, "eval_collision_count", 0)),
            }
            try:
                if hasattr(self, "mission_nodes") and self.mission_nodes and hasattr(self, "current_node_idx"):
                    nid = int(self.mission_nodes[self.current_node_idx].get("id", -1))
                    payload["node_id"] = nid
            except Exception:
                pass
            self._publish_eval_event(payload)
        except Exception:
            pass

    def destroy_node(self):
    # ================== 🔥 新增：安全保存录像 ==================
        if hasattr(self, 'video_writer') and self.video_writer.isOpened():
            self.video_writer.release()
            self.get_logger().info(f"💾 Dashboard recording saved to: {self.video_path}")
    # ===============================================================
    
        cv2.destroyAllWindows()
        super().destroy_node()


# =========================

# =========================
# VL Belief / Hysteresis / Verify helpers (bound to class below)
# =========================
    def _vl_prune_cooldowns(self, now=None):
        """清理过期冷却点。"""
        try:
            if now is None:
                now = time.time()
            cds = getattr(self, "vl_cooldowns", None)
            if cds is None:
                return
            maxlen = getattr(cds, "maxlen", 100)
            alive = [c for c in list(cds) if float(c.get("until", 0.0)) > now]
            from collections import deque
            self.vl_cooldowns = deque(alive, maxlen=maxlen)
        except Exception:
            pass

    def _vl_add_cooldown(self, pt_world, seconds: float, reason: str = ""):
        """添加/合并冷却点（用 BLACKLIST_RADIUS 做空间合并）。"""
        try:
            if pt_world is None:
                return
            now = time.time()
            self._vl_prune_cooldowns(now)

            wx, wy = float(pt_world[0]), float(pt_world[1])
            until = now + float(seconds)

            cds = getattr(self, "vl_cooldowns", None)
            if cds is None:
                from collections import deque
                cds = deque(maxlen=100)
                self.vl_cooldowns = cds

            radius = float(getattr(self, "BLACKLIST_RADIUS", 0.6))
            radius2 = radius * radius

            merged = False
            new_list = []
            for c in list(cds):
                cx, cy = c.get("pt", (None, None))
                if cx is None:
                    continue
                dx, dy = float(cx) - wx, float(cy) - wy
                if (dx*dx + dy*dy) <= radius2:
                    c["until"] = max(float(c.get("until", 0.0)), until)
                    if reason:
                        c["reason"] = reason
                    merged = True
                new_list.append(c)

            if not merged:
                new_list.append({"pt": (wx, wy), "until": until, "reason": reason})

            from collections import deque
            self.vl_cooldowns = deque(new_list, maxlen=getattr(cds, "maxlen", 100))
        except Exception:
            pass

    def _vl_is_on_cooldown(self, wx: float, wy: float, now=None) -> bool:
        """判断某个 world 点是否处在冷却半径内。"""
        try:
            if now is None:
                now = time.time()
            self._vl_prune_cooldowns(now)
            cds = getattr(self, "vl_cooldowns", None)
            if cds is None:
                return False

            radius = float(getattr(self, "BLACKLIST_RADIUS", 0.6))
            radius2 = radius * radius

            for c in list(cds):
                until = float(c.get("until", 0.0))
                if until <= now:
                    continue
                cx, cy = c.get("pt", (None, None))
                if cx is None:
                    continue
                dx, dy = float(cx) - float(wx), float(cy) - float(wy)
                if (dx*dx + dy*dy) <= radius2:
                    return True
            return False
        except Exception:
            return False

    def _vl_update_belief_hys(self, conf: float, now=None):
        """更新 belief + 迟滞计数。"""
        try:
            if now is None:
                now = time.time()
            c = float(conf)

            alpha = float(getattr(self, "vl_belief_alpha", 0.2))
            self.vl_belief = (1.0 - alpha) * float(getattr(self, "vl_belief", 0.0)) + alpha * c

            th_on = float(getattr(self, "vl_th_on", getattr(self, "vl_preempt_min_conf", 0.55)))
            th_off = float(getattr(self, "vl_th_off", max(0.0, th_on - 0.12)))

            if c >= th_on:
                self.vl_on_count = int(getattr(self, "vl_on_count", 0)) + 1
            else:
                self.vl_on_count = 0

            if c <= th_off:
                self.vl_off_count = int(getattr(self, "vl_off_count", 0)) + 1
            else:
                self.vl_off_count = 0
        except Exception:
            pass

    def _vl_should_preempt(self, current_time=None) -> bool:
        """是否允许从探索/巡逻抢占去 VL 目标。"""
        try:
            if current_time is None:
                current_time = time.time()
            fresh_sec = float(getattr(self, "vl_conf_fresh_sec", 0.6))
            if (current_time - float(getattr(self, "vl_conf_time", 0.0))) > fresh_sec:
                return False
            k_on = int(getattr(self, "vl_k_on", 3))
            if int(getattr(self, "vl_on_count", 0)) < k_on:
                return False
            return True
        except Exception:
            return False


    def remember_semantic_hint(self, pt_world, conf: float = 0.0, source: str = "vl_soft"):
        """缓存“软语义证据”（中等置信度目标位置），用于探索阶段对 frontier 做重排序。"""
        try:
            now = time.time()
            conf = float(conf)
            min_conf = float(getattr(self, "semantic_hint_min_conf", 0.30))
            if conf < min_conf:
                return
            self._semantic_prune_hints(now)
            pt = (float(pt_world[0]), float(pt_world[1]))
            # 若距离已有 hint 很近，则合并（取更高置信度、更近的时间）
            merge_r = float(getattr(self, "BLACKLIST_RADIUS", 0.6))
            merge_r2 = merge_r * merge_r
            hints = getattr(self, "semantic_hints", None)
            if hints is None:
                from collections import deque
                hints = deque(maxlen=300)
                self.semantic_hints = hints
            merged = False
            new_list = []
            for h in list(hints):
                hx, hy = h.get("pt", (None, None))
                if hx is None:
                    continue
                dx, dy = float(hx) - pt[0], float(hy) - pt[1]
                if (dx*dx + dy*dy) <= merge_r2:
                    h["t"] = now
                    h["conf"] = max(float(h.get("conf", 0.0)), conf)
                    if source:
                        h["source"] = source
                    merged = True
                new_list.append(h)
            if not merged:
                new_list.append({"pt": pt, "t": now, "conf": conf, "source": source})
            from collections import deque
            self.semantic_hints = deque(new_list, maxlen=getattr(hints, "maxlen", 300))
        except Exception:
            pass


    def _semantic_prune_hints(self, now=None):
        """清理过期 soft hints。"""
        try:
            if now is None:
                now = time.time()
            max_age = float(getattr(self, "semantic_hint_max_age_sec", 120.0))
            hints = getattr(self, "semantic_hints", None)
            if hints is None:
                return
            kept = []
            for h in list(hints):
                t = float(h.get("t", 0.0))
                if (now - t) <= max_age:
                    kept.append(h)
            from collections import deque
            self.semantic_hints = deque(kept, maxlen=getattr(hints, "maxlen", 300))
        except Exception:
            pass


    def _semantic_score_at(self, wx: float, wy: float, now=None) -> float:
        """计算某个 world 点的语义偏好分数（来自 soft hints）。

        设计：\n
        - hints 随时间淘汰；\n
        - 分数为 \u2211 conf * exp(-d/sigma)。
        """
        try:
            if now is None:
                now = time.time()
            self._semantic_prune_hints(now)
            hints = getattr(self, "semantic_hints", None)
            if not hints:
                return 0.0
            sigma = float(getattr(self, "semantic_hint_sigma_m", 2.5))
            if sigma <= 1e-6:
                sigma = 2.5
            inv_sigma = 1.0 / sigma
            score = 0.0
            for h in list(hints):
                hx, hy = h.get("pt", (None, None))
                if hx is None:
                    continue
                dx, dy = float(hx) - float(wx), float(hy) - float(wy)
                d = math.sqrt(dx*dx + dy*dy)
                # 5*sigma 之外影响极小，直接忽略
                if d > 5.0 * sigma:
                    continue
                conf = float(h.get("conf", 0.0))
                score += conf * math.exp(-d * inv_sigma)
            return float(score)
        except Exception:
            return 0.0


    def _mission_start_final_search(self, now=None, reason: str = "") -> bool:
        """发起最终检索：将 primary_instruction 重新发布到 /audit_nav/instruction，等待 /audit_nav/result。"""
        try:
            if not getattr(self, "primary_instruction", ""):
                return False
            if now is None:
                now = time.time()
            # 节流：避免每个 control_loop 都刷屏发布
            last = float(getattr(self, "final_search_last_issue_time", 0.0))
            if (now - last) < 1.0 and int(getattr(self, "final_search_reissue_count", 0)) > 0:
                return True

            self.search_instruction = str(self.primary_instruction)
            self.search_last_instruction = str(self.primary_instruction)
            self.search_failed_node_ids = set()
            self.vl_search_origin_node_id = None

            self.waiting_for_vlm_search = True
            self.final_search_request_time = float(now)
            self.final_search_last_issue_time = float(now)
            self.final_search_reissue_count = int(getattr(self, "final_search_reissue_count", 0)) + 1

            try:
                msg = String()
                msg.data = str(self.primary_instruction)
                # 发布到同名 topic，保持对外接口完全兼容（外部 VLN/Reasoner 监听该 topic）
                if hasattr(self, "pub_instruction") and self.pub_instruction is not None:
                    self.pub_instruction.publish(msg)
            except Exception:
                pass

            self.get_logger().info(f"🔎 Final search request issued (reason={reason}) query=[{self.primary_instruction}]")
            return True
        except Exception:
            return False



def main(args=None):
    rclpy.init(args=args)
    node = TopoNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
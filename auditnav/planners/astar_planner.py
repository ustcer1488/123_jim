import numpy as np
import heapq
import cv2
import math

class AStarPlanner:
    def __init__(self, resolution=0.05, robot_radius=0.2, allow_unknown=True, gray_penalty=5.0):
        """
        :param resolution: Map resolution (m/pixel)
        :param robot_radius: Robot physical radius (m)
        :param allow_unknown: Whether to allow traversal through unknown regions
        :param gray_penalty: Extra movement cost for unknown (gray) cells
        """
        self.resolution = resolution
        self.robot_radius_px = int((robot_radius + 0.05) / resolution)
        self.allow_unknown = allow_unknown
        self.gray_penalty = gray_penalty
        self.cost_map_debug = None

    def plan(self, map_data, start, goal):
        """
        A* path planning with repulsive potential field (wall-avoidance + gray penalty).
        """
        rows, cols = map_data.shape

        # === 1. Build cost map ===
        if self.allow_unknown:
            obs_mask = (map_data > 60).astype(np.uint8)
        else:
            obs_mask = (map_data != 0).astype(np.uint8)

        binary_map = np.ones_like(map_data, dtype=np.uint8) * 255
        binary_map[obs_mask > 0] = 0

        dist_map = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)

        # === 2. Hard inflation ===
        collision_mask = dist_map < self.robot_radius_px

        # === 3. Potential field cost ===
        safe_dist = 9
        cost_field = np.zeros_like(dist_map, dtype=np.float32)
        mask_cost = (dist_map > 0) & (dist_map < safe_dist)
        if np.any(mask_cost):
            cost_field[mask_cost] = 40.0 * (1.0 / dist_map[mask_cost])

        final_cost_map = 1.0 + cost_field

        # === 4. Gray penalty ===
        if self.allow_unknown:
            gray_mask = (map_data == -1)
            final_cost_map[gray_mask] += self.gray_penalty

        final_cost_map[collision_mask] = float('inf')
        self.cost_map_debug = final_cost_map

        # === 5. Validity check for start/goal ===
        if final_cost_map[start[0], start[1]] == float('inf'):
            valid_start = self.find_nearest_valid(final_cost_map, start)
            if valid_start:
                start = valid_start
            else:
                return []

        if final_cost_map[goal[0], goal[1]] == float('inf'):
            valid_goal = self.find_nearest_valid(final_cost_map, goal)
            if valid_goal:
                goal = valid_goal
            else:
                return []

        # === 6. A* search ===
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        movements = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        move_costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]

        while open_set:
            current_cost, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for i, move in enumerate(movements):
                neighbor = (current[0] + move[0], current[1] + move[1])

                if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                    continue

                step_cost = final_cost_map[neighbor[0], neighbor[1]]
                if step_cost == float('inf'):
                    continue

                tentative_g_score = g_score[current] + move_costs[i] * step_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return []

    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def find_nearest_valid(self, cost_map, pt, max_radius=10):
        r, c = pt
        rows, cols = cost_map.shape
        for radius in range(1, max_radius + 1):
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    nr, nc = r + i, c + j
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if cost_map[nr, nc] != float('inf'):
                            return (nr, nc)
        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

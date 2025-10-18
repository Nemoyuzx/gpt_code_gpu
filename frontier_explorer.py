from collections import deque
import heapq
import math
import numpy as np

class FrontierExplorer:
    """前沿探索与路径规划模块。确定下一个目标前沿并规划路径。"""
    def __init__(self, safety_distance=1.0):
        """
        初始化前沿探索器。
        
        参数:
        - safety_distance: 路径规划时与障碍物保持的安全距离（栅格单位）
        """
        self.safety_distance = float(safety_distance)
        # 前沿缓存
        self._cached_frontiers = None
        self._cached_occupancy_hash = None

    def set_safety_distance(self, safety_distance: float) -> None:
        """Update the default safety distance (in grid units)."""
        self.safety_distance = float(max(0.0, safety_distance))

    def _cluster_frontiers(self, frontiers, cluster_radius=3):
        """
        将前沿点聚类，每个簇选一个代表点。
        
        参数:
        - frontiers: 前沿点列表 [(x, y), ...]
        - cluster_radius: 聚类半径（栅格单位）
        
        返回: 聚类后的代表点列表
        """
        if not frontiers:
            return []
        
        frontiers = list(frontiers)
        clusters = []
        used = set()
        
        for i, (fx, fy) in enumerate(frontiers):
            if i in used:
                continue
            
            # 创建新簇
            cluster = [(fx, fy)]
            used.add(i)
            
            # 找到所有邻近的前沿点
            for j, (ox, oy) in enumerate(frontiers):
                if j in used:
                    continue
                dist = math.sqrt((fx - ox)**2 + (fy - oy)**2)
                if dist <= cluster_radius:
                    cluster.append((ox, oy))
                    used.add(j)
            
            # 计算簇的质心作为代表点
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            # 找到最接近质心的实际前沿点
            rep = min(cluster, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
            clusters.append(rep)
        
        return clusters

    def find_frontiers(self, occupancy):
        """
        查找所有前沿单元（frontier）。前沿定义为：已知空闲且邻接未知区域的栅格。
        返回前沿单元列表，每个为(tuple: (x_idx, y_idx))。
        """
        frontiers = []
        h, w = occupancy.shape
        # 遍历每个栅格
        for j in range(h):
            for i in range(w):
                if occupancy[j, i] == 0:  # 空闲
                    # 检查邻居是否存在未知栅格
                    frontier = False
                    for dj in [-1, 0, 1]:
                        for di in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            nj = j + dj
                            ni = i + di
                            if 0 <= nj < h and 0 <= ni < w:
                                if occupancy[nj, ni] == -1:
                                    frontier = True
                                    break
                        if frontier:
                            break
                    if frontier:
                        frontiers.append((i, j))
        return frontiers

    def find_nearest_frontier(self, occupancy, start):
        """
        从start出发，使用多策略方法找到最近的前沿单元及路径（优化版本）。
        
        优化策略：
        1. 局部BFS搜索（快速）- 扩大搜索限制
        2. 全局前沿检测 + 聚类（减少候选数量）
        3. 限制路径规划次数
        
        参数:
        - occupancy: 占用栅格地图
        - start: (x_idx, y_idx) 起点栅格索引
        
        返回:
        - (frontier_cell, path): 找到时返回目标前沿和路径
        - (None, None): 未找到前沿时返回
        """
        h, w = occupancy.shape
        sx, sy = start

        if not (0 <= sx < w and 0 <= sy < h):
            return None, None

        # 策略1: 局部BFS搜索（保持原有逻辑，较快）
        visited = [[False]*w for _ in range(h)]
        parent = {}
        dq = deque()
        dq.append((sx, sy))
        visited[sy][sx] = True
        parent[(sx, sy)] = None
        bfs_limit = min(w * h, 10000)  # 扩大BFS范围限制（5000->10000）
        bfs_count = 0
        
        while dq and bfs_count < bfs_limit:
            x, y = dq.popleft()
            bfs_count += 1

            if occupancy[y, x] == 0:
                neighbor_unknown = any(
                    0 <= nx < w and 0 <= ny < h and occupancy[ny, nx] == -1
                    for nx, ny in [
                        (x-1, y), (x+1, y), (x, y-1), (x, y+1),
                        (x-1, y-1), (x+1, y+1), (x-1, y+1), (x+1, y-1)
                    ]
                )

                if neighbor_unknown and (x, y) != (sx, sy):
                    if self._is_safe(occupancy, x, y, safety_distance=self.safety_distance):
                        path = []
                        cur = (x, y)
                        while cur is not None:
                            path.append(cur)
                            cur = parent[cur]
                        path.reverse()
                        return (x, y), path

            # BFS扩展
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1),
                          (-1,-1), (1,1), (-1,1), (1,-1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h and 
                    not visited[ny][nx] and
                    (occupancy[ny, nx] == 0 or occupancy[ny, nx] == -1)):
                    
                    if not self._is_safe(occupancy, nx, ny, safety_distance=max(0.5, self.safety_distance)):
                        continue
                    
                    if abs(dx) == 1 and abs(dy) == 1:
                        if (occupancy[y, x + dx] == 1) or (occupancy[y + dy, x] == 1):
                            continue
                    
                    visited[ny][nx] = True
                    parent[(nx, ny)] = (x, y)
                    dq.append((nx, ny))

        # 策略2: 全局前沿检测 + 简化的聚类优化
        frontiers = self._find_all_frontiers(occupancy)
        if not frontiers:
            return None, None

        # 简化聚类：只在前沿点很多时才聚类
        if len(frontiers) > 100:
            clustered_frontiers = self._cluster_frontiers(frontiers, cluster_radius=5)
        else:
            clustered_frontiers = frontiers
        
        origin = (sx, sy)
        # 按启发式距离排序
        clustered_frontiers = sorted(clustered_frontiers, key=lambda cell: self._heuristic(origin, cell))
        
        best_frontier = None
        best_path = None
        min_dist = float('inf')
        max_candidates = 50  # 进一步减少候选数量（80->50）

        for idx, frontier in enumerate(clustered_frontiers):
            if idx >= max_candidates:
                break
            heuristic_lower = self._heuristic(origin, frontier)
            if heuristic_lower >= min_dist:
                break
            
            # 直接使用标准A*，不限制范围（限制范围反而可能更慢）
            path = self.plan_path(occupancy, start, frontier)
            
            if path is None:
                relaxed_sd = max(0.0, self.safety_distance * 0.6)
                path = self.plan_path(
                    occupancy,
                    start,
                    frontier,
                    safety_distance=relaxed_sd,
                    max_unknown_cells=12,
                    unknown_step_penalty=1.5,
                )
            
            if path is not None:
                path_length = len(path)
                if path_length < min_dist:
                    min_dist = path_length
                    best_frontier = frontier
                    best_path = path
                    if path_length <= heuristic_lower + 1:
                        break

        return best_frontier, best_path

    def _plan_path_limited(self, occupancy, start, goal, max_radius):
        """
        A*路径规划的限制版本，只在指定半径范围内搜索。
        
        参数:
        - max_radius: 最大搜索半径（栅格单位）
        """
        sx, sy = start
        gx, gy = goal
        if start == goal:
            return [start]

        h, w = occupancy.shape
        sd = self.safety_distance
        
        start_state = (sx, sy)
        open_set = []
        start_h = self._heuristic((sx, sy), (gx, gy))
        heapq.heappush(open_set, (start_h, 0.0, sx, sy))

        came_from = {}
        g_score = {start_state: 0.0}
        closed_set = set()
        
        directions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414)
        ]
        
        while open_set:
            f_current, g_current, x, y = heapq.heappop(open_set)

            state = (x, y)
            if state in closed_set:
                continue
            
            # 检查是否超出搜索半径
            dist_from_start = math.sqrt((x - sx)**2 + (y - sy)**2)
            if dist_from_start > max_radius:
                continue
                
            closed_set.add(state)

            if (x, y) == (gx, gy):
                path = []
                current_state = state
                while current_state in came_from:
                    cx, cy = current_state
                    path.append((cx, cy))
                    current_state = came_from[current_state]
                path.append(start)
                path.reverse()
                return path

            for dx, dy, cost in directions:
                nx, ny = x + dx, y + dy
                
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                
                if occupancy[ny, nx] != 0:
                    continue
                
                if not self._is_safe(occupancy, nx, ny, safety_distance=sd):
                    continue
                
                if abs(dx) == 1 and abs(dy) == 1:
                    if (0 <= x + dx < w and 0 <= y < h and occupancy[y, x + dx] == 1) or \
                       (0 <= x < w and 0 <= y + dy < h and occupancy[y + dy, x] == 1):
                        continue
                
                next_state = (nx, ny)
                if next_state in closed_set:
                    continue
                    
                tentative_g = g_score[state] + cost

                if next_state not in g_score or tentative_g < g_score[next_state]:
                    came_from[next_state] = state
                    g_score[next_state] = tentative_g
                    heuristic = self._heuristic((nx, ny), (gx, gy))
                    f_score = tentative_g + heuristic
                    heapq.heappush(open_set, (f_score, tentative_g, nx, ny))
        
        return None

    def plan_path(self, occupancy, start, goal, safety_distance=None,
                  max_unknown_cells=0, unknown_step_penalty=2.0):
        """
        使用A*算法规划从start到goal的路径，支持8方向移动（包括对角线）。
        start: (x_idx, y_idx), goal: (x_idx, y_idx)
        safety_distance: 与障碍物保持的安全距离，如果为None则使用实例默认值。
        max_unknown_cells: 允许经过的未知栅格数量上限（0表示禁用）。
        unknown_step_penalty: 每穿越一个未知栅格额外增加的路径代价，用于优先选择已知区域。

        返回路径单元格坐标列表，包含start和goal。若无法到达返回None。
        """
        sx, sy = start
        gx, gy = goal
        if start == goal:
            return [start]

        h, w = occupancy.shape
        sd = self.safety_distance if (safety_distance is None) else safety_distance
        allow_unknown = max_unknown_cells is not None and max_unknown_cells > 0

        start_state = (sx, sy, 0)
        open_set = []
        start_h = self._heuristic((sx, sy), (gx, gy))
        heapq.heappush(open_set, (start_h, 0.0, 0, sx, sy))

        came_from = {}
        g_score = {start_state: 0.0}
        closed_set = set()
        
        # 8个方向的移动，包括对角线
        directions = [
            (-1, 0, 1.0),   # 左
            (1, 0, 1.0),    # 右
            (0, -1, 1.0),   # 上
            (0, 1, 1.0),    # 下
            (-1, -1, 1.414), # 左上对角线
            (1, -1, 1.414),  # 右上对角线
            (-1, 1, 1.414),  # 左下对角线
            (1, 1, 1.414)    # 右下对角线
        ]
        
        while open_set:
            f_current, g_current, unknown_used, x, y = heapq.heappop(open_set)

            state = (x, y, unknown_used)
            if state in closed_set:
                continue
            closed_set.add(state)

            if (x, y) == (gx, gy):
                path = []
                current_state = state
                while current_state in came_from:
                    cx, cy, _ = current_state
                    path.append((cx, cy))
                    current_state = came_from[current_state]
                path.append(start)
                path.reverse()
                return path

            for dx, dy, cost in directions:
                nx, ny = x + dx, y + dy
                
                # 检查边界
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                
                cell_value = occupancy[ny, nx]
                if cell_value == 1:
                    continue

                next_unknown_used = unknown_used
                step_penalty = 0.0
                allow_unknown_cell = False
                if cell_value == -1:
                    if not allow_unknown:
                        continue
                    next_unknown_used = unknown_used + 1
                    if max_unknown_cells is not None and next_unknown_used > max_unknown_cells:
                        continue
                    allow_unknown_cell = True
                    step_penalty = unknown_step_penalty
                
                if not self._is_safe(occupancy, nx, ny, safety_distance=sd, allow_unknown_cell=allow_unknown_cell):
                    continue
                
                # 对于对角线移动，检查是否会穿过墙角
                if abs(dx) == 1 and abs(dy) == 1:
                    if (0 <= x + dx < w and 0 <= y < h and occupancy[y, x + dx] == 1) or \
                       (0 <= x < w and 0 <= y + dy < h and occupancy[y + dy, x] == 1):
                        continue
                
                next_state = (nx, ny, next_unknown_used)
                if next_state in closed_set:
                    continue
                tentative_g = g_score[state] + cost + step_penalty

                if next_state not in g_score or tentative_g < g_score[next_state]:
                    came_from[next_state] = state
                    g_score[next_state] = tentative_g
                    heuristic = self._heuristic((nx, ny), (gx, gy))
                    f_score = tentative_g + heuristic + next_unknown_used * 0.25
                    heapq.heappush(open_set, (f_score, tentative_g, next_unknown_used, nx, ny))
        
        return None  # 无法找到路径
    
    def plan_path_no_safety(self, occupancy, start, goal):
        """
        使用A*算法规划从start到goal的路径，不考虑安全距离，支持8方向移动（包括对角线）。
        适用于返回起点时的紧急路径规划。
        start: (x_idx, y_idx), goal: (x_idx, y_idx)
        返回路径单元格坐标列表，包含start和goal。若无法到达返回None。
        """
        sx, sy = start
        gx, gy = goal
        if start == goal:
            return [start]
        
        h, w = occupancy.shape
        
        # A*算法数据结构
        open_set = []
        heapq.heappush(open_set, (0.0, sx, sy))
        
        came_from = {}
        g_score = {(sx, sy): 0.0}
        f_score = {(sx, sy): self._heuristic((sx, sy), (gx, gy))}
        
        closed_set = set()
        
        # 8个方向的移动，包括对角线
        directions = [
            (-1, 0, 1.0),   # 左
            (1, 0, 1.0),    # 右
            (0, -1, 1.0),   # 上
            (0, 1, 1.0),    # 下
            (-1, -1, 1.414), # 左上对角线
            (1, -1, 1.414),  # 右上对角线
            (-1, 1, 1.414),  # 左下对角线
            (1, 1, 1.414)    # 右下对角线
        ]
        
        while open_set:
            _, x, y = heapq.heappop(open_set)
            
            if (x, y) in closed_set:
                continue
                
            closed_set.add((x, y))
            
            if (x, y) == (gx, gy):
                # 重建路径
                path = []
                current = (x, y)
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for dx, dy, cost in directions:
                nx, ny = x + dx, y + dy
                
                # 检查边界
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                
                if occupancy[ny, nx] != 0:
                    continue
                
                # 对于对角线移动，检查是否会穿过墙角
                if abs(dx) == 1 and abs(dy) == 1:
                    # 检查两个相邻的直角方向是否可通行
                    if (occupancy[y, x + dx] != 0) or (occupancy[y + dy, x] != 0):
                        continue
                
                if (nx, ny) in closed_set:
                    continue
                
                tentative_g_score = g_score[(x, y)] + cost
                
                if (nx, ny) not in g_score or tentative_g_score < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = (x, y)
                    g_score[(nx, ny)] = tentative_g_score
                    f_score[(nx, ny)] = tentative_g_score + self._heuristic((nx, ny), (gx, gy))
                    heapq.heappush(open_set, (f_score[(nx, ny)], nx, ny))
        
        return None  # 无法找到路径
    
    def _heuristic(self, a, b):
        """计算两点间的启发式距离（对角线距离）"""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        # 对角线距离：允许对角线移动的最短距离
        return max(dx, dy) + (1.414 - 1) * min(dx, dy)

    def _is_safe(self, occupancy, x, y, safety_distance=1.0, allow_unknown_cell=False):
        """
        检查点(x,y)是否与障碍物保持足够的安全距离（混合策略优化版本）。
        
        策略：
        - 对于小范围检查（safety_distance <= 5）：使用快速局部搜索
        - 对于大范围或全局检测：使用距离变换图（仅在 _find_all_frontiers 中）
        
        参数:
        - occupancy: 占用栅格地图
        - x, y: 要检查的点坐标
        - safety_distance: 与障碍物保持的安全距离（单位：栅格数）
        
        返回:
        - True: 如果点与所有障碍物的距离都大于安全距离
        - False: 如果点太靠近障碍物
        """
        h, w = occupancy.shape
        
        # 检查边界
        if not (0 <= x < w and 0 <= y < h):
            return False
        
        # 检查是否该点本身是障碍物
        cell_value = occupancy[y, x]
        if cell_value == 1:
            return False
        if cell_value == -1 and not allow_unknown_cell:
            return False
        
        # 使用快速局部搜索（对于大多数调用场景更快）
        search_range = math.ceil(safety_distance)
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if occupancy[ny, nx] == 1:
                    actual_dist = math.sqrt(dx * dx + dy * dy)
                    if actual_dist <= safety_distance:
                        return False
        return True
    
    def _find_all_frontiers(self, occupancy):
        """
        遍历整张地图找出所有满足安全距离的前沿点（优化版本）。
        使用缓存和聚类减少重复计算。
        前沿定义为：已知空闲且邻接未知区域的栅格。

        返回:
        - list[(x, y)]: 所有前沿点的坐标列表
        """
        # 检查缓存
        occupancy_hash = hash(occupancy.tobytes())
        if self._cached_occupancy_hash == occupancy_hash and self._cached_frontiers is not None:
            return self._cached_frontiers
        
        h, w = occupancy.shape
        frontiers = []
        
        # 只检查空闲栅格
        free_mask = (occupancy == 0)
        # 找到邻接未知区域的栅格
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        
        # 优化：先找到所有潜在前沿点（邻接未知区域），再批量检查安全距离
        potential_frontiers = []
        for y in range(h):
            for x in range(w):
                if not free_mask[y, x]:
                    continue
                
                # 快速检查是否邻接未知区域
                has_unknown_neighbor = False
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and occupancy[ny, nx] == -1:
                        has_unknown_neighbor = True
                        break
                
                if has_unknown_neighbor:
                    potential_frontiers.append((x, y))
        
        # 批量检查安全距离（只对潜在前沿点检查）
        for x, y in potential_frontiers:
            if self._is_safe(occupancy, x, y, safety_distance=self.safety_distance):
                frontiers.append((x, y))
        
        # 缓存结果
        self._cached_frontiers = frontiers
        self._cached_occupancy_hash = occupancy_hash
        
        return frontiers

    def calculate_path_length(self, path, resolution):
        """计算路径的实际长度（米）。"""
        if not path or len(path) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            grid_distance = math.sqrt(dx * dx + dy * dy)
            total_length += grid_distance * resolution

        return total_length

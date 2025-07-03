from collections import deque
import heapq
import math

class FrontierExplorer:
    """前沿探索与路径规划模块。确定下一个目标前沿并规划路径。"""
    def __init__(self, safety_distance=1.0):
        """
        初始化前沿探索器。
        
        参数:
        - safety_distance: 路径规划时与障碍物保持的安全距离（栅格单位）
        """
        self.safety_distance = safety_distance

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
        从start出发，使用多策略方法找到最近的前沿单元及路径。
        
        策略顺序：
        1. 局部BFS搜索 - 从起点开始在连通区域内搜索前沿
        2. 全局前沿检测 - 如果局部搜索失败，寻找所有前沿并选择最近的
        
        参数:
        - occupancy: 占用栅格地图
        - start: (x_idx, y_idx) 起点栅格索引
        
        返回:
        - (frontier_cell, path): 找到时返回目标前沿和路径
        - (None, None): 未找到前沿时返回
        """
        h, w = occupancy.shape
        sx, sy = start

        # 策略1: 局部BFS搜索
        visited = [[False]*w for _ in range(h)]
        parent = {}
        dq = deque()
        dq.append((sx, sy))
        visited[sy][sx] = True
        parent[(sx, sy)] = None
        frontier_cell = None
        
        while dq:
            x, y = dq.popleft()
            # 检查当前点是否为前沿
            if occupancy[y, x] == 0:
                is_frontier = False
                for (nx, ny) in [(x-1,y), (x+1,y), (x,y-1), (x,y+1),
                                (x-1,y-1), (x+1,y+1), (x-1,y+1), (x+1,y-1)]:
                    if 0 <= nx < w and 0 <= ny < h and occupancy[ny, nx] == -1:
                        is_frontier = True
                        break
                if is_frontier and (x, y) != (sx, sy):
                    frontier_cell = (x, y)
                    break
                    
            # BFS扩展
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1),
                          (-1,-1), (1,1), (-1,1), (1,-1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h and 
                    not visited[ny][nx] and
                    (occupancy[ny, nx] == 0 or occupancy[ny, nx] == -1)):  # 允许扩展到未知区域
                    
                    # 检查安全距离
                    if not self._is_safe(occupancy, nx, ny, safety_distance=max(0.5, self.safety_distance)):  # 降低安全距离要求
                        continue
                    
                    # 对角线移动时检查墙角
                    if abs(dx) == 1 and abs(dy) == 1:
                        if (occupancy[y, x + dx] == 1) or (occupancy[y + dy, x] == 1):  # 只检查确定的障碍物
                            continue
                    
                    visited[ny][nx] = True
                    parent[(nx, ny)] = (x, y)
                    dq.append((nx, ny))

        # 如果BFS找到了前沿，构建路径并返回
        if frontier_cell is not None:
            path = []
            cur = frontier_cell
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return frontier_cell, path

        # 策略2: 全局前沿检测
        frontiers = self._find_all_frontiers(occupancy)
        if not frontiers:
            return None, None  # 没有找到任何前沿

        # 计算到每个前沿的距离，选择最近的
        best_frontier = None
        best_path = None
        min_dist = float('inf')

        for frontier in frontiers:
            # 使用A*算法尝试规划路径
            path = self.plan_path(occupancy, start, frontier)
            if path is not None:
                # 计算路径长度
                path_length = len(path)
                if path_length < min_dist:
                    min_dist = path_length
                    best_frontier = frontier
                    best_path = path

        return best_frontier, best_path

    def plan_path(self, occupancy, start, goal):
        """
        使用A*算法规划从start到goal的路径，支持8方向移动（包括对角线）。
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
                
                # 检查是否为可通行区域
                if occupancy[ny, nx] != 0:
                    continue
                    
                # 检查是否与障碍物保持足够的安全距离
                if not self._is_safe(occupancy, nx, ny, safety_distance=self.safety_distance):
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
                
                # 只检查是否为可通行区域，不考虑安全距离
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

    def _is_safe(self, occupancy, x, y, safety_distance=1.0):
        """
        检查点(x,y)是否与障碍物保持足够的安全距离。
        
        参数:
        - occupancy: 占用栅格地图
        - x, y: 要检查的点坐标
        - safety_distance: 与障碍物保持的安全距离（单位：栅格数）
        
        返回:
        - True: 如果点与所有障碍物的距离都大于安全距离
        - False: 如果点太靠近障碍物
        """
        h, w = occupancy.shape
        
        # 检查是否该点本身是障碍物
        if occupancy[y, x] != 0:
            return False
            
        # 检查周围一定范围内是否有障碍物
        search_range = math.ceil(safety_distance)
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                # 跳过超出边界的点
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                    
                # 如果是障碍物，计算实际距离
                if occupancy[ny, nx] == 1:  # 1表示障碍物
                    # 计算与障碍物的欧氏距离
                    actual_dist = math.sqrt(dx * dx + dy * dy)
                    if actual_dist <= safety_distance:
                        return False
                        
        return True
    
    def _find_all_frontiers(self, occupancy):
        """
        遍历整张地图找出所有前沿点。
        前沿定义为：已知空闲且邻接未知区域的栅格。
        
        参数:
        - occupancy: 占用栅格地图
        
        返回:
        - list of (x, y): 所有前沿点的坐标列表
        """
        h, w = occupancy.shape
        frontiers = []
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]
        
        for y in range(h):
            for x in range(w):
                # 只检查空闲格
                if occupancy[y, x] == 0:
                    # 检查周围8个方向是否有未知区域
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if occupancy[ny, nx] == -1:  # -1表示未知区域
                                # 检查安全距离
                                if self._is_safe(occupancy, x, y, self.safety_distance):
                                    frontiers.append((x, y))
                                    break
        return frontiers

    def calculate_path_length(self, path, resolution):
        """
        计算路径的实际长度（米）
        
        参数:
        - path: 路径点列表 [(x, y), ...]
        - resolution: 栅格分辨率（米/栅格）
        
        返回:
        - float: 路径长度（米）
        """
        if not path or len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # 计算两点间的欧氏距离（栅格单位）
            dx = x2 - x1
            dy = y2 - y1
            grid_distance = math.sqrt(dx * dx + dy * dy)
            
            # 转换为实际距离（米）
            actual_distance = grid_distance * resolution
            total_length += actual_distance
        
        return total_length

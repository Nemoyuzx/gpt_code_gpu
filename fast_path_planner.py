"""
优化的路径规划模块
实现多种高效路径规划算法，根据距离自动选择最优策略
"""
import numpy as np
from collections import deque
import heapq
from typing import List, Tuple, Optional, Set, Dict
import time


class FastPathPlanner:
    """
    快速路径规划器
    根据起点和终点距离自动选择最优算法：
    - 短距离(<50格): BFS
    - 中等距离(50-200格): 双端BFS
    - 长距离(>200格): 优化A*
    """
    
    def __init__(self, safety_distance: float = 0.0):
        self.safety_distance = safety_distance
        
        # 8方向移动
        self.directions_8 = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
            (-1, -1), (1, -1), (-1, 1), (1, 1)  # 对角线
        ]
        
        # 对角线移动的代价（根号2）
        self.diagonal_cost = 1.414
        
        # 路径缓存（简单的LRU缓存）
        self.path_cache = {}
        self.cache_size_limit = 100
        
    def plan_path(self, occupancy: np.ndarray, start: Tuple[int, int], 
                  goal: Tuple[int, int], safety_distance: Optional[float] = None,
                  max_unknown_cells: int = 0, unknown_step_penalty: float = 2.0,
                  max_iterations: int = 10000) -> Optional[List[Tuple[int, int]]]:
        """
        智能路径规划：根据距离自动选择算法
        """
        if start == goal:
            return [start]
        
        sd = self.safety_distance if safety_distance is None else safety_distance
        
        # 检查缓存
        cache_key = (start, goal, sd, max_unknown_cells)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # 计算曼哈顿距离
        manhattan_dist = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        
        # 根据距离选择算法
        if manhattan_dist < 50:
            # 短距离：使用BFS
            path = self._bfs_path(occupancy, start, goal, sd, max_unknown_cells)
        elif manhattan_dist < 200:
            # 中等距离：使用双端BFS
            path = self._bidirectional_bfs(occupancy, start, goal, sd, max_unknown_cells)
        else:
            # 长距离：使用优化A*
            path = self._optimized_astar(
                occupancy, start, goal, sd, max_unknown_cells, 
                unknown_step_penalty, max_iterations
            )
        
        # 缓存结果
        if path is not None:
            self._add_to_cache(cache_key, path)
        
        return path
    
    def _bfs_path(self, occupancy: np.ndarray, start: Tuple[int, int],
                  goal: Tuple[int, int], safety_distance: float,
                  max_unknown_cells: int) -> Optional[List[Tuple[int, int]]]:
        """
        BFS路径规划（最快，适合短距离）
        使用parent字典而非路径列表，避免每步都复制路径
        """
        h, w = occupancy.shape
        sx, sy = start
        gx, gy = goal
        
        # 使用numpy数组作为visited，比dict/set更快
        visited = np.zeros((h, w), dtype=bool)
        visited[sy, sx] = True
        
        # 使用deque作为队列，存储parent字典用于重建路径
        queue = deque([(sx, sy, 0)])  # (x, y, unknown_count)
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        
        allow_unknown = max_unknown_cells > 0
        
        while queue:
            x, y, unknown_used = queue.popleft()
            
            # 检查是否到达目标
            if (x, y) == (gx, gy):
                # 重建路径
                path = []
                current = (x, y)
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                return path
            
            # 8方向扩展
            for dx, dy in self.directions_8:
                nx, ny = x + dx, y + dy
                
                # 边界检查
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                
                # 已访问检查
                if visited[ny, nx]:
                    continue
                
                # 障碍物检查
                cell_value = occupancy[ny, nx]
                if cell_value == 1:
                    continue
                
                # 未知格子检查
                next_unknown = unknown_used
                if cell_value == -1:
                    if not allow_unknown:
                        continue
                    next_unknown += 1
                    if next_unknown > max_unknown_cells:
                        continue
                
                # 安全距离检查
                if not self._is_safe_fast(occupancy, nx, ny, safety_distance, 
                                         allow_unknown=(cell_value == -1)):
                    continue
                
                # 对角线穿墙检查
                if abs(dx) == 1 and abs(dy) == 1:
                    if occupancy[y, x + dx] == 1 or occupancy[y + dy, x] == 1:
                        continue
                
                # 标记已访问并加入队列
                visited[ny, nx] = True
                parent[(nx, ny)] = (x, y)
                queue.append((nx, ny, next_unknown))
        
        return None
    
    def _bidirectional_bfs(self, occupancy: np.ndarray, start: Tuple[int, int],
                           goal: Tuple[int, int], safety_distance: float,
                           max_unknown_cells: int) -> Optional[List[Tuple[int, int]]]:
        """
        双端BFS（两端同时搜索，搜索空间减半）
        """
        h, w = occupancy.shape
        sx, sy = start
        gx, gy = goal
        
        # 前向搜索
        visited_fwd = np.zeros((h, w), dtype=bool)
        visited_fwd[sy, sx] = True
        queue_fwd = deque([(sx, sy)])
        parent_fwd: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        
        # 后向搜索
        visited_bwd = np.zeros((h, w), dtype=bool)
        visited_bwd[gy, gx] = True
        queue_bwd = deque([(gx, gy)])
        parent_bwd: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {goal: None}
        
        allow_unknown = max_unknown_cells > 0
        meeting_point = None
        
        # 交替进行前向和后向搜索
        iteration = 0
        max_iter = min(h * w, 5000)
        
        while queue_fwd and queue_bwd and iteration < max_iter:
            iteration += 1
            
            # 前向搜索一步
            if queue_fwd:
                x, y = queue_fwd.popleft()
                
                # 检查是否与后向搜索相遇
                if visited_bwd[y, x]:
                    meeting_point = (x, y)
                    break
                
                # 扩展
                for dx, dy in self.directions_8:
                    nx, ny = x + dx, y + dy
                    
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    if visited_fwd[ny, nx]:
                        continue
                    
                    cell_value = occupancy[ny, nx]
                    if cell_value == 1:
                        continue
                    if cell_value == -1 and not allow_unknown:
                        continue
                    
                    if not self._is_safe_fast(occupancy, nx, ny, safety_distance,
                                             allow_unknown=(cell_value == -1)):
                        continue
                    
                    if abs(dx) == 1 and abs(dy) == 1:
                        if occupancy[y, x + dx] == 1 or occupancy[y + dy, x] == 1:
                            continue
                    
                    visited_fwd[ny, nx] = True
                    parent_fwd[(nx, ny)] = (x, y)
                    queue_fwd.append((nx, ny))
            
            # 后向搜索一步
            if queue_bwd:
                x, y = queue_bwd.popleft()
                
                # 检查是否与前向搜索相遇
                if visited_fwd[y, x]:
                    meeting_point = (x, y)
                    break
                
                # 扩展
                for dx, dy in self.directions_8:
                    nx, ny = x + dx, y + dy
                    
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    if visited_bwd[ny, nx]:
                        continue
                    
                    cell_value = occupancy[ny, nx]
                    if cell_value == 1:
                        continue
                    if cell_value == -1 and not allow_unknown:
                        continue
                    
                    if not self._is_safe_fast(occupancy, nx, ny, safety_distance,
                                             allow_unknown=(cell_value == -1)):
                        continue
                    
                    if abs(dx) == 1 and abs(dy) == 1:
                        if occupancy[y, x + dx] == 1 or occupancy[y + dy, x] == 1:
                            continue
                    
                    visited_bwd[ny, nx] = True
                    parent_bwd[(nx, ny)] = (x, y)
                    queue_bwd.append((nx, ny))
        
        # 如果找到相遇点，重建路径
        if meeting_point:
            # 从起点到相遇点
            path_fwd = []
            current = meeting_point
            while current is not None:
                path_fwd.append(current)
                current = parent_fwd.get(current)
            path_fwd.reverse()
            
            # 从相遇点到终点
            path_bwd = []
            current = parent_bwd.get(meeting_point)
            while current is not None:
                path_bwd.append(current)
                current = parent_bwd.get(current)
            
            return path_fwd + path_bwd
        
        return None
    
    def _optimized_astar(self, occupancy: np.ndarray, start: Tuple[int, int],
                        goal: Tuple[int, int], safety_distance: float,
                        max_unknown_cells: int, unknown_step_penalty: float,
                        max_iterations: int) -> Optional[List[Tuple[int, int]]]:
        """
        优化的A*算法（适合长距离）
        优化：
        1. 使用numpy数组代替dict存储g_score
        2. 减少方向数（去掉马步跳跃）
        3. 更激进的剪枝
        """
        h, w = occupancy.shape
        sx, sy = start
        gx, gy = goal
        
        # 使用numpy数组存储g_score（初始化为无穷大）
        g_score = np.full((h, w), np.inf, dtype=np.float32)
        g_score[sy, sx] = 0.0
        
        # 使用numpy数组标记closed
        closed = np.zeros((h, w), dtype=bool)
        
        # 优先队列
        open_set = []
        start_h = self._heuristic(start, goal)
        heapq.heappush(open_set, (start_h, 0.0, sx, sy, 0))  # (f, g, x, y, unknown)
        
        # parent字典用于重建路径
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        allow_unknown = max_unknown_cells > 0
        iterations = 0
        
        # 更激进的代价限制
        cost_limit = start_h * 1.5
        
        while open_set and iterations < max_iterations:
            f_current, g_current, x, y, unknown_used = heapq.heappop(open_set)
            iterations += 1
            
            # 如果已经在closed中，跳过
            if closed[y, x]:
                continue
            closed[y, x] = True
            
            # 早期终止
            if g_current > cost_limit:
                continue
            
            # 到达目标
            if (x, y) == goal:
                path = []
                current = (x, y)
                while current in parent:
                    path.append(current)
                    current = parent[current]
                path.append(start)
                path.reverse()
                return path
            
            # 扩展邻居（只使用8方向，不用马步）
            for dx, dy in self.directions_8:
                nx, ny = x + dx, y + dy
                
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                
                if closed[ny, nx]:
                    continue
                
                cell_value = occupancy[ny, nx]
                if cell_value == 1:
                    continue
                
                next_unknown = unknown_used
                step_penalty = 0.0
                if cell_value == -1:
                    if not allow_unknown:
                        continue
                    next_unknown += 1
                    if next_unknown > max_unknown_cells:
                        continue
                    step_penalty = unknown_step_penalty
                
                if not self._is_safe_fast(occupancy, nx, ny, safety_distance,
                                         allow_unknown=(cell_value == -1)):
                    continue
                
                # 对角线穿墙检查
                if abs(dx) == 1 and abs(dy) == 1:
                    if occupancy[y, x + dx] == 1 or occupancy[y + dy, x] == 1:
                        continue
                
                # 计算代价
                move_cost = self.diagonal_cost if (abs(dx) + abs(dy) == 2) else 1.0
                tentative_g = g_current + move_cost + step_penalty
                
                # 如果找到更短路径
                if tentative_g < g_score[ny, nx]:
                    g_score[ny, nx] = tentative_g
                    parent[(nx, ny)] = (x, y)
                    
                    h = self._heuristic((nx, ny), goal)
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, tentative_g, nx, ny, next_unknown))
        
        return None
    
    def _is_safe_fast(self, occupancy: np.ndarray, x: int, y: int,
                     safety_distance: float, allow_unknown: bool = False) -> bool:
        """
        快速安全检查（使用numpy切片）
        """
        if safety_distance <= 0:
            return True
        
        h, w = occupancy.shape
        sd = int(np.ceil(safety_distance))
        
        # 计算检查区域
        y_min = max(0, y - sd)
        y_max = min(h, y + sd + 1)
        x_min = max(0, x - sd)
        x_max = min(w, x + sd + 1)
        
        # 使用numpy切片检查区域
        region = occupancy[y_min:y_max, x_min:x_max]
        
        # 检查是否有障碍物
        if np.any(region == 1):
            # 需要精确距离检查
            for dy in range(y_min - y, y_max - y):
                for dx in range(x_min - x, x_max - x):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= nx < w and 0 <= ny < h:
                        if occupancy[ny, nx] == 1:
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist <= safety_distance:
                                return False
        
        return True
    
    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        启发式函数（对角线距离）
        """
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])
        # 对角线距离：min * sqrt(2) + (max - min) * 1
        return 1.414 * min(dx, dy) + abs(dx - dy)
    
    def _add_to_cache(self, key, path):
        """添加到缓存（简单LRU）"""
        if len(self.path_cache) >= self.cache_size_limit:
            # 删除最早的条目
            self.path_cache.pop(next(iter(self.path_cache)))
        self.path_cache[key] = path
    
    def clear_cache(self):
        """清空缓存（地图更新时调用）"""
        self.path_cache.clear()


def benchmark_planners():
    """性能对比测试"""
    from frontier_explorer import FrontierExplorer
    
    # 创建测试地图
    map_size = 300
    occupancy = np.zeros((map_size, map_size), dtype=np.int8)
    
    # 添加一些障碍物
    occupancy[100:120, 50:200] = 1
    occupancy[150:170, 100:250] = 1
    
    # 测试用例
    test_cases = [
        ((10, 10), (30, 30), "短距离"),
        ((10, 10), (100, 100), "中等距离"),
        ((10, 10), (250, 250), "长距离"),
    ]
    
    fast_planner = FastPathPlanner(safety_distance=1.0)
    old_explorer = FrontierExplorer(safety_distance=1.0)
    
    print("\n" + "="*70)
    print("路径规划性能对比测试")
    print("="*70)
    
    for start, goal, label in test_cases:
        print(f"\n{label}: {start} -> {goal}")
        print("-" * 70)
        
        # 测试新规划器
        t0 = time.perf_counter()
        path_new = fast_planner.plan_path(occupancy, start, goal)
        t1 = time.perf_counter()
        time_new = (t1 - t0) * 1000
        
        # 测试旧规划器
        t0 = time.perf_counter()
        path_old = old_explorer.plan_path(occupancy, start, goal, max_iterations=10000)
        t1 = time.perf_counter()
        time_old = (t1 - t0) * 1000
        
        speedup = time_old / time_new if time_new > 0 else 0
        
        print(f"  旧A*规划器: {time_old:.2f} ms, 路径长度: {len(path_old) if path_old else 'None'}")
        print(f"  新快速规划器: {time_new:.2f} ms, 路径长度: {len(path_new) if path_new else 'None'}")
        print(f"  ⚡ 性能提升: {speedup:.2f}x")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    benchmark_planners()

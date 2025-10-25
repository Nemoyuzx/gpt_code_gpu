import math
import numpy as np
import json  # 添加json模块导入

class MazeLoader:
    """迷宫地图加载器。用于加载墙壁信息和起始/目标点。"""
    def __init__(self, grid_resolution: float = 0.1):
        self.maze = None
        self.grid_resolution = max(1e-3, float(grid_resolution))

    # 添加解析配置文件的方法
    def _parse_config_file(self, file_path):
        """
        读取迷宫配置文件，解析起点、终点和墙壁数据。
        文件格式：
         第一行：起点X Y
         第二行：终点X Y
         剩余每行：X1 Y1 X2 Y2代表墙壁线段
        支持以#开头的注释和空行
        """
        walls = []
        start = None
        goal = None
        with open(file_path, 'r') as f:
            lines = f.readlines()
        data_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            data_lines.append(line)
        # 起点
        if len(data_lines) >= 1:
            vals = data_lines[0].split()
            start = (float(vals[0]), float(vals[1]))
        # 终点
        if len(data_lines) >= 2:
            vals = data_lines[1].split()
            goal = (float(vals[0]), float(vals[1]))
        # 墙壁
        for seg in data_lines[2:]:
            vals = seg.split()
            x1, y1, x2, y2 = map(float, vals)
            walls.append(((x1, y1), (x2, y2)))
        # 解析完成后确保start和goal不为None
        if start is None:
            start = (0.0, 0.0)
        if goal is None:
            goal = start
        return walls, start, goal

    def _parse_json_file(self, file_path):
        """
        读取JSON格式的迷宫配置文件，解析起点和墙壁数据。
        文件格式：
        {
          "segments": [
            {"start": [x1, y1], "end": [x2, y2]},
            ...
          ],
          "start_point": [start_x, start_y]
        }
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 解析墙壁线段
        walls = []
        if "segments" in data:
            for segment in data["segments"]:
                start = tuple(segment["start"])
                end = tuple(segment["end"])
                walls.append((start, end))
        
        # 解析起点
        start = tuple(data.get("start_point", (0.0, 0.0)))
        
        # 由于JSON格式可能没有明确的终点，将终点设为与起点相同
        goal = start
        
        return walls, start, goal

    def load(self, config=None):
        """
        加载迷宫配置。如果提供文件路径或配置则解析，否则使用默认迷宫配置文件maze.json。
        config: 可以是配置文件路径或配置数据(dict)。
        返回 Maze 实例。
        """
        # 根据config类型加载配置：字符串表示文件路径，dict表示配置数据，否则使用默认配置文件
        if isinstance(config, str) or config is None:
            cfg_file = config if isinstance(config, str) else '2.json'
            
            # 根据文件扩展名选择解析方法
            if cfg_file.lower().endswith('.json'):
                walls, start, goal = self._parse_json_file(cfg_file)
            else:
                walls, start, goal = self._parse_config_file(cfg_file)
        else:
            walls = config.get("walls", [])
            start = tuple(config.get("start", (0.0, 0.0)))
            goal = tuple(config.get("goal", start))
            
        # 确保start和goal存在
        if start is None:
            start = (0.0, 0.0)
        if goal is None:
            goal = start
        # 计算边界范围
        min_x = min(min(p[0] for p in wall) for wall in walls)
        min_y = min(min(p[1] for p in wall) for wall in walls)
        max_x = max(max(p[0] for p in wall) for wall in walls)
        max_y = max(max(p[1] for p in wall) for wall in walls)
        
        # 扩大地图边界，为SLAM探索提供更大的画布
        # 增大缓冲区以防止SLAM建图时边缘被标记为墙壁
        map_buffer = 3.0  # 在每个方向扩展5米的缓冲区（从2.0增加到5.0）
        min_x -= map_buffer
        min_y -= map_buffer
        max_x += map_buffer
        max_y += map_buffer
        
        # 将坐标平移使最小值为0（若为负）
        if min_x < 0 or min_y < 0:
            shift_x = -min_x if min_x < 0 else 0
            shift_y = -min_y if min_y < 0 else 0
            shifted_walls = []
            for (p1, p2) in walls:
                shifted_walls.append(((p1[0]+shift_x, p1[1]+shift_y),
                                       (p2[0]+shift_x, p2[1]+shift_y)))
            walls = shifted_walls
            start = (start[0] + (shift_x if min_x < 0 else 0),
                     start[1] + (shift_y if min_y < 0 else 0))
            goal = (goal[0] + (shift_x if min_x < 0 else 0),
                    goal[1] + (shift_y if min_y < 0 else 0))
            # 更新边界
            max_x += (shift_x if min_x < 0 else 0)
            max_y += (shift_y if min_y < 0 else 0)
            min_x = 0.0
            min_y = 0.0
        # 地图尺寸（用于栅格地图）
        res = self.grid_resolution
        width = math.ceil((max_x - min_x) / res)
        height = math.ceil((max_y - min_y) / res)
        # 栅格初始化为-1（未知）
        grid = -np.ones((int(height), int(width)), dtype=int)
        # 保存Maze信息
        self.maze = Maze(walls, start, goal, (min_x, min_y, max_x, max_y), res, grid)
        return self.maze

class Maze:
    """表示迷宫环境，包括墙壁列表、起点终点、边界和栅格地图结构等。"""
    def __init__(self, walls, start, goal, bounds, resolution, grid):
        self.walls = walls      # 墙壁线段列表，每个为((x1,y1),(x2,y2))
        self.start = start      # 起点坐标 (x,y)
        self.goal = goal        # 终点坐标 (x,y)
        self.bounds = bounds    # 边界 (min_x, min_y, max_x, max_y)
        self.resolution = resolution  # 栅格分辨率
        self.grid = grid        # 占据栅格地图 (numpy数组)

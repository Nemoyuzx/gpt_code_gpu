import math
import numpy as np

class Lidar:
    """激光雷达模拟器。生成360度距离扫描数据。"""
    def __init__(self, walls, max_range=12.0, angle_resolution=1.0, noise=0.0):
        """
        walls: 墙壁线段列表，每个为((x1,y1),(x2,y2))，在同一全局坐标系下。
        max_range: 最大探测距离 (米)。
        angle_resolution: 角分辨率 (度)。
        noise: 距离测量噪声标准差 (米)。
        """
        self.walls = walls
        self.max_range = max_range
        self.angle_resolution = angle_resolution
        self.noise = noise

    def scan(self, pose):
        """
        模拟一次360度扫描。返回距离列表（长度为360/angle_resolution）。
        pose: 机器人位姿 (x, y, theta) 用于确定激光雷达发射点和朝向。
        """
        x, y, theta = pose
        # 扫描角度范围0-360度
        num_beams = int(360 / self.angle_resolution)
        distances = []
        # 将角度转为弧度增量
        angle_step = math.radians(self.angle_resolution)
        # 遍历每条激光束
        for i in range(num_beams):
            angle = theta + i * angle_step  # 全局参考系下光束角度
            # 规范化角度0-2pi
            angle = math.atan2(math.sin(angle), math.cos(angle))
            # 射线方向向量
            dx = math.cos(angle)
            dy = math.sin(angle)
            # 遍历所有墙，找到最近交点
            closest_dist = self.max_range
            for (p1, p2) in self.walls:
                # 计算射线与墙壁线段的交点
                x1, y1 = p1
                x2, y2 = p2
                v_x, v_y = dx, dy  # 射线方向
                w_x, w_y = (x2 - x1), (y2 - y1)  # 框壁段向量
                # 计算叉积
                denom = v_x * w_y - v_y * w_x
                if abs(denom) < 1e-6:
                    # 射线与墙平行或重合，跳过
                    continue
                # 计算参数t和u
                t = ((x1 - x) * w_y - (y1 - y) * w_x) / denom
                u = ((x1 - x) * v_y - (y1 - y) * v_x) / denom
                if t >= 0 and 0 <= u <= 1:
                    dist = t
                    if dist < closest_dist:
                        closest_dist = dist
            # 添加噪声并截取最大距离
            measured_dist = closest_dist
            if self.noise > 0:
                measured_dist += np.random.normal(0, self.noise)
                # 防止噪声导致负或超出范围
                if measured_dist < 0:
                    measured_dist = 0.0
            if measured_dist > self.max_range:
                measured_dist = self.max_range
            distances.append(measured_dist)
        return distances
    
        
    def is_far_range(self, dist, threshold_ratio=0.8):
        """
        判断测量距离是否超出最大探测范围的指定比例
        
        参数:
        - dist: 测量距离
        - threshold_ratio: 阈值比例，默认为0.8（80%）
        
        返回:
        - True: 如果距离超出最大范围的threshold_ratio
        - False: 如果距离在范围内
        """
        return dist >= self.max_range * threshold_ratio

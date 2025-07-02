import math
import numpy as np

class Robot:
    """机器人运动模型与控制器。支持简单前进/旋转运动，保留接口支持差速驱动。"""
    def __init__(self, start_pose, odom_noise=(0.0, 0.0)):
        """
        start_pose: 起始位姿 (x, y, theta) （theta为朝向，弧度制）。
        odom_noise: 里程计噪声标准差 (trans_noise, rot_noise)，用于模拟运动噪声。
        """
        self.x, self.y, self.theta = start_pose
        # 轨迹记录（用于路径导出）
        self.trajectory = [(self.x, self.y)]
        # 里程计读数（初始化为真值）
        self.odom_x, self.odom_y, self.odom_theta = self.x, self.y, self.theta
        # 噪声标准差
        self.trans_noise, self.rot_noise = odom_noise

    def move(self, distance):
        """
        沿当前朝向前进一定距离。模拟实际移动并更新机器人真实位置和里程计读数。
        """
        # 更新真实位置 (无误差假设机器人实际移动即目标距离)
        self.x += distance * math.cos(self.theta)
        self.y += distance * math.sin(self.theta)
        # 更新里程计读数，加入噪声
        if self.trans_noise > 0:
            distance_odom = distance + np.random.normal(0, self.trans_noise)
        else:
            distance_odom = distance
        # 朝向theta在前进时不变
        self.odom_x += distance_odom * math.cos(self.odom_theta)
        self.odom_y += distance_odom * math.sin(self.odom_theta)
        # 记录轨迹
        self.trajectory.append((self.x, self.y))
        # 不返回值，更新内部状态

    def rotate(self, angle):
        """
        旋转机器人朝向angle（弧度）。正角度为逆时针转动。
        """
        # 更新真实朝向
        self.theta += angle
        # 归一化角度到[-pi, pi)
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        # 更新里程计朝向，加入噪声
        if self.rot_noise > 0:
            angle_odom = angle + np.random.normal(0, self.rot_noise)
        else:
            angle_odom = angle
        self.odom_theta += angle_odom
        self.odom_theta = math.atan2(math.sin(self.odom_theta), math.cos(self.odom_theta))
        # 旋转在原地，不改变位置
        # 记录轨迹（仅当旋转也想记录，可选；此处不记录纯旋转的位移，因为位置未变）

    def get_pose(self):
        """获取机器人真实位姿 (x, y, theta)。"""
        return (self.x, self.y, self.theta)

    def get_odom_pose(self):
        """获取机器人里程计估计的位姿 (x, y, theta)。"""
        return (self.odom_x, self.odom_y, self.odom_theta)

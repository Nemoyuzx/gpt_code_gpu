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
        
        # 降噪滤波器引用（由外部设置）
        self.noise_filter = None

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
        
    def move_to(self, target_x, target_y):
        """
        移动到指定的目标点(target_x, target_y)。
        
        注意：此方法假设机器人已经朝向目标点方向，
        即在调用此方法前应先调用rotate使机器人朝向目标。
        
        返回实际移动的距离。
        """
        # 计算目标点的方向和距离
        dx = target_x - self.x
        dy = target_y - self.y
        desired_theta = math.atan2(dy, dx)
        distance = math.hypot(dx, dy)
        
        # 确保机器人朝向与目标方向一致（允许小误差）
        angle_diff = abs(self.theta - desired_theta)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)
        if angle_diff > 0.1:  # 如果偏离超过0.1弧度（约5.7度），发出警告
            print(f"警告：机器人朝向({self.theta:.2f})与目标方向({desired_theta:.2f})不一致，可能导致移动误差")
        
        # 执行移动
        self.move(distance)
        return distance

    def get_pose(self):
        """获取机器人真实位姿 (x, y, theta)。"""
        return (self.x, self.y, self.theta)

    def get_odom_pose(self):
        """获取机器人里程计估计的位姿 (x, y, theta)。"""
        # 如果有滤波器，使用滤波后的数据
        if self.noise_filter is not None:
            filtered_x, filtered_y, filtered_theta = self.noise_filter.filter_odometry_data(
                self.odom_x, self.odom_y, self.odom_theta
            )
            return (filtered_x, filtered_y, filtered_theta)
        else:
            return (self.odom_x, self.odom_y, self.odom_theta)
    
    def set_noise_filter(self, noise_filter):
        """设置降噪滤波器"""
        self.noise_filter = noise_filter

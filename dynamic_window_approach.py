"""

移动机器人运动规划示例，使用动态窗口法（Dynamic Window Approach）

作者: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

"""

import math
from enum import Enum

import matplotlib.pyplot as plt
from matplotlib.backend_bases import Event
from matplotlib.patches import Circle
import numpy as np

show_animation = True


def _on_key_release(event: Event) -> None:
    if getattr(event, "key", None) == "escape":
        exit(0)


def dwa_control(x, config, goal, ob):
    """
    动态窗口法控制
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    仿真参数类
    """

    def __init__(self):
        # 机器人参数
        self.max_speed = 1.0  # [m/s] 最大速度
        self.min_speed = -0.5  # [m/s] 最小速度
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s] 最大角速度
        self.max_accel = 0.2  # [m/ss] 最大加速度
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss] 最大角加速度
        self.v_resolution = 0.01  # [m/s] 速度分辨率
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s] 角速度分辨率
        self.dt = 0.1  # [s] 运动预测的时间步长
        self.predict_time = 3.0  # [s] 预测时间
        self.to_goal_cost_gain = 0.15  # 目标代价权重
        self.speed_cost_gain = 1.0  # 速度代价权重
        self.obstacle_cost_gain = 1.0  # 障碍物代价权重
        self.robot_stuck_flag_cons = 0.001  # 防止机器人卡住的常数
        self.robot_type = RobotType.circle

        # 如果 robot_type == RobotType.circle
        # 也用于两种类型判断是否到达目标
        self.robot_radius = 1.0  # [m] 碰撞检测半径

        # 如果 robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] 碰撞检测宽度
        self.robot_length = 1.2  # [m] 碰撞检测长度
        # 障碍物 [x(m) y(m), ....]
        self.ob = np.array([[-1, -1],
                            [0, 2],
                            [4.0, 2.0],
                            [5.0, 4.0],
                            [5.0, 5.0],
                            [5.0, 6.0],
                            [5.0, 9.0],
                            [8.0, 9.0],
                            [7.0, 9.0],
                            [8.0, 10.0],
                            [9.0, 11.0],
                            [12.0, 13.0],
                            [12.0, 12.0],
                            [15.0, 15.0],
                            [13.0, 13.0]
                            ])

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type 必须是 RobotType 的实例")
        self._robot_type = value


config = Config()


def motion(x, u, dt):
    """
    运动模型
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    基于当前状态 x 计算动态窗口
    """

    # 由机器人参数确定的动态窗口
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # 由运动模型确定的动态窗口
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    用输入预测轨迹
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    用动态窗口计算最终输入
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # 在动态窗口内采样所有输入，评估所有轨迹
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)
            # 计算代价
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # 搜索最小代价的轨迹
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # 保证机器人不会卡住在
                    # 最优 v=0 m/s（在障碍物前面）且
                    # 最优 omega=0 rad/s（朝向目标，角度差为0）
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory


def calc_obstacle_cost(trajectory, ob, config):
    """
    计算障碍物代价，碰撞时为无穷大
    """
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK


def calc_to_goal_cost(trajectory, goal):
    """计算到目标的代价，基于角度差."""

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    return abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)
def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    else:
        ax = plt.gca()
        ax.add_patch(Circle((x, y), config.robot_radius, color="b"))
        out_x, out_y = (np.array([x, y]) +
                        np.array([math.cos(yaw), math.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")


def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):
    print(__file__ + " start!!")
    # 初始状态 [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # 目标位置 [x(m), y(m)]
    goal = np.array([gx, gy])

    # 输入 [前进速度, 角速度]

    config.robot_type = robot_type
    trajectory = np.array(x)
    ob = config.ob
    while True:
        u, predicted_trajectory = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)  # 仿真机器人运动
        trajectory = np.vstack((trajectory, x))  # 存储状态历史

        if show_animation:
            plt.cla()
            # 按下 esc 键停止仿真
            plt.gcf().canvas.mpl_connect('key_release_event', _on_key_release)
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # 检查是否到达目标
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main(robot_type=RobotType.rectangle)
    # main(robot_type=RobotType.circle)


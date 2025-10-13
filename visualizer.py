import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import math
import numpy as np

# 可视化参数
VISUALIZATION_UPDATE_TIME = 0.0001  # 可视化更新时间
LIDAR_DISPLAY_MAX_RANGE = 12.0  # 激光雷达显示最大范围
LIDAR_DISPLAY_MAX_RANGE = 12.0  # 激光雷达显示的最大范围

class Visualizer:
    """可视化模块：使用Matplotlib实时渲染机器人、地图和前沿探索状态。"""
    def __init__(self, maze, robot=None, slam=None):
        """
        maze: Maze对象，提供地图尺寸和分辨率等信息。
        robot: Robot对象（可选，用于导出路径）。
        slam: ICPSlam对象（可选，用于导出地图）。
        """
        self.maze = maze
        self.robot = robot
        self.slam = slam
        # 主SLAM窗口
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        plt.ion()
        plt.show()
        # 边界和刻度
        min_x, min_y, max_x, max_y = maze.bounds
        self.ax.set_xlim(min_x, max_x)
        self.ax.set_ylim(min_y, max_y)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("SLAM Exploration")
        self.paused = False
        # 按键事件绑定
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        # 存储紧急路径（红色显示）
        self.emergency_path = None
        self.obstacle_search_region = None
        self.bfs_debug_points = None
        self.bfs_debug_color = 'cyan'

    def _on_key_press(self, event):
        """键盘事件回调。'p'暂停/继续， 's'保存地图和路径。"""
        if event.key == 'p':
            self.paused = not self.paused
            print("[Visualizer] Pause toggled:", "Paused" if self.paused else "Running")
        elif event.key == 's':
            # 保存当前地图和路径
            map_file = "map.png"
            path_file = "path.csv"
            self.save_map(map_file)
            self.save_path(path_file)
            print(f"[Visualizer] 当前地图已保存至 {map_file}, 路径已保存至 {path_file}")

    def update(self, robot_pose, scan, frontiers=None, target=None, path=None, occupancy=None,
               predicted_traj=None, robot_radius=None, actual_traj=None, actual_traj_style=None,
               extra_trajs=None):
        self._render_update(
            robot_pose,
            scan,
            frontiers=frontiers,
            target=target,
            path=path,
            occupancy=occupancy,
            predicted_traj=predicted_traj,
            robot_radius=robot_radius,
            actual_traj=actual_traj,
            actual_traj_style=actual_traj_style,
            extra_trajs=extra_trajs,
        )

    def _render_update(self, robot_pose, scan, frontiers=None, target=None, path=None, occupancy=None,
                       predicted_traj=None, robot_radius=None, actual_traj=None, actual_traj_style=None,
                       extra_trajs=None):
        """
        更新绘制当前状态。
        robot_pose: 机器人位姿 (x, y, theta)。
        scan: 当前激光雷达扫描距离列表。
        frontiers: 当前所有前沿的栅格坐标列表 [(ix,iy), ...] （可选，用于显示前沿区域）。
        target: 当前目标前沿栅格 (ix, iy) （可选，用于突出显示目标）。
        path: 导航路径栅格序列 [(ix,iy), ...] （可选，用于显示规划路径）。
        occupancy: 当前栅格地图 (numpy数组) （可选，用于绘制地图）。
        predicted_traj: 由DWA预测的轨迹 (N×5 numpy数组，使用 [:,0],[ :,1 ] 作为XY)（可选）。
        robot_radius: 机器人半径（米），若提供则以圆形边界显示机器人（可选）。
        """
        x, y, theta = robot_pose
        # 清除之前的绘图
        self.ax.cla()
        # 绘制栅格地图
        if occupancy is not None:
            h, w = occupancy.shape
            # 使用矢量化构建显示矩阵：未知=灰(0.5), 空闲=白(1), 占据=黑(0)
            display_grid = np.full((h, w), 0.5, dtype=float)
            display_grid[occupancy == 0] = 1.0
            display_grid[occupancy == 1] = 0.0
            # 显示栅格地图
            min_x, min_y, max_x, max_y = self.maze.bounds
            extent = (min_x, max_x, min_y, max_y)
            self.ax.imshow(display_grid, origin='lower', cmap='gray', extent=extent, vmin=0.0, vmax=1.0)
        
        # 绘制障碍物搜索区域边界
        if self.obstacle_search_region is not None:
            min_x, min_y, max_x, max_y = self.obstacle_search_region
            try:
                rect = Rectangle(
                    (min_x, min_y),
                    max_x - min_x,
                    max_y - min_y,
                    linewidth=1.8,
                    edgecolor='yellow',
                    facecolor='none',
                    linestyle='--',
                    label='Obstacle Search Bounds'
                )
                self.ax.add_patch(rect)
            except Exception:
                pass

        # 绘制BFS调试点
        if self.bfs_debug_points:
            try:
                pts = np.asarray(self.bfs_debug_points, dtype=float)
                if pts.ndim == 2 and pts.shape[0] > 0:
                    wx = self.maze.bounds[0] + (pts[:, 0] + 0.5) * self.maze.resolution
                    wy = self.maze.bounds[1] + (pts[:, 1] + 0.5) * self.maze.resolution
                    self.ax.scatter(wx, wy, s=12, c=self.bfs_debug_color, alpha=0.25, marker='s', label='BFS Region')
            except Exception:
                pass

        # 绘制目标前沿
        if target:
            tx = self.maze.bounds[0] + (target[0] + 0.5) * self.maze.resolution
            ty = self.maze.bounds[1] + (target[1] + 0.5) * self.maze.resolution
            self.ax.scatter([tx], [ty], c='r', marker='*', s=100, label='Target Frontier')
        # 绘制规划路径
        if path:
            px = [self.maze.bounds[0] + (ix + 0.5) * self.maze.resolution for (ix, iy) in path]
            py = [self.maze.bounds[1] + (iy + 0.5) * self.maze.resolution for (ix, iy) in path]
            if len(px) > 1:
                self.ax.plot(px, py, color='g', linestyle='--', label='Path')
                
        # 绘制紧急路径（红色线条）
        if self.emergency_path:
            epx = [self.maze.bounds[0] + (ix + 0.5) * self.maze.resolution for (ix, iy) in self.emergency_path]
            epy = [self.maze.bounds[1] + (iy + 0.5) * self.maze.resolution for (ix, iy) in self.emergency_path]
            if len(epx) > 1:
                self.ax.plot(epx, epy, color='red', linewidth=2, label='Emergency Path (No Safety)')
        # 绘制激光雷达当前扫描点云
        if scan and self.slam:
            scan_pts_x = []
            scan_pts_y = []
            num_beams = len(scan)
            max_range = LIDAR_DISPLAY_MAX_RANGE  # 使用固定的最大范围
            for i, dist in enumerate(scan):
                if dist < max_range:
                    angle = theta + math.radians(i * (360.0/num_beams))
                    sx = x + dist * math.cos(angle)
                    sy = y + dist * math.sin(angle)
                    scan_pts_x.append(sx)
                    scan_pts_y.append(sy)
            self.ax.scatter(scan_pts_x, scan_pts_y, c='b', s=5, label='Lidar Points')
        # 绘制DWA预测轨迹（绿色折线）
        if predicted_traj is not None and len(predicted_traj) >= 2:
            try:
                px = predicted_traj[:, 0]
                py = predicted_traj[:, 1]
                self.ax.plot(px, py, "-g", linewidth=2, alpha=0.8, label="Predicted Traj")
            except Exception:
                pass
        # 绘制实际轨迹（橙色折线）
        if actual_traj is not None and len(actual_traj) >= 2:
            try:
                traj_arr = np.asarray(actual_traj, dtype=float)
                style = {
                    'color': 'orange',
                    'linewidth': 1.2,
                    'alpha': 0.9,
                    'label': 'Actual Traj'
                }
                if isinstance(actual_traj_style, dict):
                    style.update(actual_traj_style)
                self.ax.plot(traj_arr[:, 0], traj_arr[:, 1], **style)
            except Exception:
                pass
        if extra_trajs:
            for entry in extra_trajs:
                try:
                    pts = entry.get('points', None)
                    if pts is None or len(pts) < 2:
                        continue
                    pts_arr = np.asarray(pts, dtype=float)
                    style = {
                        'color': 'orange',
                        'linewidth': 1.2,
                        'alpha': 0.8,
                        'label': 'Trajectory'
                    }
                    custom_style = entry.get('style')
                    if isinstance(custom_style, dict):
                        style.update(custom_style)
                    self.ax.plot(pts_arr[:, 0], pts_arr[:, 1], **style)
                except Exception:
                    continue
        # 绘制机器人当前位置和朝向 (箭头表示朝向)
        arrow_length = 0.5
        self.ax.arrow(x, y, arrow_length * math.cos(theta), arrow_length * math.sin(theta),
                      head_width=0.2, head_length=0.2, fc='r', ec='r')
        self.ax.scatter([x], [y], c='r')  # 机器人位置
        # 机器人圆形边界（若提供半径）
        if robot_radius is not None and robot_radius > 0:
            try:
                circle = Circle((x, y), robot_radius, edgecolor='c', facecolor='none', linewidth=1.5, alpha=0.9)
                self.ax.add_artist(circle)
                # 朝向指示到圆周
                hx = x + robot_radius * math.cos(theta)
                hy = y + robot_radius * math.sin(theta)
                self.ax.plot([x, hx], [y, hy], color='c', linewidth=1.2)
            except Exception:
                pass
        # 图例和标题
        self.ax.set_title("SLAM Exploration")
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.legend(loc='upper right')
        plt.draw()
        plt.pause(VISUALIZATION_UPDATE_TIME)  
        
    def set_emergency_path(self, path):
        """设置紧急路径（无安全距离的最短路径），用红色线条显示"""
        self.emergency_path = path

    def set_obstacle_search_region(self, bounds):
        """设置障碍物搜索范围的世界坐标边界 (min_x, min_y, max_x, max_y)。"""
        if bounds is None:
            self.obstacle_search_region = None
            return
        try:
            min_x, min_y, max_x, max_y = bounds
            if max_x <= min_x or max_y <= min_y:
                return
            self.obstacle_search_region = (float(min_x), float(min_y), float(max_x), float(max_y))
        except Exception:
            pass

    def set_bfs_debug_points(self, points, color='cyan'):
        """设置BFS调试可视化点（栅格坐标列表）。传入None清除。"""
        if points is None:
            self.bfs_debug_points = None
            return
        try:
            self.bfs_debug_points = list(points)
            if color:
                self.bfs_debug_color = color
        except Exception:
            self.bfs_debug_points = None

    def save_map(self, filename):
        """将当前地图绘制保存为图像文件。"""
        self.fig.savefig(filename)

    def save_path(self, filename):
        """将机器人行驶路径保存为CSV文件。"""
        if self.robot is None:
            return
        try:
            with open(filename, 'w') as f:
                f.write("x,y\n")
                for (x, y) in self.robot.trajectory:
                    f.write(f"{x:.3f},{y:.3f}\n")
        except Exception as e:
            print("Error saving path:", e)

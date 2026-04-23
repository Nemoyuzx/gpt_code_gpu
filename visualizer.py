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
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
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
        # 持久 artist 缓存（避免每帧 cla()+重建）
        self._img_artist = None
        self._scan_artist = None
        self._path_artist = None
        self._emergency_artist = None
        self._predicted_artist = None
        self._actual_artist = None
        self._extra_artists = []
        self._target_artist = None
        self._robot_pt_artist = None
        self._robot_arrow_artist = None
        self._robot_circle_artist = None
        self._robot_heading_artist = None
        self._obstacle_rect_artist = None
        self._bfs_scatter_artist = None
        self._legend_done = False
        # 占用栅格节流：imshow.set_data 每帧约十几 ms，且占用图变化缓慢，
        # 每 N 帧才真正刷新一次（其他动态艺术家仍每帧更新，不降帧率）。
        self._occ_update_interval = 3
        self._occ_frame_counter = 0
        self._last_occ_id = None

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
        更新绘制当前状态（持久 artist + set_data，避免 cla() 重建）。
        """
        x, y, theta = robot_pose
        ax = self.ax
        # --- 栅格地图：复用 AxesImage，仅 set_data ---
        if occupancy is not None:
            min_x, min_y, max_x, max_y = self.maze.bounds
            extent = (min_x, max_x, min_y, max_y)
            if self._img_artist is None:
                # 首帧必须创建 artist
                h, w = occupancy.shape
                display_grid = np.full((h, w), 0.5, dtype=float)
                display_grid[occupancy == 0] = 1.0
                display_grid[occupancy == 1] = 0.0
                self._img_artist = ax.imshow(display_grid, origin='lower', cmap='gray',
                                             extent=extent, vmin=0.0, vmax=1.0, zorder=0,
                                             interpolation='nearest')
                self._last_occ_id = id(occupancy)
            else:
                # 节流：每 N 帧才真正刷新 imshow 数据（占用变化缓慢，节流看不出差异）
                self._occ_frame_counter += 1
                if self._occ_frame_counter >= self._occ_update_interval:
                    self._occ_frame_counter = 0
                    h, w = occupancy.shape
                    display_grid = np.full((h, w), 0.5, dtype=float)
                    display_grid[occupancy == 0] = 1.0
                    display_grid[occupancy == 1] = 0.0
                    self._img_artist.set_data(display_grid)

        # --- 障碍物搜索区域边界 ---
        if self.obstacle_search_region is not None:
            min_x, min_y, max_x, max_y = self.obstacle_search_region
            if self._obstacle_rect_artist is None:
                self._obstacle_rect_artist = Rectangle(
                    (min_x, min_y), max_x - min_x, max_y - min_y,
                    linewidth=1.8, edgecolor='yellow', facecolor='none', linestyle='--',
                    label='Obstacle Search Bounds')
                ax.add_patch(self._obstacle_rect_artist)
            else:
                self._obstacle_rect_artist.set_bounds(min_x, min_y, max_x - min_x, max_y - min_y)
                self._obstacle_rect_artist.set_visible(True)
        elif self._obstacle_rect_artist is not None:
            self._obstacle_rect_artist.set_visible(False)

        # --- BFS 调试点 ---
        if self.bfs_debug_points:
            try:
                pts = np.asarray(self.bfs_debug_points, dtype=float)
                if pts.ndim == 2 and pts.shape[0] > 0:
                    wx = self.maze.bounds[0] + (pts[:, 0] + 0.5) * self.maze.resolution
                    wy = self.maze.bounds[1] + (pts[:, 1] + 0.5) * self.maze.resolution
                    coords = np.column_stack([wx, wy])
                    if self._bfs_scatter_artist is None:
                        self._bfs_scatter_artist = ax.scatter(wx, wy, s=12, c=self.bfs_debug_color,
                                                              alpha=0.25, marker='s', label='BFS Region')
                    else:
                        self._bfs_scatter_artist.set_offsets(coords)
                        self._bfs_scatter_artist.set_visible(True)
            except Exception:
                pass
        elif self._bfs_scatter_artist is not None:
            self._bfs_scatter_artist.set_visible(False)

        # --- 目标前沿 ---
        if target:
            tx = self.maze.bounds[0] + (target[0] + 0.5) * self.maze.resolution
            ty = self.maze.bounds[1] + (target[1] + 0.5) * self.maze.resolution
            if self._target_artist is None:
                self._target_artist, = ax.plot([tx], [ty], c='r', marker='*',
                                               markersize=12, linestyle='None', label='Target Frontier')
            else:
                self._target_artist.set_data([tx], [ty])
                self._target_artist.set_visible(True)
        elif self._target_artist is not None:
            self._target_artist.set_visible(False)

        # --- 规划路径 ---
        if path and len(path) > 1:
            px = [self.maze.bounds[0] + (ix + 0.5) * self.maze.resolution for (ix, iy) in path]
            py = [self.maze.bounds[1] + (iy + 0.5) * self.maze.resolution for (ix, iy) in path]
            if self._path_artist is None:
                self._path_artist, = ax.plot(px, py, color='g', linestyle='--', label='Path')
            else:
                self._path_artist.set_data(px, py)
                self._path_artist.set_visible(True)
        elif self._path_artist is not None:
            self._path_artist.set_visible(False)

        # --- 紧急路径 ---
        if self.emergency_path and len(self.emergency_path) > 1:
            epx = [self.maze.bounds[0] + (ix + 0.5) * self.maze.resolution for (ix, iy) in self.emergency_path]
            epy = [self.maze.bounds[1] + (iy + 0.5) * self.maze.resolution for (ix, iy) in self.emergency_path]
            if self._emergency_artist is None:
                self._emergency_artist, = ax.plot(epx, epy, color='red', linewidth=2,
                                                  label='Emergency Path (No Safety)')
            else:
                self._emergency_artist.set_data(epx, epy)
                self._emergency_artist.set_visible(True)
        elif self._emergency_artist is not None:
            self._emergency_artist.set_visible(False)

        # --- 激光雷达扫描点云（向量化）---
        if scan is not None and self.slam and len(scan) > 0:
            scan_arr = np.asarray(scan, dtype=float)
            num_beams = scan_arr.shape[0]
            angles = theta + np.arange(num_beams) * (2.0 * math.pi / num_beams)
            mask = scan_arr < LIDAR_DISPLAY_MAX_RANGE
            if np.any(mask):
                sx = x + scan_arr[mask] * np.cos(angles[mask])
                sy = y + scan_arr[mask] * np.sin(angles[mask])
                if self._scan_artist is None:
                    self._scan_artist, = ax.plot(sx, sy, linestyle='None', marker='.',
                                                 markersize=3, color='b', label='Lidar Points')
                else:
                    self._scan_artist.set_data(sx, sy)
                    self._scan_artist.set_visible(True)
            elif self._scan_artist is not None:
                self._scan_artist.set_visible(False)

        # --- DWA 预测轨迹 ---
        if predicted_traj is not None and len(predicted_traj) >= 2:
            try:
                pxv = predicted_traj[:, 0]
                pyv = predicted_traj[:, 1]
                if self._predicted_artist is None:
                    self._predicted_artist, = ax.plot(pxv, pyv, "-g", linewidth=2,
                                                      alpha=0.8, label="Predicted Traj")
                else:
                    self._predicted_artist.set_data(pxv, pyv)
                    self._predicted_artist.set_visible(True)
            except Exception:
                pass
        elif self._predicted_artist is not None:
            self._predicted_artist.set_visible(False)

        # --- 实际轨迹 ---
        if actual_traj is not None and len(actual_traj) >= 2:
            try:
                traj_arr = np.asarray(actual_traj, dtype=float)
                if self._actual_artist is None:
                    style = {'color': 'orange', 'linewidth': 1.2, 'alpha': 0.9, 'label': 'Actual Traj'}
                    if isinstance(actual_traj_style, dict):
                        style.update(actual_traj_style)
                    self._actual_artist, = ax.plot(traj_arr[:, 0], traj_arr[:, 1], **style)
                else:
                    self._actual_artist.set_data(traj_arr[:, 0], traj_arr[:, 1])
                    if isinstance(actual_traj_style, dict):
                        for k, v in actual_traj_style.items():
                            if k == 'label':
                                self._actual_artist.set_label(v)
                            elif k == 'color':
                                self._actual_artist.set_color(v)
                            elif k == 'linewidth':
                                self._actual_artist.set_linewidth(v)
                            elif k == 'alpha':
                                self._actual_artist.set_alpha(v)
                    self._actual_artist.set_visible(True)
            except Exception:
                pass
        elif self._actual_artist is not None:
            self._actual_artist.set_visible(False)

        # --- 额外轨迹（探索段等）---
        # 复用已有条目，超出/不足时增删
        if extra_trajs:
            for i, entry in enumerate(extra_trajs):
                try:
                    pts = entry.get('points', None)
                    if pts is None or len(pts) < 2:
                        continue
                    pts_arr = np.asarray(pts, dtype=float)
                    if i < len(self._extra_artists):
                        art = self._extra_artists[i]
                        art.set_data(pts_arr[:, 0], pts_arr[:, 1])
                        art.set_visible(True)
                    else:
                        style = {'color': 'orange', 'linewidth': 1.2, 'alpha': 0.8, 'label': 'Trajectory'}
                        custom_style = entry.get('style')
                        if isinstance(custom_style, dict):
                            style.update(custom_style)
                        art, = ax.plot(pts_arr[:, 0], pts_arr[:, 1], **style)
                        self._extra_artists.append(art)
                except Exception:
                    continue
            # 多余的隐藏
            for j in range(len(extra_trajs), len(self._extra_artists)):
                self._extra_artists[j].set_visible(False)
        else:
            for art in self._extra_artists:
                art.set_visible(False)

        # --- 机器人位置与朝向 ---
        arrow_length = 0.5
        # matplotlib Arrow 不能 set_data，改用 FancyArrow：每帧移除重建开销比线段大，改用 plot 线表示箭头
        hx_tip = x + arrow_length * math.cos(theta)
        hy_tip = y + arrow_length * math.sin(theta)
        if self._robot_arrow_artist is None:
            self._robot_arrow_artist, = ax.plot([x, hx_tip], [y, hy_tip],
                                                color='r', linewidth=2.0)
        else:
            self._robot_arrow_artist.set_data([x, hx_tip], [y, hy_tip])
        if self._robot_pt_artist is None:
            self._robot_pt_artist, = ax.plot([x], [y], linestyle='None', marker='o',
                                             markersize=6, color='r')
        else:
            self._robot_pt_artist.set_data([x], [y])

        # --- 机器人半径圆 + 朝向线 ---
        if robot_radius is not None and robot_radius > 0:
            if self._robot_circle_artist is None:
                self._robot_circle_artist = Circle((x, y), robot_radius, edgecolor='c',
                                                   facecolor='none', linewidth=1.5, alpha=0.9)
                ax.add_artist(self._robot_circle_artist)
            else:
                self._robot_circle_artist.center = (x, y)
                self._robot_circle_artist.set_radius(robot_radius)
                self._robot_circle_artist.set_visible(True)
            hx = x + robot_radius * math.cos(theta)
            hy = y + robot_radius * math.sin(theta)
            if self._robot_heading_artist is None:
                self._robot_heading_artist, = ax.plot([x, hx], [y, hy], color='c', linewidth=1.2)
            else:
                self._robot_heading_artist.set_data([x, hx], [y, hy])
                self._robot_heading_artist.set_visible(True)
        else:
            if self._robot_circle_artist is not None:
                self._robot_circle_artist.set_visible(False)
            if self._robot_heading_artist is not None:
                self._robot_heading_artist.set_visible(False)

        # --- 标题与图例：只在第一次绘制时设置 legend，避免每帧重建 ---
        if not self._legend_done:
            ax.set_title("SLAM Exploration")
            ax.set_aspect('equal', adjustable='box')
            try:
                ax.legend(loc='upper right', fontsize=8)
            except Exception:
                pass
            self._legend_done = True

        # plt.pause 会内部调用 draw_idle + start_event_loop，
        # 强制每帧都走一次事件循环处理渲染队列，避免 draw_idle/flush_events
        # 组合下出现的奇偶帧批量（0.3ms / 30ms 交替）。
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.start_event_loop(0.001)
        except Exception:
            self.fig.canvas.flush_events()
        
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

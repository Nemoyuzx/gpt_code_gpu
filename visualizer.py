import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
from matplotlib.patches import Patch
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
        # 边界和刻度
        min_x, min_y, max_x, max_y = maze.bounds
        self.ax.set_xlim(min_x, max_x)
        self.ax.set_ylim(min_y, max_y)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title("SLAM Exploration - Initializing...")
        self.ax.grid(True, alpha=0.3)
        # 立即显示窗口
        plt.show(block=False)
        plt.pause(0.01)
        self.paused = False
        # 自动控制开关（按's'启动，按'p'停止）
        self.auto_control_enabled = False
        self.force_stop_requested = False
        # 按键事件绑定
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        # 禁用matplotlib默认的快捷键（特别是's'键保存功能）
        plt.rcParams['keymap.save'] = []  # 禁用's'键的默认保存功能
        plt.rcParams['keymap.quit'] = []  # 禁用'q'键的默认退出功能
        # 存储紧急路径（红色显示）
        self.emergency_path = None
        self.obstacle_search_region = None
        self.bfs_debug_points = None
        self.bfs_debug_color = 'cyan'
        # 网格系统
        self.grid_system = None
        self.target_cell_id = None

    def _on_key_press(self, event):
        """
        键盘事件回调：
        's' - 启动自动控制（开始发送电机命令，不保存地图）
        'p' - 强制停止小车（发送 CMD-SET 0 0）
        'm' - 保存地图和路径
        '空格' - 暂停/继续可视化更新
        """
        if event.key == 's':
            if not self.auto_control_enabled:
                self.auto_control_enabled = True
                print("[Visualizer] ✅ 自动控制已启动！现在将开始发送电机命令。")
                print("[Visualizer] 💡 提示：按 'm' 键可随时保存地图和路径")
            else:
                print("[Visualizer] ⚠️  自动控制已经启用，无需重复按键。按 'p' 可停止。")
        elif event.key == 'p':
            self.force_stop_requested = True
            if self.auto_control_enabled:
                self.auto_control_enabled = False
                print("[Visualizer] 🛑 强制停止！小车将立即停止并关闭自动控制。")
            else:
                print("[Visualizer] 🛑 强制停止！小车将立即停止。")
        elif event.key == 'm':
            # 保存地图和路径
            map_file = "map.png"
            path_file = "path.csv"
            self.save_map(map_file)
            self.save_path(path_file)
            print(f"[Visualizer] 💾 地图已保存至 {map_file}, 路径已保存至 {path_file}")
        elif event.key == ' ':
            # 空格键：暂停/继续可视化
            self.paused = not self.paused
            if self.paused:
                print("[Visualizer] ⏸️  可视化已暂停（控制循环继续运行）")
            else:
                print("[Visualizer] ▶️  可视化已继续")
    
    def is_auto_control_enabled(self):
        """返回是否启用自动控制"""
        return self.auto_control_enabled
    
    def check_and_clear_force_stop(self):
        """检查并清除强制停止请求，返回是否有停止请求"""
        if self.force_stop_requested:
            self.force_stop_requested = False
            return True
        return False
    
    def set_grid_system(self, grid_system, target_cell_id):
        """设置网格系统用于可视化"""
        self.grid_system = grid_system
        self.target_cell_id = target_cell_id

    def update(self, robot_pose, scan, frontiers=None, target=None, path=None, occupancy=None,
               predicted_traj=None, robot_radius=None, safety_radius=None,
               actual_traj=None, actual_traj_style=None, extra_trajs=None,
               scan_angles=None, unsafe_mask=None, dwa_eval_paths=None):
        self._render_update(
            robot_pose,
            scan,
            frontiers=frontiers,
            target=target,
            path=path,
            occupancy=occupancy,
            predicted_traj=predicted_traj,
            robot_radius=robot_radius,
            safety_radius=safety_radius,
            actual_traj=actual_traj,
            actual_traj_style=actual_traj_style,
            extra_trajs=extra_trajs,
            scan_angles=scan_angles,
            unsafe_mask=unsafe_mask,
            dwa_eval_paths=dwa_eval_paths,
        )

    def _render_update(self, robot_pose, scan, frontiers=None, target=None, path=None, occupancy=None,
                       predicted_traj=None, robot_radius=None, safety_radius=None,
                       actual_traj=None, actual_traj_style=None, extra_trajs=None,
                       scan_angles=None, unsafe_mask=None, dwa_eval_paths=None):
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
    safety_radius: 膨胀后的安全边界半径（米），用于绘制虚线警戒圈（可选）。
        """
        x, y, theta = robot_pose
        # 清除之前的绘图
        self.ax.cla()
        # 绘制栅格地图
        view_min_x, view_min_y, view_max_x, view_max_y = self.maze.bounds
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
            if unsafe_mask is not None:
                # 保留遮挡逻辑入口，但实际绘制关闭以避免红色安全区覆盖视图
                pass
            known_mask = occupancy != -1
            if np.any(known_mask):
                ys, xs = np.nonzero(known_mask)
                res = self.maze.resolution
                view_min_x = min_x + xs.min() * res
                view_max_x = min_x + (xs.max() + 1) * res
                view_min_y = min_y + ys.min() * res
                view_max_y = min_y + (ys.max() + 1) * res
                margin = max(0.3, 2.0 * res)
                view_min_x -= margin
                view_max_x += margin
                view_min_y -= margin
                view_max_y += margin
        
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
        # 注意：即使 self.slam 是 None（多进程模式），也可以绘制扫描数据
        if scan:
            scan_pts_x = []
            scan_pts_y = []
            num_beams = len(scan)
            max_range = LIDAR_DISPLAY_MAX_RANGE  # 使用固定的最大范围
            angle_offset = getattr(self.slam, "laser_angle_offset", 0.0) if self.slam else 0.0
            for i, dist in enumerate(scan):
                if dist < max_range:
                    if scan_angles is not None and i < len(scan_angles) and math.isfinite(scan_angles[i]):
                        beam_angle = theta + angle_offset + math.radians(scan_angles[i])
                    else:
                        beam_angle = theta + angle_offset + i * (2.0 * math.pi / max(1, num_beams))
                    sx = x + dist * math.cos(beam_angle)
                    sy = y + dist * math.sin(beam_angle)
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
        # 绘制DWA采样评估范围（灰色线簇）
        if dwa_eval_paths:
            try:
                segments = []
                for pts in dwa_eval_paths:
                    arr = np.asarray(pts, dtype=float)
                    if arr.ndim != 2 or arr.shape[0] < 2:
                        continue
                    segments.append(arr[:, :2])
                if segments:
                    lc = LineCollection(segments, colors=(0.6, 0.6, 0.6, 0.4), linewidths=0.6)
                    lc.set_label('DWA Evaluated Traj')
                    self.ax.add_collection(lc)
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
        
        # 绘制网格系统
        if self.grid_system is not None:
            for cell in self.grid_system.cells:
                min_x, min_y, max_x, max_y = cell.get_bounds()
                # 根据单元格类型选择颜色和样式
                if cell.id == self.target_cell_id:
                    # 目标单元格：绿色粗边框，实线
                    edgecolor = 'lime'
                    linewidth = 2.5
                    alpha = 0.9
                    linestyle = '-'  # 实线
                elif cell.id == self.grid_system.get_center_cell().id:
                    # 起始单元格（中心）：青色边框，实线
                    edgecolor = 'cyan'
                    linewidth = 2.0
                    alpha = 0.8
                    linestyle = '-'  # 实线
                else:
                    # 普通单元格：灰色细边框，虚线
                    edgecolor = 'gray'
                    linewidth = 0.8
                    alpha = 0.5
                    linestyle = '--'  # 虚线
                
                rect = Rectangle(
                    (min_x, min_y),
                    max_x - min_x,
                    max_y - min_y,
                    linewidth=linewidth,
                    edgecolor=edgecolor,
                    facecolor='none',
                    alpha=alpha,
                    linestyle=linestyle
                )
                self.ax.add_patch(rect)
                
                # 在单元格中心显示序号
                text_color = 'lime' if cell.id == self.target_cell_id else 'cyan' if cell.id == self.grid_system.get_center_cell().id else 'lightgray'
                text_alpha = 1.0 if cell.id == self.target_cell_id or cell.id == self.grid_system.get_center_cell().id else 0.7
                text_size = 9 if cell.id == self.target_cell_id or cell.id == self.grid_system.get_center_cell().id else 7
                self.ax.text(
                    cell.center_x,
                    cell.center_y,
                    str(cell.id),
                    fontsize=text_size,
                    ha='center',
                    va='center',
                    color=text_color,
                    alpha=text_alpha,
                    weight='bold' if cell.id == self.target_cell_id or cell.id == self.grid_system.get_center_cell().id else 'normal',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3, edgecolor='none') if cell.id == self.target_cell_id or cell.id == self.grid_system.get_center_cell().id else None
                )
        
        # 绘制机器人当前位置和朝向 (箭头表示朝向)
        arrow_length = 0.3
        self.ax.arrow(
            x,
            y,
            arrow_length * math.cos(theta),
            arrow_length * math.sin(theta),
            head_width=0.12,
            head_length=0.14,
            fc='r',
            ec='r'
        )
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
        # 安全半径圈显示已停用（避免靠墙时额外圈层遮挡）
        # 图例和标题
        control_status = "🚗 Auto Control Active" if self.auto_control_enabled else "📍 Mapping Only (press 's' to start)"
        self.ax.set_title(f"SLAM Exploration - {control_status}")
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(view_min_x, view_max_x)
        self.ax.set_ylim(view_min_y, view_max_y)
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

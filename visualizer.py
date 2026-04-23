import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
from matplotlib.patches import Patch
import math
import numpy as np
from output_paths import MAP_IMAGE, PATH_CSV, ensure_parent

# 可视化参数
# 使用 draw_idle + flush_events 驱动事件循环，pause 会引入 >= 1ms 的额外睡眠；默认关闭
VISUALIZATION_UPDATE_TIME = 0.0  # 可视化更新时间；>0 时每帧额外 plt.pause 该秒数
LIDAR_DISPLAY_MAX_RANGE = 12.0  # 激光雷达显示最大范围

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
        # --- 持久化 artist（避免每帧 cla + 重建） ---
        self._im_occ = None                 # 占据栅格 imshow
        self._occ_extent = None             # 上一次 extent，以便判定是否需重建
        self._occ_shape = None              # 上一次 occupancy.shape
        self._scan_scatter = None           # lidar 点 scatter
        self._target_scatter = None         # 目标前沿 scatter
        self._path_line = None              # 规划路径
        self._emergency_line = None         # 紧急路径
        self._predicted_line = None         # DWA 预测轨迹
        self._actual_line = None            # 实际轨迹
        self._robot_scatter = None          # 机器人位置 scatter
        self._robot_arrow = None            # 机器人朝向箭头（Line2D）
        self._robot_circle = None           # 机器人半径圆
        self._robot_heading_line = None     # 机器人朝向半径线
        self._dwa_collection = None         # DWA 评估轨迹 LineCollection
        self._obstacle_rect = None          # 障碍搜索区矩形
        self._bfs_scatter = None            # BFS 调试点
        self._extra_traj_lines = []         # 额外轨迹 Line2D 列表（可变数量）
        self._grid_artists = []             # 网格系统 Rectangle / Text 列表（可能多次更新）
        self._grid_signature = None         # 网格系统内容签名，用于判定是否重建
        self._legend_signature = None       # 图例签名，用于判定是否需要重建图例
        self._legend_obj = None             # 当前图例对象

    def _on_key_press(self, event):
        """
        键盘事件回调：
        's' - 启动自动控制（开始发送电机命令，不保存地图）
        'p' - 强制停止小车（发送 CMD-VW 0 0）
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
            map_file = MAP_IMAGE
            path_file = PATH_CSV
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

    # 空 (N,2) 数组作为 set_offsets 的占位，避免 matplotlib 对空列表报 warning
    _EMPTY_XY = np.empty((0, 2), dtype=float)

    def _update_or_create_line(self, attr_name, xs, ys, **style):
        line = getattr(self, attr_name)
        if xs is None or len(xs) == 0:
            if line is not None:
                line.set_visible(False)
            return
        if line is None:
            (line,) = self.ax.plot(xs, ys, **style)
            setattr(self, attr_name, line)
        else:
            line.set_data(xs, ys)
            # 若传入样式中包含常用属性，同步覆盖
            if 'color' in style:
                line.set_color(style['color'])
            if 'linewidth' in style:
                line.set_linewidth(style['linewidth'])
            if 'alpha' in style:
                line.set_alpha(style['alpha'])
            if 'linestyle' in style:
                line.set_linestyle(style['linestyle'])
            if 'label' in style:
                line.set_label(style['label'])
            line.set_visible(True)

    def _update_or_create_scatter(self, attr_name, xs, ys, **kwargs):
        sc = getattr(self, attr_name)
        if xs is None or len(xs) == 0:
            if sc is not None:
                sc.set_visible(False)
            return
        offsets = np.column_stack([xs, ys])
        if sc is None:
            sc = self.ax.scatter(xs, ys, **kwargs)
            setattr(self, attr_name, sc)
        else:
            sc.set_offsets(offsets)
            sc.set_visible(True)

    def _render_update(self, robot_pose, scan, frontiers=None, target=None, path=None, occupancy=None,
                       predicted_traj=None, robot_radius=None, safety_radius=None,
                       actual_traj=None, actual_traj_style=None, extra_trajs=None,
                       scan_angles=None, unsafe_mask=None, dwa_eval_paths=None):
        """增量更新持久化 artist（替代 cla + 完全重绘，显著提升 FPS）。"""
        x, y, theta = robot_pose
        maze_min_x, maze_min_y, maze_max_x, maze_max_y = self.maze.bounds
        view_min_x, view_min_y, view_max_x, view_max_y = self.maze.bounds
        res = self.maze.resolution

        # --- 1) 占据栅格：就地更新 imshow，避免 cla + 重建 ---
        if occupancy is not None:
            h, w = occupancy.shape
            # 向量化映射：未知=-1→0.5；空闲=0→1.0；占据=1→0.0
            display_grid = np.full((h, w), 0.5, dtype=np.float32)
            display_grid[occupancy == 0] = 1.0
            display_grid[occupancy == 1] = 0.0
            extent = (maze_min_x, maze_max_x, maze_min_y, maze_max_y)
            if (self._im_occ is None or self._occ_shape != (h, w) or self._occ_extent != extent):
                if self._im_occ is not None:
                    try:
                        self._im_occ.remove()
                    except Exception:
                        pass
                self._im_occ = self.ax.imshow(
                    display_grid, origin='lower', cmap='gray',
                    extent=extent, vmin=0.0, vmax=1.0, zorder=0,
                )
                self._occ_shape = (h, w)
                self._occ_extent = extent
            else:
                self._im_occ.set_data(display_grid)
            known_mask = occupancy != -1
            if known_mask.any():
                ys, xs = np.nonzero(known_mask)
                view_min_x = maze_min_x + xs.min() * res
                view_max_x = maze_min_x + (xs.max() + 1) * res
                view_min_y = maze_min_y + ys.min() * res
                view_max_y = maze_min_y + (ys.max() + 1) * res
                margin = max(0.3, 2.0 * res)
                view_min_x -= margin
                view_max_x += margin
                view_min_y -= margin
                view_max_y += margin

        # --- 2) 障碍搜索区域矩形 ---
        if self.obstacle_search_region is not None:
            omx, omy, oMx, oMy = self.obstacle_search_region
            if self._obstacle_rect is None:
                self._obstacle_rect = Rectangle(
                    (omx, omy), oMx - omx, oMy - omy,
                    linewidth=1.8, edgecolor='yellow',
                    facecolor='none', linestyle='--',
                    label='Obstacle Search Bounds',
                )
                self.ax.add_patch(self._obstacle_rect)
            else:
                self._obstacle_rect.set_bounds(omx, omy, oMx - omx, oMy - omy)
                self._obstacle_rect.set_visible(True)
        elif self._obstacle_rect is not None:
            self._obstacle_rect.set_visible(False)

        # --- 3) BFS 调试点 ---
        if self.bfs_debug_points:
            pts = np.asarray(self.bfs_debug_points, dtype=float)
            if pts.ndim == 2 and pts.shape[0] > 0:
                wx = maze_min_x + (pts[:, 0] + 0.5) * res
                wy = maze_min_y + (pts[:, 1] + 0.5) * res
                self._update_or_create_scatter(
                    '_bfs_scatter', wx, wy,
                    s=12, c=self.bfs_debug_color, alpha=0.25,
                    marker='s', label='BFS Region',
                )
            elif self._bfs_scatter is not None:
                self._bfs_scatter.set_visible(False)
        elif self._bfs_scatter is not None:
            self._bfs_scatter.set_visible(False)

        # --- 4) 目标前沿 ---
        if target:
            tx = maze_min_x + (target[0] + 0.5) * res
            ty = maze_min_y + (target[1] + 0.5) * res
            self._update_or_create_scatter(
                '_target_scatter', [tx], [ty],
                c='r', marker='*', s=100, label='Target Frontier',
            )
        elif self._target_scatter is not None:
            self._target_scatter.set_visible(False)

        # --- 5) 规划路径 ---
        if path and len(path) > 1:
            path_arr = np.asarray(path, dtype=float)
            px = maze_min_x + (path_arr[:, 0] + 0.5) * res
            py = maze_min_y + (path_arr[:, 1] + 0.5) * res
            self._update_or_create_line('_path_line', px, py,
                                         color='g', linestyle='--', label='Path')
        elif self._path_line is not None:
            self._path_line.set_visible(False)

        # --- 6) 紧急路径 ---
        if self.emergency_path and len(self.emergency_path) > 1:
            ep_arr = np.asarray(self.emergency_path, dtype=float)
            epx = maze_min_x + (ep_arr[:, 0] + 0.5) * res
            epy = maze_min_y + (ep_arr[:, 1] + 0.5) * res
            self._update_or_create_line(
                '_emergency_line', epx, epy,
                color='red', linewidth=2, linestyle='-',
                label='Emergency Path (No Safety)',
            )
        elif self._emergency_line is not None:
            self._emergency_line.set_visible(False)

        # --- 7) 激光扫描：向量化计算点云坐标 ---
        if scan:
            scan_np = np.asarray(scan, dtype=np.float32)
            max_range = LIDAR_DISPLAY_MAX_RANGE
            valid = (scan_np < max_range) & np.isfinite(scan_np)
            if valid.any():
                num_beams = scan_np.shape[0]
                angle_offset = getattr(self.slam, "laser_angle_offset", 0.0) if self.slam else 0.0
                if scan_angles is not None and len(scan_angles) == num_beams:
                    ang_np = np.asarray(scan_angles, dtype=np.float32)
                    finite_ang = np.isfinite(ang_np)
                    ang_rad = np.where(finite_ang, np.radians(ang_np),
                                        np.arange(num_beams) * (2.0 * math.pi / max(1, num_beams)))
                else:
                    ang_rad = np.arange(num_beams, dtype=np.float32) * (2.0 * math.pi / max(1, num_beams))
                beam_angle = theta + angle_offset + ang_rad
                sx_arr = x + scan_np * np.cos(beam_angle)
                sy_arr = y + scan_np * np.sin(beam_angle)
                sx_v = sx_arr[valid]
                sy_v = sy_arr[valid]
                self._update_or_create_scatter(
                    '_scan_scatter', sx_v, sy_v,
                    c='b', s=5, label='Lidar Points',
                )
            elif self._scan_scatter is not None:
                self._scan_scatter.set_visible(False)
        elif self._scan_scatter is not None:
            self._scan_scatter.set_visible(False)

        # --- 8) DWA 预测轨迹 ---
        if predicted_traj is not None and len(predicted_traj) >= 2:
            try:
                pt_arr = np.asarray(predicted_traj)
                self._update_or_create_line(
                    '_predicted_line', pt_arr[:, 0], pt_arr[:, 1],
                    color='g', linewidth=2, alpha=0.8, linestyle='-',
                    label='Predicted Traj',
                )
            except Exception:
                pass
        elif self._predicted_line is not None:
            self._predicted_line.set_visible(False)

        # --- 9) DWA 评估轨迹 LineCollection ---
        if dwa_eval_paths:
            segments = []
            for pts in dwa_eval_paths:
                arr = np.asarray(pts, dtype=float)
                if arr.ndim == 2 and arr.shape[0] >= 2:
                    segments.append(arr[:, :2])
            if segments:
                if self._dwa_collection is None:
                    self._dwa_collection = LineCollection(
                        segments, colors=(0.6, 0.6, 0.6, 0.4),
                        linewidths=0.6, label='DWA Evaluated Traj',
                    )
                    self.ax.add_collection(self._dwa_collection)
                else:
                    self._dwa_collection.set_segments(segments)
                    self._dwa_collection.set_visible(True)
            elif self._dwa_collection is not None:
                self._dwa_collection.set_visible(False)
        elif self._dwa_collection is not None:
            self._dwa_collection.set_visible(False)

        # --- 10) 实际轨迹 ---
        if actual_traj is not None and len(actual_traj) >= 2:
            try:
                traj_arr = np.asarray(actual_traj, dtype=float)
                style = {'color': 'orange', 'linewidth': 1.2,
                         'alpha': 0.9, 'label': 'Actual Traj'}
                if isinstance(actual_traj_style, dict):
                    style.update(actual_traj_style)
                self._update_or_create_line(
                    '_actual_line', traj_arr[:, 0], traj_arr[:, 1],
                    **style,
                )
            except Exception:
                pass
        elif self._actual_line is not None:
            self._actual_line.set_visible(False)

        # --- 11) 额外轨迹：数量可变，先复用，再补创建，多余隐藏 ---
        extras = list(extra_trajs) if extra_trajs else []
        for idx, entry in enumerate(extras):
            try:
                pts = entry.get('points', None) if isinstance(entry, dict) else None
                if pts is None or len(pts) < 2:
                    continue
                pts_arr = np.asarray(pts, dtype=float)
                style = {'color': 'orange', 'linewidth': 1.2,
                         'alpha': 0.8, 'label': 'Trajectory'}
                custom = entry.get('style') if isinstance(entry, dict) else None
                if isinstance(custom, dict):
                    style.update(custom)
                if idx < len(self._extra_traj_lines):
                    line = self._extra_traj_lines[idx]
                    line.set_data(pts_arr[:, 0], pts_arr[:, 1])
                    line.set_color(style['color'])
                    line.set_linewidth(style['linewidth'])
                    line.set_alpha(style['alpha'])
                    line.set_label(style['label'])
                    if 'linestyle' in style:
                        line.set_linestyle(style['linestyle'])
                    line.set_visible(True)
                else:
                    (line,) = self.ax.plot(pts_arr[:, 0], pts_arr[:, 1], **style)
                    self._extra_traj_lines.append(line)
            except Exception:
                continue
        # 未用到的旧 extra 线隐藏
        for j in range(len(extras), len(self._extra_traj_lines)):
            self._extra_traj_lines[j].set_visible(False)

        # --- 12) 网格系统：仅在签名变化时重建（标签变化很少） ---
        if self.grid_system is not None:
            center_id = self.grid_system.get_center_cell().id
            signature = (id(self.grid_system), center_id, self.target_cell_id, len(self.grid_system.cells))
            if signature != self._grid_signature:
                # 清掉旧 artist
                for art in self._grid_artists:
                    try:
                        art.remove()
                    except Exception:
                        pass
                self._grid_artists.clear()
                for cell in self.grid_system.cells:
                    cmn_x, cmn_y, cmx_x, cmx_y = cell.get_bounds()
                    is_target = (cell.id == self.target_cell_id)
                    is_center = (cell.id == center_id)
                    if is_target:
                        edgecolor, lw, al, ls = 'lime', 2.5, 0.9, '-'
                    elif is_center:
                        edgecolor, lw, al, ls = 'cyan', 2.0, 0.8, '-'
                    else:
                        edgecolor, lw, al, ls = 'gray', 0.8, 0.5, '--'
                    rect = Rectangle(
                        (cmn_x, cmn_y), cmx_x - cmn_x, cmx_y - cmn_y,
                        linewidth=lw, edgecolor=edgecolor,
                        facecolor='none', alpha=al, linestyle=ls,
                    )
                    self.ax.add_patch(rect)
                    self._grid_artists.append(rect)
                    text_color = 'lime' if is_target else 'cyan' if is_center else 'lightgray'
                    text_alpha = 1.0 if (is_target or is_center) else 0.7
                    text_size = 9 if (is_target or is_center) else 7
                    txt = self.ax.text(
                        cell.center_x, cell.center_y, str(cell.id),
                        fontsize=text_size, ha='center', va='center',
                        color=text_color, alpha=text_alpha,
                        weight='bold' if (is_target or is_center) else 'normal',
                        bbox=(dict(boxstyle='round,pad=0.3', facecolor='black',
                                   alpha=0.3, edgecolor='none')
                              if (is_target or is_center) else None),
                    )
                    self._grid_artists.append(txt)
                self._grid_signature = signature
        elif self._grid_artists:
            for art in self._grid_artists:
                try:
                    art.remove()
                except Exception:
                    pass
            self._grid_artists.clear()
            self._grid_signature = None

        # --- 13) 机器人位置 / 朝向 / 半径圆 ---
        arrow_length = 0.3
        hx = x + arrow_length * math.cos(theta)
        hy = y + arrow_length * math.sin(theta)
        if self._robot_arrow is None:
            (self._robot_arrow,) = self.ax.plot(
                [x, hx], [y, hy], color='r', linewidth=1.8,
            )
        else:
            self._robot_arrow.set_data([x, hx], [y, hy])
            self._robot_arrow.set_visible(True)
        if self._robot_scatter is None:
            self._robot_scatter = self.ax.scatter([x], [y], c='r', s=25, zorder=6)
        else:
            self._robot_scatter.set_offsets(np.array([[x, y]]))
            self._robot_scatter.set_visible(True)

        if robot_radius is not None and robot_radius > 0:
            if self._robot_circle is None:
                self._robot_circle = Circle(
                    (x, y), robot_radius, edgecolor='c',
                    facecolor='none', linewidth=1.5, alpha=0.9,
                )
                self.ax.add_patch(self._robot_circle)
            else:
                self._robot_circle.center = (x, y)
                self._robot_circle.set_radius(robot_radius)
                self._robot_circle.set_visible(True)
            bx = x + robot_radius * math.cos(theta)
            by = y + robot_radius * math.sin(theta)
            if self._robot_heading_line is None:
                (self._robot_heading_line,) = self.ax.plot(
                    [x, bx], [y, by], color='c', linewidth=1.2,
                )
            else:
                self._robot_heading_line.set_data([x, bx], [y, by])
                self._robot_heading_line.set_visible(True)
        else:
            if self._robot_circle is not None:
                self._robot_circle.set_visible(False)
            if self._robot_heading_line is not None:
                self._robot_heading_line.set_visible(False)

        # --- 14) 标题 / 视图范围 / 图例（仅在签名变化时重建） ---
        control_status = "🚗 Auto Control Active" if self.auto_control_enabled else "📍 Mapping Only (press 's' to start)"
        self.ax.set_title(f"SLAM Exploration - {control_status}")
        self.ax.set_xlim(view_min_x, view_max_x)
        self.ax.set_ylim(view_min_y, view_max_y)

        labeled = []
        for a in self.ax.get_children():
            if not hasattr(a, 'get_label') or not a.get_visible():
                continue
            lbl = a.get_label()
            if not isinstance(lbl, str) or not lbl or lbl.startswith('_'):
                continue
            labeled.append(lbl)
        legend_sig = tuple(sorted(set(labeled)))
        if legend_sig != self._legend_signature:
            self._legend_obj = self.ax.legend(loc='upper right')
            self._legend_signature = legend_sig

        # 使用 draw_idle + flush_events 替代 draw + pause，避免 pause 引入的额外睡眠
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass
        if VISUALIZATION_UPDATE_TIME > 0:
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
        self.fig.savefig(ensure_parent(filename))

    def save_path(self, filename):
        """将机器人行驶路径保存为CSV文件。"""
        if self.robot is None:
            return
        try:
            target = ensure_parent(filename)
            with target.open('w', encoding='utf-8') as f:
                f.write("x,y\n")
                for (x, y) in self.robot.trajectory:
                    f.write(f"{x:.3f},{y:.3f}\n")
        except Exception as e:
            print("Error saving path:", e)

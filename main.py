import time
import math
import os
import matplotlib.pyplot as plt
from maze_loader import MazeLoader
from robot import Robot
from lidar import Lidar
from icp_slam import ICPSlam
from frontier_explorer import FrontierExplorer
from visualizer import Visualizer

SAFETY_DISTANCE_FACTOR = 0.7  # 路径截断百分比，表示只执行路径的前80%

def check_exit_condition(scan, max_range=12.0, min_angle_range=180.0):
    """
    检查机器人是否走出迷宫
    当有超过180度的连续角度范围没有激光雷达返回数据时，认为已走出迷宫
    
    Args:
        scan: 激光雷达扫描数据列表
        max_range: 激光雷达最大探测距离
        min_angle_range: 最小连续角度范围（度），超过此范围认为走出迷宫
     
    Returns:
        bool: True表示已走出迷宫，False表示仍在迷宫内
    """
    if not scan or len(scan) == 0:
        return False
    
    # 将扫描数据转换为布尔数组，True表示该角度没有检测到障碍物
    no_obstacle = [dist >= max_range for dist in scan]
    
    # 寻找最长的连续True序列
    max_consecutive = 0
    current_consecutive = 0
    
    # 由于激光雷达是360度扫描，需要考虑环形连接
    # 先处理普通的连续序列
    for has_no_obstacle in no_obstacle:
        if has_no_obstacle:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    # 处理跨越0度的环形连续序列
    # 从开头开始计算连续的True
    start_consecutive = 0
    for has_no_obstacle in no_obstacle:
        if has_no_obstacle:
            start_consecutive += 1
        else:
            break
    
    # 从末尾开始计算连续的True
    end_consecutive = 0
    for has_no_obstacle in reversed(no_obstacle):
        if has_no_obstacle:
            end_consecutive += 1
        else:
            break
    
    # 如果开头和末尾都有连续的True，且它们可能是连接的
    if start_consecutive > 0 and end_consecutive > 0:
        # 检查是否整个扫描都是True（特殊情况）
        if start_consecutive + end_consecutive >= len(no_obstacle):
            max_consecutive = len(no_obstacle)
        else:
            # 跨越0度的连续长度
            wrap_around_consecutive = start_consecutive + end_consecutive
            max_consecutive = max(max_consecutive, wrap_around_consecutive)
    
    # 计算角度：假设激光雷达是360度均匀分布
    angle_per_scan = 360.0 / len(scan)
    max_angle_range = max_consecutive * angle_per_scan
    
    #print(f"最大连续无障碍角度范围: {max_angle_range:.1f}度 (阈值: {min_angle_range}度)")
    
    return max_angle_range >= min_angle_range

def main():
    # 检查是否有环境变量控制GPU使用
    device_type = os.environ.get("DEVICE_TYPE", "auto").lower()
    if device_type == "cpu":
        os.environ["FORCE_CPU"] = "1"
        print("由环境变量设置使用CPU模式")
    elif device_type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("由环境变量设置优先使用MPS (Apple Metal)")
    
    # 1. 加载迷宫地图和参数
    loader = MazeLoader()
    maze = loader.load()  # 加载默认迷宫（可修改为自定义配置或文件路径）
    start_x, start_y = maze.start
    # 初始朝向设为0（朝向x正方向）
    start_pose = (start_x, start_y, 0.0)
    # 安全移动参数
    safety_distance_factor = SAFETY_DISTANCE_FACTOR  # 路径截断百分比，表示只执行路径的前50%
    # 2. 初始化机器人、传感器、SLAM等模块
    robot = Robot(start_pose, odom_noise=(0,0))  # 设置一定里程计噪声
    lidar = Lidar(maze.walls, max_range=12.0, angle_resolution=1.0, noise=0)
    slam = ICPSlam(maze, start_pose)
    explorer = FrontierExplorer(safety_distance=6.0)  # 设置与障碍物的安全距离
    viz = Visualizer(maze, robot=robot, slam=slam)
    # 3. 初始扫描并建立初始地图
    scan = lidar.scan(robot.get_pose())
    slam.update((0.0, 0.0), scan)
    # 初始可视化
    robot_pose = robot.get_pose()
    frontiers = explorer.find_frontiers(slam.get_occupancy())
    target_cell = None
    path = None
    viz.update(robot_pose, scan, frontiers=frontiers, target=target_cell, path=path, occupancy=slam.get_occupancy())
    
    # 初始化探索状态标志和探索进度跟踪
    exploration_complete = False
    total_distance_traveled = 0.0  # 总移动距离
    frontiers_explored = 0  # 已探索的前沿数量
    min_exploration_distance = 20.0  # 最小探索距离阈值
    min_frontiers_to_explore = 3  # 最小探索前沿数量
    
    # 4. 前沿探索主循环
    while True:
        # 检查暂停状态
        if viz.paused:
            # 暂停循环，直到恢复
            plt.pause(0.005)  # 减少暂停时的等待时间，提高响应速度
            continue
        
        # 注意：路径规划会保持与障碍物的安全距离，防止穿墙
        # 只有在进行了充分探索后才检查是否检测到超过180度的连续无障碍区域
        scan = lidar.scan(robot.get_pose())
        has_sufficient_exploration = (total_distance_traveled >= min_exploration_distance and 
                                     frontiers_explored >= min_frontiers_to_explore)
        
        if has_sufficient_exploration and check_exit_condition(scan, max_range=lidar.max_range, min_angle_range=180.0):
            print(f"检测到超过180度的连续无障碍区域 - 已探索距离: {total_distance_traveled:.1f}m, 已探索前沿: {frontiers_explored}个")
            print("探索充分，开始返回起点！")
            # 更新SLAM和可视化
            slam.update((0.0, 0.0), scan)
            robot_pose = robot.get_pose()
            frontiers = explorer.find_frontiers(slam.get_occupancy())
            viz.update(robot_pose, scan, frontiers=frontiers, target=None, path=None, occupancy=slam.get_occupancy())
            # 设置标志表示需要返回起点
            should_return_to_start = True
            break
        
        # 查找最近的前沿
        # 获取机器人当前所在栅格索引
        rx_idx = int((robot.x - maze.bounds[0]) / maze.resolution)
        ry_idx = int((robot.y - maze.bounds[1]) / maze.resolution)
        frontier_cell, path = explorer.find_nearest_frontier(slam.get_occupancy(), (rx_idx, ry_idx))
        if frontier_cell is None:
            # 没有前沿但还未走出迷宫，可能是探索完成但仍在迷宫内
            print("没有更多前沿但仍在迷宫内 - 尝试寻找出口...")
            # 可以在这里添加寻找出口的逻辑，暂时继续执行原逻辑
            print("迷宫内部探索完成。")
            break
        
        # 打印路径信息来验证对角线移动
        print(f"Planning path to frontier {frontier_cell}")
        if path and len(path) > 1:
            diagonal_moves = 0
            for i in range(len(path) - 1):
                dx = abs(path[i+1][0] - path[i][0])
                dy = abs(path[i+1][1] - path[i][1])
                if dx == 1 and dy == 1:
                    diagonal_moves += 1
            print(f"Path length: {len(path)}, Diagonal moves: {diagonal_moves}")
        
        # 在可视化中标记当前目标前沿
        target_cell = frontier_cell
        # 增加已探索前沿计数
        frontiers_explored += 1
        print(f"开始探索第 {frontiers_explored} 个前沿点: {frontier_cell}")
        
        # 5. 沿规划路径移动机器人
        if path:  # 确保路径存在
            # 对路径进行百分比截断，只执行前safety_distance_factor比例的路径
            if len(path) > 1:  # 确保路径至少有两个点
                path_length = len(path) - 1  # 减去当前位置
                truncated_length = max(1, int(path_length * safety_distance_factor))  # 至少保留一步
                truncated_path = path[:truncated_length+1]  # +1是因为path[0]是当前位置
                print(f"路径截断: 原路径长度={path_length}，截断后长度={truncated_length} (保留{safety_distance_factor*100:.0f}%)")
                path = truncated_path  # 使用截断后的路径
            
            for step in path[1:]:  # path[0] 是当前位置
                # 计算目标栅格中心的世界坐标
                ix, iy = step
                target_x = maze.bounds[0] + (ix + 0.5) * maze.resolution
                target_y = maze.bounds[1] + (iy + 0.5) * maze.resolution
                # 计算需要旋转的角度
                dx = target_x - robot.x
                dy = target_y - robot.y
                desired_theta = math.atan2(dy, dx)
                # 计算最小旋转角度差
                d_theta = desired_theta - robot.theta
                # 将角度差规范化到[-pi, pi]
                d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
                if abs(d_theta) > 1e-3:
                    # 执行旋转
                    old_odom_theta = robot.odom_theta
                    robot.rotate(d_theta)
                    # 计算里程计角增量
                    dtheta_odom = robot.odom_theta - old_odom_theta
                    # 获取旋转后的扫描数据并更新SLAM
                    scan = lidar.scan(robot.get_pose())
                    slam.update((0.0, dtheta_odom), scan)
                    # 更新可视化
                    viz.update(robot.get_pose(), scan, frontiers=frontiers, target=target_cell, path=path, occupancy=slam.get_occupancy())
                # 前进到目标格中心，不再缩短距离
                distance = math.hypot(target_x - robot.x, target_y - robot.y)
                #print(f"目标点: ({target_x:.2f}, {target_y:.2f}), 距离: {distance:.2f}")
                if distance > 0:
                    # 执行移动，使用完整距离
                    old_odom_x, old_odom_y = robot.odom_x, robot.odom_y
                    robot.move(distance)  # 移动到目标位置
                    # 计算里程计距离增量（直线移动，朝向不变）
                    d_trans = math.hypot(robot.odom_x - old_odom_x, robot.odom_y - old_odom_y)
                    total_distance_traveled += d_trans  # 累计总移动距离
                    scan = lidar.scan(robot.get_pose())
                    slam.update((d_trans, 0.0), scan)
                    
                    # 只有在进行了充分探索后才检查180度条件
                    has_sufficient_exploration = (total_distance_traveled >= min_exploration_distance and 
                                                 frontiers_explored >= min_frontiers_to_explore)
                    
                    if has_sufficient_exploration and check_exit_condition(scan, max_range=lidar.max_range, min_angle_range=180.0):
                        print(f"移动过程中检测到超过180度连续无障碍区域！")
                        print(f"探索统计 - 总距离: {total_distance_traveled:.1f}m, 已探索前沿: {frontiers_explored}个")
                        robot_pose = robot.get_pose()
                        frontiers = explorer.find_frontiers(slam.get_occupancy())
                        viz.update(robot_pose, scan, frontiers=frontiers, target=target_cell, path=path, occupancy=slam.get_occupancy())
                        # 设置标志表示需要返回起点
                        should_return_to_start = True
                        break  # 跳出移动循环
                    
                    # 更新可视化
                    robot_pose = robot.get_pose()
                    frontiers = explorer.find_frontiers(slam.get_occupancy())
                    viz.update(robot_pose, scan, frontiers=frontiers, target=target_cell, path=path, occupancy=slam.get_occupancy())
            
            # 如果在移动过程中检测到需要返回起点的条件，跳出外层循环
            if 'should_return_to_start' in locals() and should_return_to_start:
                break
                
    # 根据探索结束的原因决定后续行为
    if 'should_return_to_start' in locals() and should_return_to_start:
        print(f"探索过程中检测到超过180度连续无障碍区域，现在返回起点...")
        print(f"探索总结 - 总移动距离: {total_distance_traveled:.1f}m, 总共探索了 {frontiers_explored} 个前沿点")
    else:
        print("探索完成：迷宫内部区域已完全探索。")
        print(f"探索总结 - 总移动距离: {total_distance_traveled:.1f}m, 总共探索了 {frontiers_explored} 个前沿点")
    
    # 无论什么情况都尝试返回起点（除非程序因其他原因结束）
    print("规划返回起点路径...")
    start_idx_x = int((maze.start[0] - maze.bounds[0]) / maze.resolution)
    start_idx_y = int((maze.start[1] - maze.bounds[1]) / maze.resolution)
    current_idx_x = int((robot.x - maze.bounds[0]) / maze.resolution)
    current_idx_y = int((robot.y - maze.bounds[1]) / maze.resolution)
    back_path = explorer.plan_path(slam.get_occupancy(), (current_idx_x, current_idx_y), (start_idx_x, start_idx_y))
    if back_path:
        print("Returning to start...")
        print(f"返回路径长度: {len(back_path)-1} 步，直接走到起点")
        
        for step in back_path[1:]:
                ix, iy = step
                target_x = maze.bounds[0] + (ix + 0.5) * maze.resolution
                target_y = maze.bounds[1] + (iy + 0.5) * maze.resolution
                dx = target_x - robot.x
                dy = target_y - robot.y
                desired_theta = math.atan2(dy, dx)
                d_theta = desired_theta - robot.theta
                d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
                if abs(d_theta) > 1e-3:
                    old_odom_theta = robot.odom_theta
                    robot.rotate(d_theta)
                    dtheta_odom = robot.odom_theta - old_odom_theta
                    # 返回时获取扫描数据但不用于SLAM建图，仅用于可视化
                    scan = lidar.scan(robot.get_pose())
                    # slam.update((0.0, dtheta_odom), scan)  # 注释掉SLAM更新
                    viz.update(robot.get_pose(), scan, frontiers=None, target=None, path=back_path, occupancy=slam.get_occupancy())
                distance = math.hypot(target_x - robot.x, target_y - robot.y)
                if distance > 1e-6:
                    # 执行移动，使用完整距离
                    old_odom_x, old_odom_y = robot.odom_x, robot.odom_y
                    robot.move(distance)
                    d_trans = math.hypot(robot.odom_x - old_odom_x, robot.odom_y - old_odom_y)
                    # 返回时获取扫描数据但不用于SLAM建图，仅用于可视化
                    scan = lidar.scan(robot.get_pose())
                    # slam.update((d_trans, 0.0), scan)  # 注释掉SLAM更新
                    viz.update(robot.get_pose(), scan, frontiers=None, target=None, path=back_path, occupancy=slam.get_occupancy())
        print("Robot returned to start.")
    else:
        print("无法规划返回起点的路径。")
    
    # 根据结束条件输出相应信息
    if 'should_return_to_start' in locals() and should_return_to_start:
        print("仿真结束：检测到超过180度连续无障碍区域，机器人已返回起点！")
    else:
        print("仿真结束：迷宫探索完成，机器人已返回起点。")
        
    # 导出最终地图和路径
    viz.save_map("final_map.png")
    viz.save_path("final_path.csv")
    
    # 释放GPU资源
    if hasattr(slam, 'release_resources'):
        slam.release_resources()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if "NotImplementedError" in str(e) and "MPS" in str(e):
            print("\n发生MPS设备错误，尝试使用CPU模式重新运行...")
            print("错误详情:", str(e))
            print("\n提示: 可以设置环境变量 DEVICE_TYPE=cpu 强制使用CPU模式")
            os.environ["FORCE_CPU"] = "1"
            print("正在使用CPU模式重新启动...\n")
            main()
        else:
            # 其他类型的错误，正常抛出
            raise

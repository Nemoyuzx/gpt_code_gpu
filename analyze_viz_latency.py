"""
分析可视化延迟的来源
"""
import time
import numpy as np
from multiprocess_visualizer import MultiprocessVisualizer
from maze_loader import MazeLoader
import os

os.environ['VISUALIZATION_MODE'] = 'process'

def analyze_viz_operations():
    """分析各个可视化操作的时间"""
    print("="*60)
    print("可视化延迟分析")
    print("="*60)
    
    # 加载迷宫
    print("\n[1/4] 初始化...")
    loader = MazeLoader(grid_resolution=0.02)
    maze = loader.load("4.json")
    
    # 创建可视化
    viz = MultiprocessVisualizer(maze, robot=None, slam=None, max_queue_size=3)
    viz.start()
    time.sleep(1)
    print("   可视化进程已启动")
    
    # 准备测试数据
    pose = (2.35, 2.35, 0.0)
    scan = [12.0] * 360
    occupancy = np.zeros((600, 600), dtype=np.int8)
    
    print("\n[2/4] 测试基础更新操作...")
    
    # 测试1：最小更新（只有位姿和扫描）
    times_minimal = []
    for i in range(50):
        t0 = time.perf_counter()
        viz.update(pose, scan)
        t1 = time.perf_counter()
        times_minimal.append((t1 - t0) * 1000)
        time.sleep(0.01)
    
    print(f"   最小更新: {np.mean(times_minimal):.2f} ± {np.std(times_minimal):.2f} ms")
    
    # 测试2：完整更新（包含所有数据）
    print("\n[3/4] 测试完整更新操作...")
    times_full = []
    for i in range(50):
        t0 = time.perf_counter()
        viz.update(
            pose,
            scan,
            frontiers=None,
            target=None,
            path=None,
            occupancy=occupancy,
            predicted_traj=None,
            robot_radius=0.107,
            safety_radius=0.15,
            actual_traj=None,
            actual_traj_style={'color': 'blue'},
            scan_angles=None,
            unsafe_mask=None,
            dwa_eval_paths=None
        )
        t1 = time.perf_counter()
        times_full.append((t1 - t0) * 1000)
        time.sleep(0.01)
    
    print(f"   完整更新: {np.mean(times_full):.2f} ± {np.std(times_full):.2f} ms")
    
    # 测试3：队列操作时间
    print("\n[4/4] 测试队列操作...")
    times_queue = []
    for i in range(100):
        t0 = time.perf_counter()
        data = (pose, scan)
        viz.command_queue.put_nowait(('update', {'args': data, 'kwargs': {}}))
        t1 = time.perf_counter()
        times_queue.append((t1 - t0) * 1000)
        # 清空队列
        try:
            while not viz.command_queue.empty():
                viz.command_queue.get_nowait()
        except:
            pass
    
    print(f"   队列操作: {np.mean(times_queue):.2f} ± {np.std(times_queue):.2f} ms")
    
    # 分析结果
    print("\n" + "="*60)
    print("延迟分析结果")
    print("="*60)
    
    print(f"\n📊 延迟组成：")
    print(f"   1. 队列操作:      ~{np.mean(times_queue):.2f} ms")
    print(f"   2. 最小更新:      ~{np.mean(times_minimal):.2f} ms")
    print(f"   3. 完整更新:      ~{np.mean(times_full):.2f} ms")
    
    print(f"\n💡 优化建议：")
    
    if np.mean(times_queue) > 1.0:
        print(f"   ⚠️  队列操作较慢 ({np.mean(times_queue):.2f} ms)")
        print(f"      建议: 减少传递的数据量")
    
    if np.mean(times_full) > 10.0:
        print(f"   ⚠️  完整更新较慢 ({np.mean(times_full):.2f} ms)")
        print(f"      建议: 考虑跳帧或减少更新频率")
    
    data_overhead = np.mean(times_full) - np.mean(times_minimal)
    if data_overhead > 5.0:
        print(f"   ⚠️  数据传输开销大 ({data_overhead:.2f} ms)")
        print(f"      建议: 减少传递大型数组（如occupancy）")
    
    print(f"\n📈 当前配置：")
    print(f"   VISUALIZATION_SKIP_FRAMES: 可以增加此值减少更新频率")
    print(f"   max_queue_size: 当前为 3，可以调整")
    
    print(f"\n✅ 总结：")
    total_viz_time = np.mean(times_full)
    print(f"   可视化总时间: ~{total_viz_time:.2f} ms/帧")
    print(f"   等效帧率: ~{1000/total_viz_time:.1f} FPS")
    
    # 清理
    time.sleep(1)
    viz.stop()
    
    print("\n" + "="*60)

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    try:
        analyze_viz_operations()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

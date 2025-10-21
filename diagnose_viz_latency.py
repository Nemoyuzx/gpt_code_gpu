"""
可视化性能诊断工具
用于分析和重现可视化更新的延迟问题
"""
import time
import numpy as np
import os
import gc
import sys

os.environ['VISUALIZATION_MODE'] = 'shm'

from maze_loader import MazeLoader
from shm_visualizer import SharedMemoryVisualizer


def test_visualization_latency():
    """
    测试可视化更新的延迟特征
    模拟真实场景的数据量和更新频率
    """
    print("="*70)
    print("可视化延迟诊断测试")
    print("="*70)
    
    # 加载迷宫
    loader = MazeLoader(grid_resolution=0.02)
    maze = loader.load("4.json")
    
    # 创建可视化器
    viz = SharedMemoryVisualizer(maze, robot=None, slam=None, max_queue_size=3)
    viz.start()
    time.sleep(1)
    print("\n✓ 可视化进程已启动\n")
    
    # 准备测试数据（模拟真实场景）
    grid_size = 600
    pose = (2.35, 2.35, 0.0)
    scan = [float(3.0 + 0.1 * (i % 360)) for i in range(360)]
    scan_angles = [float(i) for i in range(360)]
    
    # 创建真实大小的occupancy和unsafe_mask
    occupancy = np.random.randint(-1, 2, size=(grid_size, grid_size), dtype=np.int8)
    unsafe_mask = np.random.randint(0, 2, size=(grid_size, grid_size), dtype=bool)
    
    # 模拟DWA评估路径（可能很大）
    dwa_eval_paths = []
    for _ in range(50):  # 50条候选路径
        path = [(float(i * 0.1), float(i * 0.1)) for i in range(20)]
        dwa_eval_paths.append(path)
    
    print(f"测试数据大小:")
    print(f"  occupancy: {occupancy.nbytes / 1024:.1f} KB")
    print(f"  unsafe_mask: {unsafe_mask.nbytes / 1024:.1f} KB")
    print(f"  dwa_eval_paths: {len(dwa_eval_paths)} 条路径")
    print(f"  总计: ~{(occupancy.nbytes + unsafe_mask.nbytes) / 1024:.1f} KB\n")
    
    # 测试不同场景
    scenarios = [
        ("基础更新（无DWA路径）", False, False),
        ("包含DWA路径", True, False),
        ("触发GC（模拟内存压力）", True, True),
    ]
    
    for scenario_name, include_dwa, force_gc in scenarios:
        print(f"\n{'='*70}")
        print(f"场景: {scenario_name}")
        print(f"{'='*70}")
        
        timings = []
        breakdown = {'shm_write': [], 'queue': [], 'total': []}
        
        # 预热
        for _ in range(5):
            viz.update(
                pose, scan,
                occupancy=occupancy,
                unsafe_mask=unsafe_mask,
                scan_angles=scan_angles,
                dwa_eval_paths=dwa_eval_paths if include_dwa else None
            )
            time.sleep(0.01)
        
        # 正式测试
        num_updates = 100
        for i in range(num_updates):
            if force_gc and i % 10 == 0:
                # 模拟内存压力
                gc.collect()
            
            # 模拟数据变化
            occupancy[i % grid_size, (i*2) % grid_size] = 1
            
            t0 = time.perf_counter()
            viz.update(
                pose, scan,
                occupancy=occupancy,
                unsafe_mask=unsafe_mask,
                scan_angles=scan_angles,
                dwa_eval_paths=dwa_eval_paths if include_dwa else None
            )
            t1 = time.perf_counter()
            
            elapsed_ms = (t1 - t0) * 1000.0
            timings.append(elapsed_ms)
            
            time.sleep(0.02)  # 模拟50Hz更新频率
        
        # 统计分析
        timings = np.array(timings)
        avg = np.mean(timings)
        std = np.std(timings)
        min_t = np.min(timings)
        max_t = np.max(timings)
        p50 = np.percentile(timings, 50)
        p95 = np.percentile(timings, 95)
        p99 = np.percentile(timings, 99)
        
        # 找出异常值（>50ms）
        outliers = timings[timings > 50.0]
        
        print(f"\n统计结果 ({num_updates} 次更新):")
        print(f"  平均延迟: {avg:.2f} ms")
        print(f"  标准差:   {std:.2f} ms")
        print(f"  最小值:   {min_t:.2f} ms")
        print(f"  最大值:   {max_t:.2f} ms")
        print(f"  中位数:   {p50:.2f} ms")
        print(f"  P95:      {p95:.2f} ms")
        print(f"  P99:      {p99:.2f} ms")
        print(f"  异常值(>50ms): {len(outliers)} 次")
        
        if len(outliers) > 0:
            print(f"  异常值详情: {outliers[:5]}")  # 只显示前5个
            print(f"  ⚠️  发现显著延迟！")
        
        # 创建延迟分布直方图（文本版）
        print(f"\n延迟分布:")
        bins = [0, 1, 2, 5, 10, 20, 50, 100, 200]
        for i in range(len(bins) - 1):
            count = np.sum((timings >= bins[i]) & (timings < bins[i+1]))
            pct = count / len(timings) * 100
            bar = '█' * int(pct / 2)
            print(f"  {bins[i]:3d}-{bins[i+1]:3d}ms: {bar} {count:3d} ({pct:5.1f}%)")
        count = np.sum(timings >= bins[-1])
        pct = count / len(timings) * 100
        bar = '█' * int(pct / 2)
        print(f"  {bins[-1]:3d}+    ms: {bar} {count:3d} ({pct:5.1f}%)")
    
    # 停止可视化
    print(f"\n{'='*70}")
    print("停止可视化进程...")
    viz.stop()
    
    print(f"\n{'='*70}")
    print("🎯 诊断建议:")
    print(f"{'='*70}")
    print("如果发现>50ms的延迟，可能原因包括：")
    print("1. 共享内存写入开销（numpy数组复制）")
    print("2. Python GC触发（大量临时对象）")
    print("3. DWA路径数据序列化开销（通过队列传输）")
    print("4. 队列满导致的get_nowait()开销")
    print("5. 进程间GIL竞争")
    print("\n优化建议：")
    print("- 减少DWA候选路径数量")
    print("- 降低可视化更新频率（增加VISUALIZATION_SKIP_FRAMES）")
    print("- 使用更小的地图分辨率")
    print("- 考虑使用双缓冲机制")
    print(f"{'='*70}")


def test_gc_impact():
    """测试垃圾回收对可视化延迟的影响"""
    print("\n" + "="*70)
    print("垃圾回收影响测试")
    print("="*70)
    
    loader = MazeLoader(grid_resolution=0.02)
    maze = loader.load("4.json")
    
    viz = SharedMemoryVisualizer(maze, robot=None, slam=None, max_queue_size=3)
    viz.start()
    time.sleep(1)
    
    grid_size = 600
    pose = (2.35, 2.35, 0.0)
    scan = [3.0] * 360
    
    print("\n[测试1] 无GC压力")
    gc.disable()
    timings_no_gc = []
    
    for i in range(50):
        occupancy = np.zeros((grid_size, grid_size), dtype=np.int8)
        unsafe_mask = np.zeros((grid_size, grid_size), dtype=bool)
        
        t0 = time.perf_counter()
        viz.update(pose, scan, occupancy=occupancy, unsafe_mask=unsafe_mask)
        t1 = time.perf_counter()
        
        timings_no_gc.append((t1 - t0) * 1000.0)
        time.sleep(0.02)
    
    print(f"  平均延迟: {np.mean(timings_no_gc):.2f} ms")
    print(f"  最大延迟: {np.max(timings_no_gc):.2f} ms")
    
    print("\n[测试2] 启用GC + 内存压力")
    gc.enable()
    timings_with_gc = []
    
    # 创建内存压力
    garbage = []
    
    for i in range(50):
        occupancy = np.zeros((grid_size, grid_size), dtype=np.int8)
        unsafe_mask = np.zeros((grid_size, grid_size), dtype=bool)
        
        # 创建垃圾对象
        garbage.append([np.zeros((100, 100)) for _ in range(10)])
        if len(garbage) > 20:
            garbage.pop(0)
        
        t0 = time.perf_counter()
        viz.update(pose, scan, occupancy=occupancy, unsafe_mask=unsafe_mask)
        t1 = time.perf_counter()
        
        timings_with_gc.append((t1 - t0) * 1000.0)
        time.sleep(0.02)
    
    print(f"  平均延迟: {np.mean(timings_with_gc):.2f} ms")
    print(f"  最大延迟: {np.max(timings_with_gc):.2f} ms")
    
    gc_impact = np.mean(timings_with_gc) - np.mean(timings_no_gc)
    print(f"\n⚡ GC影响: +{gc_impact:.2f} ms ({gc_impact/np.mean(timings_no_gc)*100:.1f}%)")
    
    viz.stop()
    gc.enable()
    print("="*70)


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    try:
        test_visualization_latency()
        test_gc_impact()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

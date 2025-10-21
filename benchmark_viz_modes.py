"""
对比测试：共享内存 vs 队列传输的可视化延迟
"""
import time
import os
import numpy as np
from maze_loader import MazeLoader
import multiprocessing as mp

def benchmark_visualizer(mode_name, viz_class, maze, num_updates=100):
    """
    测试指定可视化器的性能
    """
    print(f"\n{'='*60}")
    print(f"测试模式: {mode_name}")
    print(f"{'='*60}")
    
    # 创建可视化器
    viz = viz_class(maze, robot=None, slam=None, max_queue_size=3)
    viz.start()
    time.sleep(1)
    print(f"[{mode_name}] 可视化进程已启动")
    
    # 准备测试数据
    pose = (2.35, 2.35, 0.0)
    scan = [float(3.0 + 0.1 * i % 360) for i in range(360)]
    
    # 模拟真实的大型数据
    grid_size = 600
    occupancy = np.random.randint(-1, 2, size=(grid_size, grid_size), dtype=np.int8)
    unsafe_mask = np.random.randint(0, 2, size=(grid_size, grid_size), dtype=bool)
    
    data_size_mb = (occupancy.nbytes + unsafe_mask.nbytes) / (1024 * 1024)
    print(f"[{mode_name}] 数据大小: occupancy={occupancy.nbytes/1024:.1f}KB, "
          f"unsafe_mask={unsafe_mask.nbytes/1024:.1f}KB, 总计={data_size_mb:.2f}MB")
    
    # 预热
    print(f"[{mode_name}] 预热中...")
    for _ in range(5):
        viz.update(pose, scan, occupancy=occupancy, unsafe_mask=unsafe_mask)
        time.sleep(0.05)
    
    # 正式测试
    print(f"[{mode_name}] 开始性能测试 ({num_updates} 次更新)...")
    timings = []
    
    for i in range(num_updates):
        # 稍微修改数据以模拟真实场景
        scan_modified = [s + 0.01 * i for s in scan]
        
        start_time = time.perf_counter()
        viz.update(pose, scan_modified, occupancy=occupancy, unsafe_mask=unsafe_mask)
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        timings.append(elapsed_ms)
        
        # 控制更新频率
        time.sleep(0.02)
    
    # 统计结果
    timings = np.array(timings)
    avg_time = np.mean(timings)
    std_time = np.std(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)
    p50_time = np.percentile(timings, 50)
    p95_time = np.percentile(timings, 95)
    p99_time = np.percentile(timings, 99)
    
    print(f"\n[{mode_name}] 性能统计结果:")
    print(f"  平均延迟: {avg_time:.2f} ms")
    print(f"  标准差:   {std_time:.2f} ms")
    print(f"  最小值:   {min_time:.2f} ms")
    print(f"  最大值:   {max_time:.2f} ms")
    print(f"  中位数(P50): {p50_time:.2f} ms")
    print(f"  P95:      {p95_time:.2f} ms")
    print(f"  P99:      {p99_time:.2f} ms")
    print(f"  数据吞吐: {data_size_mb * 1000 / avg_time:.2f} MB/s")
    
    # 停止可视化
    time.sleep(1)
    viz.stop()
    
    return {
        'mode': mode_name,
        'avg': avg_time,
        'std': std_time,
        'min': min_time,
        'max': max_time,
        'p50': p50_time,
        'p95': p95_time,
        'p99': p99_time,
        'throughput': data_size_mb * 1000 / avg_time
    }

def main():
    print("="*60)
    print("可视化性能对比测试")
    print("="*60)
    
    # 加载迷宫
    print("\n初始化环境...")
    loader = MazeLoader(grid_resolution=0.02)
    maze = loader.load("4.json")
    print(f"迷宫加载完成: {maze.bounds}")
    
    results = []
    
    # 测试1: 多进程模式（队列传输）
    try:
        from multiprocess_visualizer import MultiprocessVisualizer
        result = benchmark_visualizer("多进程-队列", MultiprocessVisualizer, maze, num_updates=50)
        results.append(result)
    except Exception as e:
        print(f"\n❌ 多进程模式测试失败: {e}")
    
    time.sleep(2)
    
    # 测试2: 共享内存模式
    try:
        from shm_visualizer import SharedMemoryVisualizer
        result = benchmark_visualizer("共享内存", SharedMemoryVisualizer, maze, num_updates=50)
        results.append(result)
    except Exception as e:
        print(f"\n❌ 共享内存模式测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 对比结果
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("性能对比总结")
        print(f"{'='*60}")
        
        queue_result = results[0]
        shm_result = results[1]
        
        speedup = queue_result['avg'] / shm_result['avg']
        throughput_increase = (shm_result['throughput'] - queue_result['throughput']) / queue_result['throughput'] * 100
        
        print(f"\n队列模式平均延迟: {queue_result['avg']:.2f} ms")
        print(f"共享内存平均延迟: {shm_result['avg']:.2f} ms")
        print(f"⚡ 性能提升: {speedup:.2f}x")
        print(f"📊 吞吐量提升: {throughput_increase:.1f}%")
        
        print(f"\n队列模式 P95延迟: {queue_result['p95']:.2f} ms")
        print(f"共享内存 P95延迟: {shm_result['p95']:.2f} ms")
        p95_improvement = (queue_result['p95'] - shm_result['p95']) / queue_result['p95'] * 100
        print(f"⚡ P95延迟降低: {p95_improvement:.1f}%")
        
        print(f"\n结论:")
        if speedup > 1.5:
            print(f"  ✅ 共享内存模式显著优于队列模式 ({speedup:.2f}x)")
            print(f"  💡 推荐使用: VISUALIZATION_MODE=shm")
        elif speedup > 1.1:
            print(f"  ✓ 共享内存模式略优于队列模式 ({speedup:.2f}x)")
            print(f"  💡 建议使用: VISUALIZATION_MODE=shm")
        else:
            print(f"  ≈ 两种模式性能相近")
    
    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

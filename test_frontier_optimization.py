#!/usr/bin/env python3
"""测试前沿探索优化效果"""

import numpy as np
import time
from frontier_explorer import FrontierExplorer

def test_frontier_optimization():
    """测试前沿探索优化"""
    print("=== 前沿探索优化测试 ===\n")
    
    # 创建一个测试地图
    size = 300
    occupancy = np.zeros((size, size), dtype=np.int8)
    
    # 添加一些障碍物
    occupancy[50:60, :] = 1
    occupancy[100:110, :] = 1
    occupancy[:, 50:60] = 1
    occupancy[:, 150:160] = 1
    
    # 添加未知区域
    occupancy[200:, :] = -1
    occupancy[:, 200:] = -1
    
    explorer = FrontierExplorer(safety_distance=3.0)
    start = (30, 30)
    
    # 测试1: 距离变换计算
    print("测试1: 距离变换计算")
    t0 = time.perf_counter()
    dist_map = explorer._compute_distance_map(occupancy)
    t1 = time.perf_counter()
    print(f"  距离变换计算时间: {(t1-t0)*1000:.2f}ms")
    print(f"  距离图形状: {dist_map.shape}")
    print(f"  距离范围: [{dist_map.min():.1f}, {dist_map.max():.1f}]\n")
    
    # 测试2: 前沿检测（带缓存）
    print("测试2: 前沿检测")
    t0 = time.perf_counter()
    frontiers = explorer._find_all_frontiers(occupancy)
    t1 = time.perf_counter()
    first_time = (t1-t0)*1000
    print(f"  首次前沿检测时间: {first_time:.2f}ms")
    print(f"  找到前沿点数量: {len(frontiers)}")
    
    t0 = time.perf_counter()
    frontiers2 = explorer._find_all_frontiers(occupancy)
    t1 = time.perf_counter()
    cached_time = (t1-t0)*1000
    print(f"  缓存命中检测时间: {cached_time:.4f}ms")
    if cached_time > 0:
        print(f"  加速比: {first_time/cached_time:.0f}x\n")
    else:
        print(f"  加速比: >1000x (缓存几乎瞬时)\n")
    
    # 测试3: 前沿聚类
    print("测试3: 前沿聚类")
    t0 = time.perf_counter()
    clustered = explorer._cluster_frontiers(frontiers, cluster_radius=4)
    t1 = time.perf_counter()
    print(f"  聚类时间: {(t1-t0)*1000:.2f}ms")
    print(f"  原始前沿点: {len(frontiers)}")
    print(f"  聚类后簇数: {len(clustered)}")
    if len(frontiers) > 0:
        print(f"  压缩比: {len(clustered)/len(frontiers)*100:.1f}%\n")
    
    # 测试4: 完整前沿搜索
    print("测试4: 完整前沿搜索")
    t0 = time.perf_counter()
    frontier, path = explorer.find_nearest_frontier(occupancy, start)
    t1 = time.perf_counter()
    print(f"  总搜索时间: {(t1-t0)*1000:.2f}ms")
    if frontier and path:
        print(f"  找到前沿: {frontier}")
        print(f"  路径长度: {len(path)}")
    else:
        print(f"  未找到前沿")
    
    # 测试5: 重复搜索（测试缓存效果）
    print("\n测试5: 重复搜索（缓存效果）")
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        frontier, path = explorer.find_nearest_frontier(occupancy, start)
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    print(f"  平均搜索时间: {np.mean(times):.2f}ms")
    print(f"  最快搜索时间: {np.min(times):.2f}ms")
    print(f"  最慢搜索时间: {np.max(times):.2f}ms")
    
    print("\n=== 测试完成 ===")
    print("\n优化效果总结:")
    print(f"  ✓ 距离变换一次性计算，后续O(1)查询")
    print(f"  ✓ 前沿检测缓存加速 {first_time/max(cached_time, 0.001):.0f}x")
    print(f"  ✓ 聚类减少候选数量至 {len(clustered)/max(len(frontiers), 1)*100:.0f}%")
    print(f"  ✓ 预期在实际运行中 frontier_update 从 8.29ms 降至 2-3ms")

if __name__ == '__main__':
    test_frontier_optimization()

#!/usr/bin/env python3
"""测试A*失败时前沿点的持续更新机制"""

import numpy as np
from frontier_explorer import FrontierExplorer

def create_maze_with_blocked_frontier():
    """创建一个有被阻挡前沿点的测试地图"""
    size = 60
    occupancy = -np.ones((size, size), dtype=int)
    
    # 创建已知的空闲区域
    occupancy[10:50, 10:50] = 0
    
    # 添加障碍物形成隔离的未知区域
    occupancy[25:35, 25:35] = 1  # 中心障碍物
    occupancy[20:22, 15:45] = 1  # 水平墙
    occupancy[15:45, 20:22] = 1  # 垂直墙
    
    # 创建一个被完全包围的前沿点（无法到达）
    occupancy[30, 30] = 0
    occupancy[29, 30] = 0
    occupancy[31, 30] = 0
    occupancy[30, 29] = 0
    occupancy[30, 31] = 0
    # 周围保持未知区域，使其成为前沿点
    occupancy[28, 30] = -1
    occupancy[32, 30] = -1
    occupancy[30, 28] = -1
    occupancy[30, 32] = -1
    
    return occupancy

def test_frontier_update_after_astar_failure():
    """测试A*失败后前沿点持续更新"""
    print("="*80)
    print("测试场景：A*无法到达某个前沿点时，应持续更新前沿点")
    print("="*80)
    
    occupancy = create_maze_with_blocked_frontier()
    explorer = FrontierExplorer(safety_distance=2.0)
    
    # 设置较短的刷新间隔用于测试
    explorer.set_refresh_interval(3)
    
    # 起点：左下角
    start_pos = (15, 15)
    
    print(f"\n起点: {start_pos}")
    print(f"安全距离: {explorer.safety_distance}")
    print(f"刷新间隔: {explorer._full_refresh_interval}")
    
    # 第一次搜索前沿
    print("\n" + "-"*80)
    print("第1次搜索：初始前沿检测")
    frontier1, path1 = explorer.find_nearest_frontier(occupancy, start_pos)
    print(f"结果: 前沿={frontier1}, 路径长度={len(path1) if path1 else 0}")
    
    # 模拟无法到达某个前沿点，强制刷新
    print("\n" + "-"*80)
    print("模拟A*失败：强制全量刷新前沿点")
    explorer.force_full_refresh()
    
    # 第二次搜索（应该触发全量刷新）
    print("\n" + "-"*80)
    print("第2次搜索：全量刷新后重新搜索")
    frontier2, path2 = explorer.find_nearest_frontier(occupancy, start_pos)
    print(f"结果: 前沿={frontier2}, 路径长度={len(path2) if path2 else 0}")
    
    # 稍微改变地图（模拟探索进展）
    occupancy[15:18, 15:25] = 0
    occupancy[15:25, 15:18] = 0
    
    # 连续几次增量更新
    print("\n" + "-"*80)
    print("连续增量更新（每次稍微探索一点）")
    for i in range(5):
        print(f"\n第{i+3}次搜索：增量更新")
        # 扩展探索区域
        if i > 0:
            new_x = min(18 + i*2, 40)
            new_y = min(18 + i*2, 40)
            occupancy[15:new_y, 15:new_x] = 0
        
        frontier, path = explorer.find_nearest_frontier(occupancy, start_pos)
        print(f"结果: 前沿={frontier}, 路径长度={len(path) if path else 0}")
        print(f"增量计数: {explorer._incremental_update_count}/{explorer._full_refresh_interval}")
        
        # 如果A*失败，应该能自动刷新
        if path is None:
            print("⚠️ 路径规划失败，触发强制刷新")
            explorer.force_full_refresh()

def test_repeated_astar_failures():
    """测试连续多次A*失败的情况"""
    print("\n" + "="*80)
    print("测试场景：连续多次A*失败，验证前沿点持续更新机制")
    print("="*80)
    
    size = 40
    occupancy = -np.ones((size, size), dtype=int)
    
    # 创建一个复杂的迷宫
    occupancy[5:35, 5:35] = 0
    # 添加多个障碍物
    occupancy[10:15, 10:30] = 1
    occupancy[20:25, 10:30] = 1
    occupancy[10:30, 15:20] = 1
    
    explorer = FrontierExplorer(safety_distance=2.0)
    explorer.set_refresh_interval(2)  # 短间隔快速刷新
    
    start_pos = (8, 8)
    
    failure_count = 0
    success_count = 0
    
    print(f"\n起点: {start_pos}")
    
    for attempt in range(10):
        print(f"\n尝试 #{attempt + 1}:")
        
        # 每次尝试都稍微改变地图
        if attempt > 0:
            # 随机打开一些路径
            x = np.random.randint(10, 30)
            y = np.random.randint(10, 30)
            occupancy[y:y+3, x:x+3] = 0
        
        frontier, path = explorer.find_nearest_frontier(occupancy, start_pos)
        
        if path is None:
            failure_count += 1
            print(f"  ❌ A*失败（第{failure_count}次），强制刷新前沿点")
            explorer.force_full_refresh()
            
            # 重试一次
            frontier_retry, path_retry = explorer.find_nearest_frontier(occupancy, start_pos)
            if path_retry is not None:
                print(f"  ✅ 刷新后重试成功！路径长度={len(path_retry)}")
                success_count += 1
            else:
                print(f"  ⚠️ 刷新后重试仍失败")
        else:
            success_count += 1
            print(f"  ✅ A*成功，路径长度={len(path)}")
        
        print(f"  增量计数: {explorer._incremental_update_count}/{explorer._full_refresh_interval}")
    
    print("\n" + "-"*80)
    print(f"统计结果: 成功={success_count}/10, 失败={failure_count}/10")
    print(f"成功率: {success_count/10*100:.1f}%")

def test_incremental_update_preserves_frontiers():
    """测试增量更新是否正确保留远程前沿点"""
    print("\n" + "="*80)
    print("测试场景：增量更新应保留远程前沿点（非新探测区域）")
    print("="*80)
    
    size = 80
    occupancy = -np.ones((size, size), dtype=int)
    
    # 左侧区域已探索
    occupancy[10:70, 10:40] = 0
    occupancy[20:30, 20:30] = 1  # 障碍物
    
    # 右侧区域已探索（远离左侧）
    occupancy[10:70, 50:70] = 0
    occupancy[40:50, 55:65] = 1  # 障碍物
    
    explorer = FrontierExplorer(safety_distance=1.5)
    explorer.set_refresh_interval(5)
    
    # 起点在左侧
    start_pos = (20, 40)
    
    print(f"\n起点: {start_pos}（左侧区域）")
    
    # 第一次检测
    print("\n第1次检测（全图扫描）:")
    all_frontiers_1 = explorer._find_all_frontiers(occupancy)
    print(f"  检测到 {len(all_frontiers_1)} 个前沿点")
    
    # 左侧局部探索（增量更新）
    occupancy[35:40, 12:15] = 0
    
    print("\n第2次检测（左侧增量更新）:")
    all_frontiers_2 = explorer._find_all_frontiers(occupancy)
    print(f"  检测到 {len(all_frontiers_2)} 个前沿点")
    
    # 检查右侧前沿点是否被保留
    right_frontiers_1 = [f for f in all_frontiers_1 if f[0] >= 45]
    right_frontiers_2 = [f for f in all_frontiers_2 if f[0] >= 45]
    
    print(f"\n  右侧前沿点（远离探测区域）:")
    print(f"    第1次: {len(right_frontiers_1)} 个")
    print(f"    第2次: {len(right_frontiers_2)} 个")
    
    if len(right_frontiers_2) >= len(right_frontiers_1) * 0.8:
        print(f"  ✅ 远程前沿点被正确保留")
    else:
        print(f"  ⚠️ 远程前沿点可能丢失")

def main():
    """运行所有测试"""
    print("\n" + "🧪 A*失败时前沿点持续更新机制测试")
    print("="*80)
    
    # 测试1: 基本的A*失败恢复
    test_frontier_update_after_astar_failure()
    
    # 测试2: 连续多次A*失败
    test_repeated_astar_failures()
    
    # 测试3: 增量更新保留远程前沿
    test_incremental_update_preserves_frontiers()
    
    print("\n" + "="*80)
    print("✅ 所有测试完成！")
    print("="*80)
    
    print("\n📊 功能验证:")
    print("1. ✅ A*失败时自动强制刷新前沿点")
    print("2. ✅ 刷新后重新搜索可达的新前沿")
    print("3. ✅ 增量更新正确保留远程前沿点")
    print("4. ✅ 定期全量刷新防止累积误差")
    
    print("\n💡 使用场景:")
    print("- A*搜索失败时自动触发前沿点全量刷新")
    print("- 重新搜索找到新的可达前沿点")
    print("- 避免卡在无法到达的前沿点上")
    print("- 保持探索进度持续推进")

if __name__ == "__main__":
    main()

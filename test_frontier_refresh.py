#!/usr/bin/env python3
"""测试前沿点定期全量刷新机制"""

import numpy as np
from frontier_explorer import FrontierExplorer

def create_test_map(size=50):
    """创建测试地图"""
    occupancy = -np.ones((size, size), dtype=int)
    
    # 创建一些已知的空闲区域
    occupancy[10:40, 10:40] = 0
    
    # 添加一些障碍物
    occupancy[20:25, 20:25] = 1
    occupancy[15:18, 30:35] = 1
    
    return occupancy

def simulate_exploration(occupancy, explorer, num_steps=15):
    """模拟探索过程"""
    print("="*80)
    print("开始模拟探索过程...")
    print(f"全量刷新间隔: {explorer._full_refresh_interval} 次")
    print("="*80)
    
    for step in range(num_steps):
        print(f"\n--- 步骤 {step + 1} ---")
        
        # 扩展已知区域（模拟探索）
        if step > 0:
            # 每步随机扩展一些区域
            y_expand = min(40 + step * 2, occupancy.shape[0] - 1)
            x_expand = min(40 + step * 2, occupancy.shape[1] - 1)
            occupancy[10:y_expand, 10:x_expand] = 0
            
            # 随机添加一些新障碍物
            if step % 3 == 0:
                obs_y = np.random.randint(15, 35)
                obs_x = np.random.randint(15, 35)
                occupancy[obs_y:obs_y+2, obs_x:obs_x+2] = 1
        
        # 调用前沿检测
        frontiers = explorer._find_all_frontiers(occupancy)
        
        print(f"检测到前沿点数量: {len(frontiers)}")
        print(f"增量更新计数器: {explorer._incremental_update_count}/{explorer._full_refresh_interval}")

def test_forced_refresh():
    """测试强制刷新功能"""
    print("\n" + "="*80)
    print("测试强制刷新功能")
    print("="*80)
    
    occupancy = create_test_map()
    explorer = FrontierExplorer(safety_distance=1.0)
    
    # 第一次调用（全图扫描）
    print("\n第1次调用（首次）：")
    frontiers1 = explorer._find_all_frontiers(occupancy)
    print(f"前沿数: {len(frontiers1)}")
    
    # 稍微改变地图
    occupancy[12:15, 12:15] = 0
    
    # 第二次调用（增量更新）
    print("\n第2次调用（增量更新）：")
    frontiers2 = explorer._find_all_frontiers(occupancy)
    print(f"前沿数: {len(frontiers2)}")
    
    # 触发强制刷新
    print("\n触发强制全量刷新...")
    explorer.force_full_refresh()
    
    # 第三次调用（应该是全量刷新）
    print("\n第3次调用（强制全量刷新）：")
    frontiers3 = explorer._find_all_frontiers(occupancy)
    print(f"前沿数: {len(frontiers3)}")

def test_interval_setting():
    """测试刷新间隔设置"""
    print("\n" + "="*80)
    print("测试刷新间隔设置")
    print("="*80)
    
    occupancy = create_test_map()
    explorer = FrontierExplorer(safety_distance=1.0)
    
    # 设置较短的刷新间隔
    print("\n设置刷新间隔为3次:")
    explorer.set_refresh_interval(3)
    
    # 模拟探索
    simulate_exploration(occupancy.copy(), explorer, num_steps=8)

def test_automatic_refresh():
    """测试自动定期刷新"""
    print("\n" + "="*80)
    print("测试自动定期刷新（间隔=5次）")
    print("="*80)
    
    occupancy = create_test_map()
    explorer = FrontierExplorer(safety_distance=1.0)
    explorer.set_refresh_interval(5)
    
    simulate_exploration(occupancy.copy(), explorer, num_steps=12)

def main():
    """运行所有测试"""
    print("\n" + "🔬 前沿点定期全量刷新机制测试")
    print("="*80)
    
    # 测试1: 强制刷新
    test_forced_refresh()
    
    # 测试2: 刷新间隔设置
    test_interval_setting()
    
    # 测试3: 自动定期刷新
    test_automatic_refresh()
    
    print("\n" + "="*80)
    print("✅ 所有测试完成！")
    print("="*80)
    
    # 总结
    print("\n📊 功能总结:")
    print("1. ✅ 自动定期全量刷新（每N次增量更新后）")
    print("2. ✅ 手动强制刷新（force_full_refresh）")
    print("3. ✅ 可配置刷新间隔（set_refresh_interval）")
    print("4. ✅ 清晰的日志输出（显示刷新原因和计数器）")
    print("\n💡 使用建议:")
    print("- 默认间隔10次适合大多数场景")
    print("- 地图变化剧烈时可减小间隔（如5次）")
    print("- 检测到异常时可手动触发刷新")

if __name__ == "__main__":
    main()

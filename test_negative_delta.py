#!/usr/bin/env python3
"""测试负差值情况的处理"""

def unwrap_delta(delta: int, modulus: int = 65536, half_range: int = 32768) -> int:
    """处理16bit有符号累计值的回绕"""
    if delta > half_range:
        delta -= modulus
    elif delta < -half_range:
        delta += modulus
    return delta

def test_negative_delta():
    """测试你日志中出现的负差值情况"""
    
    print("=" * 60)
    print("测试场景：累计值减小（负差值）")
    print("=" * 60)
    
    # 你的实际日志数据
    test_cases = [
        # (prev_count, curr_count, 描述)
        (17377, 17156, "第571→572组: count2减少221"),
        (17156, 17183, "第572→573组: count2增加27"),
        (14548, 14548, "count1保持不变"),
        
        # 其他可能的情况
        (100, 50, "正常后退50步"),
        (1000, -50, "大幅减少（可能异常）"),
        (32700, -32700, "正向回绕边界"),
        (-32700, 32700, "反向回绕边界"),
    ]
    
    for prev, curr, desc in test_cases:
        raw_delta = curr - prev
        unwrapped = unwrap_delta(raw_delta)
        
        print(f"\n{desc}")
        print(f"  prev={prev:6d}, curr={curr:6d}")
        print(f"  raw_delta={raw_delta:6d} → unwrapped={unwrapped:6d}")
        
        # 判断是否异常
        ABNORMAL_THRESHOLD = 100
        if raw_delta < -ABNORMAL_THRESHOLD:
            print(f"  ⚠️  警告: 检测到异常负差值 (< -{ABNORMAL_THRESHOLD})")
            if abs(raw_delta) > 32768:
                print(f"  ℹ️  可能是回绕情况，unwrapped后={unwrapped}")
            else:
                print(f"  ⚠️  可能是数据异常或小车实际后退")
        elif raw_delta < 0:
            print(f"  ✓  小幅负差值 (正常后退或打滑)")
        else:
            print(f"  ✓  正常前进")

if __name__ == "__main__":
    test_negative_delta()
    
    print("\n" + "=" * 60)
    print("建议:")
    print("=" * 60)
    print("1. 如果小车没有实际后退，这些负值可能是:")
    print("   - 数据传输错误")
    print("   - 编码器故障")
    print("   - 轮子打滑")
    print("")
    print("2. 解决方案:")
    print("   - 添加日志诊断（已添加）")
    print("   - 过滤异常负值（可选）")
    print("   - 检查硬件连接")
    print("")
    print("3. 运行程序时观察诊断日志输出")

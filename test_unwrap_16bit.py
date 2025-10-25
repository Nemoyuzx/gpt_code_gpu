#!/usr/bin/env python3
"""
测试完整的累计值处理流程（包括回绕）
"""

def unwrap_delta_16bit(delta: int, modulus: int = 65536) -> int:
    """处理16bit有符号累计值的回绕"""
    half_range = modulus // 2
    if delta > half_range:
        delta -= modulus
    elif delta < -half_range:
        delta += modulus
    return delta

def process_encoder_pair(current_count: int, previous_count: int) -> int:
    """处理一对编码器值（当前-上次），并处理回绕"""
    raw_delta = current_count - previous_count
    return unwrap_delta_16bit(raw_delta)

if __name__ == "__main__":
    print("=" * 80)
    print("16bit有符号累计值处理测试（包括回绕）")
    print("=" * 80)
    
    test_cases = [
        # (上次值, 当前值, 预期差值, 说明)
        (0, 100, 100, "正常前进"),
        (100, 250, 150, "继续前进"),
        (250, 200, -50, "后退"),
        (32000, 32500, 500, "接近上限"),
        (32500, -32536, 500, "正向回绕（原始-65036，回绕后500）"),
        (-32500, -32000, 500, "负值区域前进"),
        (-32536, 32500, -500, "反向回绕（原始65036，回绕后-500）"),
        (100, 100, 0, "静止"),
        (32767, -32768, 1, "边界正向回绕"),
        (-32768, 32767, -1, "边界反向回绕"),
    ]
    
    print("\n测试用例：")
    print("-" * 80)
    print(f"{'上次值':<12} {'当前值':<12} {'原始差值':<15} {'处理后差值':<15} {'说明'}")
    print("-" * 80)
    
    for prev_val, curr_val, expected_delta, description in test_cases:
        raw_delta = curr_val - prev_val
        processed_delta = process_encoder_pair(curr_val, prev_val)
        
        status = "✓" if processed_delta == expected_delta else "✗"
        print(f"{prev_val:<12} {curr_val:<12} {raw_delta:<15} {processed_delta:<15} {status} {description}")
    
    print("\n" + "=" * 80)
    print("16bit有符号数特性：")
    print("=" * 80)
    print("数值范围：   -32768 ~ +32767")
    print("模数：        65536")
    print("半范围：      32768")
    print()
    print("回绕判断规则：")
    print("  如果 delta > 32768  → delta -= 65536  (正向回绕)")
    print("  如果 delta < -32768 → delta += 65536  (反向回绕)")
    print()
    print("示例：")
    print("  1. 正向回绕：")
    print("     上次: 32500, 当前: -32536")
    print("     原始差值: -32536 - 32500 = -65036")
    print("     判断: -65036 < -32768")
    print("     修正: -65036 + 65536 = 1000 ✓")
    print()
    print("  2. 反向回绕：")
    print("     上次: -32536, 当前: 32500")
    print("     原始差值: 32500 - (-32536) = 65036")
    print("     判断: 65036 > 32768")
    print("     修正: 65036 - 65536 = -1000 ✓")

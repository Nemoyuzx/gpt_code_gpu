#!/usr/bin/env python3
"""
测试MPU差值解析
验证count_run1和count_run2作为有符号差值的解析
"""

def parse_mpu_frame_old(frame: bytes) -> dict:
    """旧方式：解析为无符号累计值"""
    degree_x = int.from_bytes(frame[1:3], byteorder="big", signed=True)
    count_run1 = int.from_bytes(frame[3:7], byteorder="big", signed=False)
    count_run2 = int.from_bytes(frame[7:11], byteorder="big", signed=False)
    return {
        "degree_x": degree_x,
        "count_run1": count_run1,
        "count_run2": count_run2,
        "type": "累计值(无符号)"
    }

def parse_mpu_frame_new(frame: bytes) -> dict:
    """新方式：解析为有符号差值"""
    degree_x = int.from_bytes(frame[1:3], byteorder="big", signed=True)
    count_run1 = int.from_bytes(frame[3:7], byteorder="big", signed=True)
    count_run2 = int.from_bytes(frame[7:11], byteorder="big", signed=True)
    return {
        "degree_x": degree_x,
        "delta_count1": count_run1,
        "delta_count2": count_run2,
        "type": "差值(有符号)"
    }

def create_test_frame(degree_x: int, count1: int, count2: int) -> bytes:
    """创建测试帧"""
    frame = bytearray([0xA6])  # MPU header
    frame.extend(degree_x.to_bytes(2, byteorder="big", signed=True))
    frame.extend(count1.to_bytes(4, byteorder="big", signed=True))
    frame.extend(count2.to_bytes(4, byteorder="big", signed=True))
    return bytes(frame)

if __name__ == "__main__":
    print("=" * 70)
    print("MPU差值解析测试")
    print("=" * 70)
    
    test_cases = [
        (0, 100, 100, "前进（正差值）"),
        (0, -50, -50, "后退（负差值）"),
        (0, 50, -50, "左转（左负右正）"),
        (0, -50, 50, "右转（左正右负）"),
        (0, 0, 0, "静止（零差值）"),
        (10, 200, 180, "前进+小角度"),
    ]
    
    print(f"\n{'描述':<20} {'degree_x':<10} {'旧方式(count1)':<20} {'新方式(delta1)':<20} {'差异'}")
    print("-" * 90)
    
    for degree_x, count1, count2, description in test_cases:
        frame = create_test_frame(degree_x, count1, count2)
        old_result = parse_mpu_frame_old(frame)
        new_result = parse_mpu_frame_new(frame)
        
        # 对于负数的情况，显示差异
        if count1 < 0:
            unsigned_value = count1 + (1 << 32)  # 转为无符号表示
            diff_note = f"无符号会误读为{unsigned_value}"
        else:
            diff_note = "一致"
        
        print(f"{description:<20} {degree_x:<10} {old_result['count_run1']:<20} {new_result['delta_count1']:<20} {diff_note}")
    
    print("\n" + "=" * 70)
    print("关键变化说明：")
    print("=" * 70)
    print("1. count_run1/count_run2 从无符号改为有符号")
    print("2. 数据含义从累计值改为差值（delta）")
    print("3. 负值表示反向运动（后退或反转）")
    print("4. 不再需要计算差值和处理回绕问题")
    print("\n示例：")
    print("  - delta_count1=100, delta_count2=100  → 直线前进")
    print("  - delta_count1=-50, delta_count2=-50  → 直线后退")
    print("  - delta_count1=50, delta_count2=-50   → 原地左转")
    print("  - delta_count1=-50, delta_count2=50   → 原地右转")

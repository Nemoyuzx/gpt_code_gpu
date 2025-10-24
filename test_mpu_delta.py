#!/usr/bin/env python3
"""
测试MPU差值解析
验证count_run1和count_run2作为有符号差值的解析
"""

def parse_mpu_frame_old(frame: bytes) -> dict:
    """旧方式：32bit无符号累计值"""
    degree_x = int.from_bytes(frame[1:3], byteorder="big", signed=True)
    count_run1 = int.from_bytes(frame[3:7], byteorder="big", signed=False)
    count_run2 = int.from_bytes(frame[7:11], byteorder="big", signed=False)
    return {
        "degree_x": degree_x,
        "count_run1": count_run1,
        "count_run2": count_run2,
        "type": "32bit累计值(无符号)"
    }

def parse_mpu_frame_new(frame: bytes) -> dict:
    """新方式：16bit有符号差值"""
    degree_x = int.from_bytes(frame[1:3], byteorder="big", signed=True)
    count_run1 = int.from_bytes(frame[3:5], byteorder="big", signed=True)
    count_run2 = int.from_bytes(frame[5:7], byteorder="big", signed=True)
    return {
        "degree_x": degree_x,
        "delta_count1": count_run1,
        "delta_count2": count_run2,
        "type": "16bit差值(有符号)"
    }

def create_test_frame_old(degree_x: int, count1: int, count2: int) -> bytes:
    """创建旧格式测试帧 (32bit)"""
    frame = bytearray([0xA6])  # MPU header
    frame.extend(degree_x.to_bytes(2, byteorder="big", signed=True))
    frame.extend(count1.to_bytes(4, byteorder="big", signed=True))
    frame.extend(count2.to_bytes(4, byteorder="big", signed=True))
    return bytes(frame)

def create_test_frame_new(degree_x: int, count1: int, count2: int) -> bytes:
    """创建新格式测试帧 (16bit)"""
    frame = bytearray([0xA6])  # MPU header
    frame.extend(degree_x.to_bytes(2, byteorder="big", signed=True))
    frame.extend(count1.to_bytes(2, byteorder="big", signed=True))
    frame.extend(count2.to_bytes(2, byteorder="big", signed=True))
    return bytes(frame)

if __name__ == "__main__":
    print("=" * 70)
    print("MPU差值解析测试 (16bit vs 32bit)")
    print("=" * 70)
    
    test_cases = [
        (0, 100, 100, "前进（正差值）"),
        (0, -50, -50, "后退（负差值）"),
        (0, 50, -50, "左转（左负右正）"),
        (0, -50, 50, "右转（左正右负）"),
        (0, 0, 0, "静止（零差值）"),
        (10, 200, 180, "前进+小角度"),
        (0, 32767, 32767, "最大正值(16bit)"),
        (0, -32768, -32768, "最大负值(16bit)"),
    ]
    
    print(f"\n{'描述':<20} {'帧长度':<10} {'count1值':<15} {'解析说明'}")
    print("-" * 70)
    
    for degree_x, count1, count2, description in test_cases:
        # 新格式：16bit
        frame_new = create_test_frame_new(degree_x, count1, count2)
        new_result = parse_mpu_frame_new(frame_new)
        
        print(f"{description:<20} {len(frame_new)}字节     {new_result['delta_count1']:<15} 16bit有符号")
    
    print("\n" + "=" * 70)
    print("帧格式对比：")
    print("=" * 70)
    print("旧格式 (32bit):")
    print("  帧长度: 11字节")
    print("  结构: [头(1)] + [degree_x(2)] + [count1(4)] + [count2(4)]")
    print("  范围: ±2,147,483,648")
    print()
    print("新格式 (16bit):")
    print("  帧长度: 7字节 ✓ 节省36%空间")
    print("  结构: [头(1)] + [degree_x(2)] + [count1(2)] + [count2(2)]")
    print("  范围: ±32,768")
    print()
    print("=" * 70)
    print("关键变化说明：")
    print("=" * 70)
    print("1. count_run1/count_run2 从 32bit 改为 16bit")
    print("2. 数据含义：有符号差值（delta）")
    print("3. 帧长度从 11字节 减少到 7字节")
    print("4. 适用于差值范围在 ±32,768 以内的场景")
    print("\n示例：")
    print("  - delta_count1=100, delta_count2=100  → 直线前进")
    print("  - delta_count1=-50, delta_count2=-50  → 直线后退")
    print("  - delta_count1=50, delta_count2=-50   → 原地左转")
    print("  - delta_count1=-50, delta_count2=50   → 原地右转")

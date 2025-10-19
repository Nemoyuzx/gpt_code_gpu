#!/usr/bin/env python3
"""
小车不动问题诊断脚本
"""

import os
import sys

print("=" * 60)
print("小车控制诊断工具")
print("=" * 60)

# 1. 检查环境变量
print("\n[1] 检查环境变量:")
print("-" * 60)

env_vars = {
    "USE_REAL_BLE_DATA": os.getenv("USE_REAL_BLE_DATA", "未设置"),
    "ENABLE_CONTROL_LOOP": os.getenv("ENABLE_CONTROL_LOOP", "未设置"),
    "BLE_DEVICE_ADDRESS": os.getenv("BLE_DEVICE_ADDRESS", "未设置"),
    "BLE_NOTIFY_CHAR": os.getenv("BLE_NOTIFY_CHAR", "未设置"),
    "BLE_WRITE_CHAR": os.getenv("BLE_WRITE_CHAR", "未设置"),
}

issues = []
for key, value in env_vars.items():
    status = "✓" if value != "未设置" else "✗"
    print(f"  {status} {key}: {value}")
    if value == "未设置" and key in ["USE_REAL_BLE_DATA", "ENABLE_CONTROL_LOOP", "BLE_DEVICE_ADDRESS"]:
        issues.append(f"{key} 未设置")

# 2. 检查控制循环配置
print("\n[2] 控制循环状态:")
print("-" * 60)

use_ble = os.getenv("USE_REAL_BLE_DATA", "0") == "1"
enable_control = os.getenv("ENABLE_CONTROL_LOOP", "1") == "1"

if use_ble:
    print("  ✓ USE_REAL_BLE_DATA = 1 (启用真实小车)")
else:
    print("  ✗ USE_REAL_BLE_DATA = 0 (仿真模式)")
    issues.append("USE_REAL_BLE_DATA 未启用，运行在仿真模式")

if enable_control:
    print("  ✓ ENABLE_CONTROL_LOOP = 1 (启用控制)")
else:
    print("  ✗ ENABLE_CONTROL_LOOP = 0 (禁用控制)")
    issues.append("ENABLE_CONTROL_LOOP 未启用，不会发送控制命令")

# 3. 检查命令格式
print("\n[3] 命令格式测试:")
print("-" * 60)

try:
    from real_robot_bridge import MotorController
    
    cmdset_max_value = float(os.getenv("CMDSET_MAX_VALUE", "999"))
    cmdset_max_speed = max(1e-6, float(os.getenv("CMDSET_MAX_SPEED", "1.2")))
    default_speed_scale = float(
        os.getenv(
            "MOTOR_SPEED_SCALE",
            f"{cmdset_max_value / (cmdset_max_speed * 2000.0):.6f}"
        )
    )

    controller = MotorController(
        wheel_track=0.168,
        ticks_per_meter=2000.0,
    min_encoder_speed=30,
        min_linear_speed=0.003,
        speed_scale=default_speed_scale,
        turn_min_scale=0.5,
        max_turn_rate=0.6,
        turn_max_ticks=60.0,
    )
    
    # 测试用例
    test_cases = [
        (0.3, 0.0, "直线前进"),
        (0.0, 1.0, "原地左转"),
        (0.35, 0.52, "右转"),
        (0.01, 0.0, "极低速"),
        (0.005, 0.0, "超低速"),
    ]
    
    print("  测试速度命令转换:")
    for v, w, desc in test_cases:
        left, right = controller.velocity_to_encoder_speeds(v, w, 0.1)
        cmd = f"CMD-SET {left} {right}"
        print(f"    {desc:20s} v={v:.2f} w={w:.2f} → {cmd}")
        
        # 检查最小速度保护
        if v > 0 and abs(left) < 20:
            print(f"      ⚠️ 左轮速度 {left} 已被保护到 >= 20")
        if v > 0 and abs(right) < 20:
            print(f"      ⚠️ 右轮速度 {right} 已被保护到 >= 20")
    
    print("  ✓ 命令格式正常")
except Exception as e:
    print(f"  ✗ 命令格式测试失败: {e}")
    issues.append(f"命令生成异常: {e}")

# 4. 检查蓝牙连接
print("\n[4] 蓝牙连接检查:")
print("-" * 60)

ble_addr = os.getenv("BLE_DEVICE_ADDRESS")
if ble_addr and ble_addr != "未设置":
    print(f"  设备地址: {ble_addr}")
    
    # 检查地址格式
    if ":" in ble_addr or "-" in ble_addr:
        print("  ✓ 地址格式看起来正确")
    else:
        print("  ✗ 地址格式可能不正确（应该包含 : 或 -）")
        issues.append("BLE_DEVICE_ADDRESS 格式可能不正确")
else:
    print("  ✗ 未配置设备地址")
    issues.append("BLE_DEVICE_ADDRESS 未配置")

write_char = os.getenv("BLE_WRITE_CHAR")
if write_char and write_char != "未设置":
    print(f"  写入特征: {write_char}")
    print("  ✓ 写入特征已配置")
else:
    print("  ⚠️ BLE_WRITE_CHAR 未配置，将使用 BLE_NOTIFY_CHAR")

# 5. DWA 速度参数
print("\n[5] DWA 速度配置:")
print("-" * 60)

try:
    from dwa import LegacyDWAConfig
    cfg = LegacyDWAConfig()
    
    print(f"  max_speed: {cfg.max_speed:.2f} m/s")
    print(f"  max_yaw_rate: {cfg.max_yaw_rate * 180 / 3.14159:.1f}°/s")
    print(f"  max_accel: {cfg.max_accel:.2f} m/s²")
    print(f"  turn_min_speed_scale: {cfg.turn_min_speed_scale:.2f}")
    
    if cfg.max_speed < 0.1:
        print("  ⚠️ max_speed 过低，小车可能移动很慢")
        issues.append("max_speed 过低")
    else:
        print("  ✓ 速度配置正常")
        
except Exception as e:
    print(f"  ✗ 无法读取 DWA 配置: {e}")

# 总结
print("\n" + "=" * 60)
print("诊断总结")
print("=" * 60)

if not issues:
    print("✓ 所有检查通过！")
    print("\n如果小车仍不动，请检查：")
    print("  1. 小车是否已开机并充电充足")
    print("  2. 蓝牙是否实际连接成功（查看 [BLE] 日志）")
    print("  3. 是否看到 [MOTOR] CMD-SET 命令输出")
    print("  4. 小车固件是否支持 CMD-SET 命令格式")
    print("  5. DWA 是否输出了非零速度（查看 [CTRL] 日志）")
else:
    print("✗ 发现以下问题：")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n建议的修复步骤：")
    print("-" * 60)
    
    if "USE_REAL_BLE_DATA 未启用" in issues:
        print("  export USE_REAL_BLE_DATA=1")
    
    if "ENABLE_CONTROL_LOOP 未启用" in issues:
        print("  export ENABLE_CONTROL_LOOP=1")
    
    if "BLE_DEVICE_ADDRESS 未配置" in issues:
        print("  export BLE_DEVICE_ADDRESS=\"XX:XX:XX:XX:XX:XX\"")
    
    if any("BLE" in issue for issue in issues):
        print("\n  然后重新运行:")
        print("  python main.py")

print("=" * 60)

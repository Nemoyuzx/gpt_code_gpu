#!/usr/bin/env python3
"""
测试新的CMD-VW命令格式
验证v和w正确转换为整数值
"""

def format_cmd_vw(v: float, w: float) -> str:
    """
    生成CMD-VW命令
    
    Args:
        v: 线速度 (m/s)
        w: 角速度 (rad/s)
        
    Returns:
        命令字符串，格式: "CMD-VW v w" (v和w都乘以1000)
    """
    v_int = int(round(v * 1000))
    w_int = int(round(w * 1000))
    return f"CMD-VW {v_int} {w_int}"


if __name__ == "__main__":
    print("=" * 60)
    print("CMD-VW 命令格式测试")
    print("=" * 60)
    
    test_cases = [
        (0.0, 0.0, "停止"),
        (0.1, 0.0, "直行 0.1 m/s"),
        (0.05, 0.314, "前进转弯"),
        (0.0, 0.5, "原地旋转"),
        (0.3, -0.2, "前进反向转弯"),
        (-0.1, 0.0, "后退"),
        (0.035, 0.157, "慢速转弯"),
    ]
    
    print(f"\n{'描述':<20} {'v (m/s)':<12} {'w (rad/s)':<12} {'命令':<20}")
    print("-" * 64)
    
    for v, w, description in test_cases:
        command = format_cmd_vw(v, w)
        print(f"{description:<20} {v:<12.3f} {w:<12.3f} {command:<20}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n说明：")
    print("- v 和 w 都乘以1000转换为整数")
    print("- 命令格式：CMD-VW v_int w_int")
    print("- 例如：v=0.1 m/s, w=0.314 rad/s → CMD-VW 100 314")

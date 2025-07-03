import json
import matplotlib.pyplot as plt
import numpy as np

def load_maze_from_json(filename):
    """
    从JSON文件加载迷宫数据。
    
    参数:
        filename: JSON文件的路径
        
    返回:
        segments: 线段列表
        start_point: 起始点坐标
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        segments = data.get('segments', [])
        start_point = data.get('start_point', [0, 0])
        
        return segments, start_point
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filename}")
        return [], [0, 0]
    except json.JSONDecodeError:
        print(f"错误: {filename} 不是有效的JSON文件")
        return [], [0, 0]

def draw_json_maze(segments, start_point, title="JSON迷宫展示"):
    """
    使用Matplotlib绘制JSON格式的迷宫。
    
    参数:
        segments: 线段列表，每个线段包含start和end点
        start_point: 起始点坐标
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制所有线段
    for segment in segments:
        start = segment['start']
        end = segment['end']
        ax.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=2)
    
    # 标记起点
    ax.plot(start_point[0], start_point[1], 'go', markersize=12, label='起点')
    
    # 设置图表属性
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (米)')
    ax.set_ylabel('Y (米)')
    ax.set_title(title)
    ax.legend()
    
    # 自动调整显示范围
    if segments:
        all_x = []
        all_y = []
        for segment in segments:
            all_x.extend([segment['start'][0], segment['end'][0]])
            all_y.extend([segment['start'][1], segment['end'][1]])
        
        margin = 1
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    plt.tight_layout()
    plt.show()

def analyze_maze(segments, start_point):
    """
    分析迷宫的基本信息。
    
    参数:
        segments: 线段列表
        start_point: 起始点坐标
    """
    if not segments:
        print("没有找到有效的线段数据")
        return
    
    print(f"迷宫分析结果:")
    print(f"- 线段数量: {len(segments)}")
    print(f"- 起始点: ({start_point[0]}, {start_point[1]})")
    
    # 计算迷宫边界
    all_x = []
    all_y = []
    for segment in segments:
        all_x.extend([segment['start'][0], segment['end'][0]])
        all_y.extend([segment['start'][1], segment['end'][1]])
    
    if all_x and all_y:
        print(f"- X坐标范围: {min(all_x)} 到 {max(all_x)} (宽度: {max(all_x) - min(all_x)}米)")
        print(f"- Y坐标范围: {min(all_y)} 到 {max(all_y)} (高度: {max(all_y) - min(all_y)}米)")
        
        # 计算从起点到各个角落的距离
        corners = [
            (min(all_x), min(all_y)),
            (max(all_x), min(all_y)),
            (min(all_x), max(all_y)),
            (max(all_x), max(all_y))
        ]
        
        max_distance = 0
        for corner in corners:
            distance = np.sqrt((start_point[0] - corner[0])**2 + (start_point[1] - corner[1])**2)
            max_distance = max(max_distance, distance)
        
        print(f"- 从起点到最远角落的距离: {max_distance:.2f}米")

def create_sample_json():
    """
    创建一个示例JSON文件用于测试。
    """
    sample_data = {
        "segments": [
            {"start": [0, 0], "end": [2, 0]},
            {"start": [2, 0], "end": [2, 2]},
            {"start": [0, 0], "end": [0, 15]},
            {"start": [0, 11], "end": [2, 11]},
            {"start": [2, 11], "end": [2, 6]},
            {"start": [2, 6], "end": [4, 6]},
            {"start": [0, 15], "end": [11, 15]},
            {"start": [2, 15], "end": [2, 13]},
            {"start": [2, 13], "end": [9, 13]},
            {"start": [4, 13], "end": [4, 8]},
            {"start": [6, 13], "end": [6, 10]},
            {"start": [6, 10], "end": [9, 10]},
            {"start": [9, 10], "end": [9, 13]},
            {"start": [11, 15], "end": [11, 10]},
            {"start": [4, 0], "end": [4, 2]},
            {"start": [4, 0], "end": [15, 0]}
        ],
        "start_point": [3, 0]
    }
    
    with open('sample_maze.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("已创建示例文件: sample_maze.json")

if __name__ == '__main__':
    # 创建示例JSON文件
    create_sample_json()
    
    # 加载和显示迷宫
    filename = '2.json'
    segments, start_point = load_maze_from_json(filename)
    
    if segments:
        print(f"成功加载迷宫文件: {filename}")
        analyze_maze(segments, start_point)
        draw_json_maze(segments, start_point, f"迷宫展示 - {filename}")
    else:
        print("无法加载迷宫数据")

"""
网格单元系统模块
实现9x9网格系统，用于迷宫探索中的区域划分
"""
from typing import List, Optional, Tuple


class GridCell:
    """表示一个0.7x0.7m的网格单元"""
    def __init__(self, cell_id: int, grid_row: int, grid_col: int, center_x: float, center_y: float, size: float = 0.7):
        self.id = cell_id  # 单元格序号（1-81）
        self.row = grid_row  # 网格行号（0-8）
        self.col = grid_col  # 网格列号（0-8）
        self.center_x = center_x  # 单元格中心X坐标（世界坐标）
        self.center_y = center_y  # 单元格中心Y坐标（世界坐标）
        self.size = size  # 单元格边长（米）
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """返回单元格边界 (min_x, min_y, max_x, max_y)"""
        half = self.size / 2
        return (
            self.center_x - half,
            self.center_y - half,
            self.center_x + half,
            self.center_y + half
        )
    
    def contains_point(self, x: float, y: float) -> bool:
        """判断点(x,y)是否在此单元格内"""
        min_x, min_y, max_x, max_y = self.get_bounds()
        return min_x <= x <= max_x and min_y <= y <= max_y
    
    def __repr__(self):
        return f"Cell#{self.id}[{self.row},{self.col}]@({self.center_x:.2f},{self.center_y:.2f})"


class GridSystem:
    """9x9网格系统，以小车初始位置为中心"""
    def __init__(self, robot_x: float, robot_y: float, cell_size: float = 0.7):
        self.cell_size = cell_size
        self.robot_x = robot_x
        self.robot_y = robot_y
        self.cells: List[GridCell] = []
        self._build_grid()
    
    def _build_grid(self):
        """构建9x9网格，机器人位置为中心(4,4)"""
        cell_id = 1
        for row in range(9):
            for col in range(9):
                # 计算相对于中心的偏移
                offset_row = row - 4  # -4 to 4
                offset_col = col - 4  # -4 to 4
                
                # 计算单元格中心坐标
                center_x = self.robot_x + offset_col * self.cell_size
                center_y = self.robot_y + offset_row * self.cell_size
                
                cell = GridCell(cell_id, row, col, center_x, center_y, self.cell_size)
                self.cells.append(cell)
                cell_id += 1
    
    def get_cell_by_id(self, cell_id: int) -> Optional[GridCell]:
        """通过序号获取单元格"""
        for cell in self.cells:
            if cell.id == cell_id:
                return cell
        return None
    
    def get_cell_at_position(self, x: float, y: float) -> Optional[GridCell]:
        """获取包含指定坐标的单元格"""
        for cell in self.cells:
            if cell.contains_point(x, y):
                return cell
        return None
    
    def get_center_cell(self) -> GridCell:
        """获取中心单元格（机器人初始位置）"""
        return self.cells[40]  # 第41个单元格（row=4, col=4）

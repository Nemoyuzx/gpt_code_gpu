"""
异步可视化模块
在独立线程中运行可视化，不阻塞主计算逻辑
"""
import threading
import queue
from visualizer import Visualizer


class AsyncVisualizer:
    """
    异步可视化器，在独立线程中运行，不阻塞主计算逻辑
    """
    def __init__(self, visualizer: Visualizer, max_queue_size: int = 3):
        self.visualizer = visualizer
        self.update_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = None
        
    def start(self):
        """启动可视化线程"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._visualization_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """停止可视化线程"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            
    def _visualization_loop(self):
        """可视化线程主循环"""
        while self.running:
            try:
                # 非阻塞获取最新的更新数据，超时时间很短
                update_data = self.update_queue.get(timeout=0.01)
                
                # 清空队列中的旧数据，只保留最新的
                while not self.update_queue.empty():
                    try:
                        update_data = self.update_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # 执行可视化更新
                if update_data is not None:
                    args, kwargs = update_data
                    try:
                        self.visualizer.update(*args, **kwargs)
                    except Exception as e:
                        print(f"[VIZ ERROR] Visualization update failed: {e}")
                        
            except queue.Empty:
                # 队列为空时继续等待
                continue
            except Exception as e:
                print(f"[VIZ ERROR] Visualization loop error: {e}")
                
    def update(self, *args, **kwargs):
        """
        非阻塞更新可视化（将数据放入队列）
        如果队列满了，丢弃旧数据
        """
        if not self.running:
            return
            
        update_data = (args, kwargs)
        
        # 如果队列满了，尝试移除最旧的数据
        if self.update_queue.full():
            try:
                self.update_queue.get_nowait()
            except queue.Empty:
                pass
        
        # 将新数据放入队列（非阻塞）
        try:
            self.update_queue.put_nowait(update_data)
        except queue.Full:
            # 队列满了就跳过这一帧，不影响主计算
            pass
    
    # 代理所有Visualizer的属性和方法
    def __getattr__(self, name):
        """代理访问底层visualizer的属性和方法"""
        return getattr(self.visualizer, name)

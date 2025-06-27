from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QSizePolicy, QApplication
from PyQt5.QtCore import pyqtSignal, QTimer, Qt
from typing import Dict, Any
from src.utils.config_manager import config_manager

class BaseTab(QWidget):
    """所有标签页的基类，提供通用功能"""
    
    # 通用信号
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self._config_applied = False
        self._config_hash = None
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)  # 减少边距
        self.layout.setSpacing(0)  # 减少间距
        
        # 创建滚动区域
        scroll = self.create_scroll_area()
        self.scroll_content = QWidget()
        scroll.setWidget(self.scroll_content)
        self.layout.addWidget(scroll)
        
        # 连接标签页切换信号，确保在标签页激活时刷新布局
        if main_window and hasattr(main_window, 'tabs'):
            main_window.tabs.currentChanged.connect(self.on_tab_changed)
        
        print(f"{self.__class__.__name__}: BaseTab初始化完成")
        
    def create_scroll_area(self):
        """创建一个滚动区域"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll.setFrameShape(QScrollArea.NoFrame)  # 移除边框
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # 设置最小高度，避免内容区域过小
        scroll.setMinimumHeight(300)
        return scroll
        
    def update_status(self, message):
        """更新状态信息"""
        self.status_updated.emit(message)
        
    def update_progress(self, value):
        """更新进度条"""
        self.progress_updated.emit(value)
        
    def on_tab_changed(self, index):
        """处理标签页切换事件，当前标签页被激活时刷新布局"""
        if self.main_window and hasattr(self.main_window, 'tabs'):
            current_widget = self.main_window.tabs.widget(index)
            if current_widget == self:
                # 使用定时器延迟执行布局刷新，确保所有控件都已加载完成
                QTimer.singleShot(10, self.refresh_layout)
    
    def refresh_layout(self):
        """强制刷新整个标签页的布局"""
        # 防止递归调用导致无限循环
        if hasattr(self, '_is_refreshing') and self._is_refreshing:
            return
            
        self._is_refreshing = True
        
        try:
            # 刷新滚动内容区域
            if hasattr(self, 'scroll_content'):
                # 更强制的调整大小方式
                self.scroll_content.adjustSize()
                self.scroll_content.updateGeometry()
                
                # 确保滚动区域内容从顶部开始显示
                if hasattr(self, 'layout') and self.layout.count() > 0:
                    scroll_area = self.layout.itemAt(0).widget()
                    if isinstance(scroll_area, QScrollArea):
                        # 滚动到顶部
                        scroll_area.verticalScrollBar().setValue(0)
                        # 确保内容正确调整大小
                        scroll_area.updateGeometry()
            
            # 更新整个布局链
            parent = self.parent()
            while parent:
                parent.updateGeometry()
                parent = parent.parent()
            
            # 更新自身布局
            self.updateGeometry()
            self.update()
            
            # 处理所有待处理的事件
            QApplication.processEvents()
            
            # 延迟再次执行一次简单刷新，解决某些情况下第一次刷新不完全的问题
            QTimer.singleShot(50, self._delayed_refresh)
        finally:
            self._is_refreshing = False
    
    def _delayed_refresh(self):
        """延迟执行的简单刷新，避免无限递归"""
        # 简单刷新，不会再调用refresh_layout
        if hasattr(self, 'scroll_content'):
            self.scroll_content.adjustSize()
            self.scroll_content.updateGeometry()
        
        # 简单更新UI
        self.update() 

    def apply_config(self, config: Dict[str, Any]):
        """智能配置应用，避免重复操作"""
        if not config:
            print(f"{self.__class__.__name__}: 配置为空，跳过应用")
            return
            
        # 计算配置哈希值
        current_hash = hash(str(sorted(config.items())))
        
        # 检查配置是否有变化
        if self._config_applied and self._config_hash == current_hash:
            print(f"{self.__class__.__name__}: 配置未变化，跳过重复应用")
            return
            
        print(f"{self.__class__.__name__}: 应用新配置...")
        
        # 调用子类的配置应用方法
        try:
            self._do_apply_config(config)
            self._config_applied = True
            self._config_hash = current_hash
            print(f"{self.__class__.__name__}: 配置应用成功")
        except Exception as e:
            print(f"{self.__class__.__name__}: 配置应用失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _do_apply_config(self, config: Dict[str, Any]):
        """子类需要重写此方法来实现具体的配置应用逻辑"""
        pass
    
    def get_config_from_manager(self) -> Dict[str, Any]:
        """从集中化配置管理器获取配置"""
        return config_manager.get_config()
    
    def force_reload_config(self):
        """强制重新加载配置"""
        self._config_applied = False
        self._config_hash = None
        config = self.get_config_from_manager()
        if config:
            self.apply_config(config)
    
    def is_config_applied(self) -> bool:
        """检查配置是否已应用"""
        return self._config_applied 
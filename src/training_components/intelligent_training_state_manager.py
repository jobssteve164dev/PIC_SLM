"""
智能训练状态管理器

用于区分智能训练的重启过程和真正的训练停止，
避免在微调参数时显示误导性的"训练已停止"弹窗。
"""

from enum import Enum
from typing import Optional, Dict, Any
from PyQt5.QtCore import QObject, pyqtSignal


class TrainingState(Enum):
    """训练状态枚举"""
    IDLE = "idle"                    # 空闲状态
    RUNNING = "running"              # 正常训练中
    INTELLIGENT_RESTARTING = "intelligent_restarting"  # 智能训练重启中
    STOPPED = "stopped"              # 真正停止
    COMPLETED = "completed"          # 训练完成
    ERROR = "error"                  # 训练错误


class IntelligentTrainingStateManager(QObject):
    """智能训练状态管理器"""
    
    # 信号定义
    state_changed = pyqtSignal(TrainingState, str)  # 状态改变信号
    intelligent_restart_started = pyqtSignal(dict)  # 智能重启开始信号
    intelligent_restart_completed = pyqtSignal(dict)  # 智能重启完成信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_state = TrainingState.IDLE
        self.restart_context: Optional[Dict[str, Any]] = None
        
    def set_state(self, new_state: TrainingState, message: str = ""):
        """设置训练状态"""
        if self.current_state != new_state:
            old_state = self.current_state
            self.current_state = new_state
            
            # 发射状态改变信号
            self.state_changed.emit(new_state, message)
            
            # 特殊处理智能重启状态
            if new_state == TrainingState.INTELLIGENT_RESTARTING:
                self.intelligent_restart_started.emit(self.restart_context or {})
            elif old_state == TrainingState.INTELLIGENT_RESTARTING and new_state == TrainingState.RUNNING:
                self.intelligent_restart_completed.emit(self.restart_context or {})
    
    def start_intelligent_restart(self, context: Dict[str, Any]):
        """开始智能重启过程"""
        self.restart_context = context
        self.set_state(TrainingState.INTELLIGENT_RESTARTING, "智能训练正在重启...")
    
    def complete_intelligent_restart(self):
        """完成智能重启过程"""
        self.set_state(TrainingState.RUNNING, "智能训练已重启")
        self.restart_context = None
    
    def start_training(self):
        """开始训练"""
        self.set_state(TrainingState.RUNNING, "训练已开始")
    
    def stop_training(self):
        """停止训练"""
        self.set_state(TrainingState.STOPPED, "训练已停止")
    
    def complete_training(self):
        """完成训练"""
        self.set_state(TrainingState.COMPLETED, "训练已完成")
    
    def set_error(self, error_message: str):
        """设置错误状态"""
        self.set_state(TrainingState.ERROR, f"训练错误: {error_message}")
    
    def is_intelligent_restarting(self) -> bool:
        """检查是否正在智能重启"""
        return self.current_state == TrainingState.INTELLIGENT_RESTARTING
    
    def is_training_active(self) -> bool:
        """检查训练是否活跃（运行中或重启中）"""
        return self.current_state in [TrainingState.RUNNING, TrainingState.INTELLIGENT_RESTARTING]
    
    def get_current_state(self) -> TrainingState:
        """获取当前状态"""
        return self.current_state
    
    def get_state_message(self) -> str:
        """获取状态消息"""
        state_messages = {
            TrainingState.IDLE: "训练空闲",
            TrainingState.RUNNING: "训练中...",
            TrainingState.INTELLIGENT_RESTARTING: "智能训练重启中...",
            TrainingState.STOPPED: "训练已停止",
            TrainingState.COMPLETED: "训练完成",
            TrainingState.ERROR: "训练错误"
        }
        return state_messages.get(self.current_state, "未知状态")

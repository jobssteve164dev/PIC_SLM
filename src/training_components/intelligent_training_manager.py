"""
智能训练管理器 - 协调智能训练编排器与现有训练系统

主要功能：
- 管理智能训练编排器的生命周期
- 协调训练系统的启动、停止、重启
- 处理参数更新和配置同步
- 提供统一的接口给UI层
"""

import os
import json
import time
import threading
from typing import Dict, Any, Optional, List
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import QMessageBox

from .intelligent_training_orchestrator import IntelligentTrainingOrchestrator
from .model_trainer import ModelTrainer
from .training_thread import TrainingThread


class IntelligentTrainingManager(QObject):
    """智能训练管理器"""
    
    # 信号定义
    training_started = pyqtSignal(dict)  # 训练开始信号
    training_completed = pyqtSignal(dict)  # 训练完成信号
    training_failed = pyqtSignal(dict)  # 训练失败信号
    training_stopped = pyqtSignal(dict)  # 训练停止信号
    training_restarted = pyqtSignal(dict)  # 训练重启信号
    config_generated = pyqtSignal(dict)  # 配置生成信号
    config_applied = pyqtSignal(dict)  # 配置应用信号
    iteration_completed = pyqtSignal(dict)  # 迭代完成信号
    intervention_occurred = pyqtSignal(dict)  # 干预发生信号
    analysis_completed = pyqtSignal(dict)  # 分析完成信号
    status_updated = pyqtSignal(str)  # 状态更新信号
    error_occurred = pyqtSignal(str)  # 错误信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.intelligent_orchestrator = None
        self.model_trainer = None
        self.training_tab = None
        self.current_training_config = None
        self.is_intelligent_mode = False
        
        # 训练状态管理
        self.training_status = 'idle'  # idle, running, stopped, restarting
        self.intervention_count = 0
        self.best_checkpoint_path = None
        
        # 配置管理
        self.config_file = "setting/intelligent_training_config.json"
        self.config = self._load_default_config()
        
        # 初始化组件
        self._initialize_components()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        default_config = {
            'auto_intervention_enabled': True,
            'intervention_thresholds': {
                'overfitting_risk': 0.8,
                'underfitting_risk': 0.7,
                'stagnation_epochs': 10,
                'divergence_threshold': 2.0,
                'min_training_epochs': 10,
            },
            'analysis_interval': 20,
            'max_interventions_per_session': 3,
            'parameter_tuning_strategy': 'conservative',
            'training_restart': {
                'max_restart_attempts': 3,
                'restart_delay': 5,
            }
        }
        
        # 尝试加载配置文件
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    default_config.update(saved_config)
            except Exception as e:
                print(f"加载配置文件失败: {str(e)}")
        
        return default_config
    
    def _initialize_components(self):
        """初始化相关组件"""
        try:
            # 初始化智能训练编排器
            self.intelligent_orchestrator = IntelligentTrainingOrchestrator()
            
            # 连接信号
            self.intelligent_orchestrator.training_started.connect(self._on_training_started)
            self.intelligent_orchestrator.training_completed.connect(self._on_training_completed)
            self.intelligent_orchestrator.training_failed.connect(self._on_training_failed)
            self.intelligent_orchestrator.config_generated.connect(self._on_config_generated)
            self.intelligent_orchestrator.config_applied.connect(self._on_config_applied)
            self.intelligent_orchestrator.iteration_completed.connect(self._on_iteration_completed)
            self.intelligent_orchestrator.status_updated.connect(self._on_orchestrator_status_updated)
            self.intelligent_orchestrator.error_occurred.connect(self._on_orchestrator_error)
            self.intelligent_orchestrator.apply_config_requested.connect(self._on_apply_config_requested)
            
            # 连接状态管理器信号
            state_manager = self.intelligent_orchestrator.get_state_manager()
            state_manager.state_changed.connect(self._on_state_changed)
            
            self.status_updated.emit("智能训练管理器初始化完成")
            
        except Exception as e:
            self.error_occurred.emit(f"初始化组件失败: {str(e)}")
    
    def set_model_trainer(self, model_trainer: ModelTrainer):
        """设置模型训练器"""
        self.model_trainer = model_trainer
        
        # 将训练器传递给智能编排器
        if self.intelligent_orchestrator:
            self.intelligent_orchestrator.set_model_trainer(model_trainer)
    
    def set_training_tab(self, training_tab):
        """设置训练标签页"""
        self.training_tab = training_tab
        
        # 将训练标签页传递给智能编排器
        if self.intelligent_orchestrator:
            self.intelligent_orchestrator.set_training_tab(training_tab)
    
    def start_intelligent_training(self, training_config: Dict[str, Any]):
        """启动智能训练"""
        try:
            if self.training_status == 'running':
                self.error_occurred.emit("训练已在运行中")
                return
            
            # 保存训练配置
            self.current_training_config = training_config.copy()
            
            # 启动智能训练编排器
            if self.intelligent_orchestrator:
                self.intelligent_orchestrator.start_intelligent_training(training_config)
                self.is_intelligent_mode = True
            
            self.training_status = 'running'
            self.status_updated.emit("智能训练已启动")
            
        except Exception as e:
            self.error_occurred.emit(f"启动智能训练失败: {str(e)}")
    
    def stop_intelligent_training(self):
        """停止智能训练"""
        try:
            # 停止智能训练编排器
            if self.intelligent_orchestrator:
                self.intelligent_orchestrator.stop_intelligent_training()
                self.is_intelligent_mode = False
            
            self.training_status = 'stopped'
            self.status_updated.emit("智能训练已停止")
            
        except Exception as e:
            self.error_occurred.emit(f"停止智能训练失败: {str(e)}")
    
    def update_training_progress(self, metrics: Dict[str, Any]):
        """更新训练进度"""
        try:
            # 将进度信息传递给智能编排器
            if self.intelligent_orchestrator:
                self.intelligent_orchestrator.update_training_progress(metrics)
                
        except Exception as e:
            self.error_occurred.emit(f"更新训练进度失败: {str(e)}")
    
    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """获取干预历史"""
        try:
            if self.intelligent_orchestrator:
                return self.intelligent_orchestrator.get_intervention_history()
            return []
        except Exception as e:
            self.error_occurred.emit(f"获取干预历史失败: {str(e)}")
            return []
    
    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """获取当前会话信息"""
        try:
            if self.intelligent_orchestrator:
                return self.intelligent_orchestrator.get_current_session_info()
            return None
        except Exception as e:
            self.error_occurred.emit(f"获取会话信息失败: {str(e)}")
            return None
    
    def save_session_report(self, file_path: str):
        """保存会话报告"""
        try:
            if self.intelligent_orchestrator:
                self.intelligent_orchestrator.save_session_report(file_path)
            else:
                self.error_occurred.emit("智能训练编排器未初始化")
        except Exception as e:
            self.error_occurred.emit(f"保存会话报告失败: {str(e)}")
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        try:
            self.config.update(new_config)
            
            # 保存配置到文件
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            self.status_updated.emit("配置已更新")
            
        except Exception as e:
            self.error_occurred.emit(f"更新配置失败: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()
    
    # 信号处理方法
    def _on_training_started(self, data: Dict[str, Any]):
        """处理训练开始信号"""
        try:
            self.training_started.emit(data)
            self.status_updated.emit("训练已开始")
        except Exception as e:
            self.error_occurred.emit(f"处理训练开始事件失败: {str(e)}")
    
    def _on_training_completed(self, data: Dict[str, Any]):
        """处理训练完成信号"""
        try:
            self.training_completed.emit(data)
            self.status_updated.emit("训练已完成")
        except Exception as e:
            self.error_occurred.emit(f"处理训练完成事件失败: {str(e)}")
    
    def _on_training_failed(self, data: Dict[str, Any]):
        """处理训练失败信号"""
        try:
            self.training_failed.emit(data)
            self.status_updated.emit("训练失败")
        except Exception as e:
            self.error_occurred.emit(f"处理训练失败事件失败: {str(e)}")
    
    def _on_config_generated(self, data: Dict[str, Any]):
        """处理配置生成信号"""
        try:
            self.config_generated.emit(data)
            self.status_updated.emit("新配置已生成")
        except Exception as e:
            self.error_occurred.emit(f"处理配置生成事件失败: {str(e)}")
    
    def _on_config_applied(self, data: Dict[str, Any]):
        """处理配置应用信号"""
        try:
            self.config_applied.emit(data)
            self.status_updated.emit("配置已应用")
        except Exception as e:
            self.error_occurred.emit(f"处理配置应用事件失败: {str(e)}")
    
    def _on_iteration_completed(self, data: Dict[str, Any]):
        """处理迭代完成信号"""
        try:
            self.iteration_completed.emit(data)
            self.status_updated.emit(f"第 {data.get('iteration', '?')} 次迭代已完成")
        except Exception as e:
            self.error_occurred.emit(f"处理迭代完成事件失败: {str(e)}")
    
    def _on_orchestrator_status_updated(self, message: str):
        """处理编排器状态更新信号"""
        try:
            self.status_updated.emit(message)
        except Exception as e:
            self.error_occurred.emit(f"处理状态更新事件失败: {str(e)}")
    
    def _on_orchestrator_error(self, error_message: str):
        """处理编排器错误信号"""
        try:
            self.error_occurred.emit(error_message)
        except Exception as e:
            self.error_occurred.emit(f"处理错误事件失败: {str(e)}")
    
    def _on_training_stopped(self, data: Dict[str, Any]):
        """处理训练停止信号"""
        try:
            # 检查是否是智能训练重启
            is_intelligent_restart = False
            if self.intelligent_orchestrator:
                state_manager = self.intelligent_orchestrator.get_state_manager()
                is_intelligent_restart = state_manager.is_intelligent_restarting()
            
            # 传递重启状态信息
            data['is_intelligent_restart'] = is_intelligent_restart
            self.training_stopped.emit(data)
            
            if is_intelligent_restart:
                self.status_updated.emit("智能训练重启中...")
            else:
                self.status_updated.emit("训练已停止")
        except Exception as e:
            self.error_occurred.emit(f"处理训练停止事件失败: {str(e)}")
    
    def _on_apply_config_requested(self, request_data: Dict[str, Any]):
        """处理配置应用请求信号"""
        try:
            # 将请求转发给编排器处理（在主线程中执行）
            if self.intelligent_orchestrator:
                self.intelligent_orchestrator.apply_config_request(request_data)
        except Exception as e:
            self.error_occurred.emit(f"处理配置应用请求失败: {str(e)}")
    
    def _on_state_changed(self, new_state, message: str):
        """处理状态改变事件"""
        try:
            # 根据状态类型发射相应的信号
            if new_state.value == 'intelligent_restarting':
                self.status_updated.emit("🔄 智能训练正在重启...")
            elif new_state.value == 'running':
                self.status_updated.emit("🚀 训练已开始")
            elif new_state.value == 'stopped':
                self.status_updated.emit("⏹️ 训练已停止")
            elif new_state.value == 'completed':
                self.status_updated.emit("✅ 训练已完成")
            elif new_state.value == 'error':
                self.status_updated.emit(f"❌ {message}")
        except Exception as e:
            self.error_occurred.emit(f"处理状态改变事件失败: {str(e)}")
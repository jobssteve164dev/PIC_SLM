"""
实时训练指标采集器 - 非侵入式数据采集

主要功能：
- 监听TensorBoard数据流，复制一份到本地文件
- 为AI分析提供实时训练数据
- 支持多种训练框架（分类、检测）
- 自动清理历史数据，保持文件大小合理
"""

import os
import json
import time
import threading
from typing import Dict, Any, List, Optional
from collections import deque
from PyQt5.QtCore import QObject, pyqtSignal
import torch
from torch.utils.tensorboard.writer import SummaryWriter


class RealTimeMetricsCollector(QObject):
    """实时训练指标采集器"""
    
    # 信号定义
    metrics_updated = pyqtSignal(dict)  # 新指标数据可用
    analysis_data_ready = pyqtSignal(str)  # 分析数据文件就绪
    
    def __init__(self, max_history=100):
        super().__init__()
        self.max_history = max_history
        self.metrics_buffer = deque(maxlen=max_history)
        self.current_training_session = None
        self.data_file_path = None
        self.lock = threading.Lock()
        
        # 创建数据存储目录
        self.data_dir = "logs/real_time_metrics"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def start_collection(self, training_session_id: str = None):
        """开始数据采集"""
        if training_session_id is None:
            training_session_id = f"training_{int(time.time())}"
            
        self.current_training_session = training_session_id
        self.data_file_path = os.path.join(
            self.data_dir, 
            f"{training_session_id}_metrics.json"
        )
        
        # 初始化数据文件
        initial_data = {
            "session_id": training_session_id,
            "start_time": time.time(),
            "metrics_history": [],
            "current_metrics": {},
            "training_status": "started"
        }
        
        with open(self.data_file_path, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)
            
        print(f"✅ 开始实时指标采集，会话ID: {training_session_id}")
        print(f"📁 数据文件: {self.data_file_path}")
        
    def collect_tensorboard_metrics(self, epoch: int, phase: str, metrics: Dict[str, Any]):
        """采集TensorBoard指标数据（非侵入式）"""
        if not self.current_training_session:
            return
            
        timestamp = time.time()
        
        # 构建标准化指标数据
        standardized_metrics = {
            "timestamp": timestamp,
            "epoch": epoch,
            "phase": phase,
            "session_id": self.current_training_session,
            **metrics
        }
        
        # 添加到缓冲区
        with self.lock:
            self.metrics_buffer.append(standardized_metrics)
            
        # 更新数据文件
        self._update_data_file(standardized_metrics)
        
        # 发送信号
        self.metrics_updated.emit(standardized_metrics)
        
    def collect_scalar_metric(self, tag: str, value: float, step: int):
        """采集单个标量指标（模拟TensorBoard的add_scalar）"""
        if not self.current_training_session:
            return
            
        timestamp = time.time()
        
        # 解析tag获取相关信息
        phase = "train" if "train" in tag.lower() else "val" if "val" in tag.lower() else "unknown"
        metric_type = tag.split('/')[-1] if '/' in tag else tag
        
        metric_data = {
            "timestamp": timestamp,
            "epoch": step,
            "phase": phase,
            "tag": tag,
            "metric_type": metric_type,
            "value": float(value),
            "session_id": self.current_training_session
        }
        
        # 添加到缓冲区
        with self.lock:
            self.metrics_buffer.append(metric_data)
            
        # 更新数据文件
        self._update_data_file(metric_data)
        
        # 发送信号
        self.metrics_updated.emit(metric_data)
        
    def get_current_training_data_for_ai(self) -> Dict[str, Any]:
        """获取当前训练数据供AI分析使用"""
        # 如果没有活动会话，尝试找到最新的数据文件
        if not self.data_file_path or not os.path.exists(self.data_file_path):
            latest_file = self._find_latest_metrics_file()
            if latest_file:
                self.data_file_path = latest_file
                print(f"📁 使用最新的训练数据文件: {latest_file}")
            else:
                return {"error": "没有可用的训练数据"}
            
        try:
            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 提取最新的关键指标
            metrics_history = data.get("metrics_history", [])
            if not metrics_history:
                return {"error": "训练数据为空"}
                
            # 获取最新的训练指标
            latest_metrics = metrics_history[-1] if metrics_history else {}
            
            # 计算训练趋势
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            epochs = []
            
            for metric in metrics_history[-10:]:  # 最近10个数据点
                if metric.get("phase") == "train":
                    if "loss" in metric:
                        train_losses.append(metric["loss"])
                    if "accuracy" in metric:
                        train_accs.append(metric["accuracy"])
                elif metric.get("phase") == "val":
                    if "loss" in metric:
                        val_losses.append(metric["loss"])
                    if "accuracy" in metric:
                        val_accs.append(metric["accuracy"])
                        
                if "epoch" in metric:
                    epochs.append(metric["epoch"])
            
            # 构建AI分析用的数据结构
            ai_data = {
                "session_id": self.current_training_session,
                "current_metrics": latest_metrics,
                "training_trends": {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accuracies": train_accs,
                    "val_accuracies": val_accs,
                    "epochs": list(set(epochs))[-10:] if epochs else []
                },
                "training_status": data.get("training_status", "unknown"),
                "total_data_points": len(metrics_history),
                "collection_duration": time.time() - (data.get("start_time") or time.time())
            }
            
            return ai_data
            
        except Exception as e:
            return {"error": f"读取训练数据失败: {str(e)}"}
            
    def get_formatted_metrics_for_analysis(self) -> str:
        """获取格式化的指标数据供AI分析"""
        ai_data = self.get_current_training_data_for_ai()
        
        if "error" in ai_data:
            return f"数据获取失败: {ai_data['error']}"
            
        current = ai_data.get("current_metrics", {})
        trends = ai_data.get("training_trends", {})
        
        # 构建分析文本
        analysis_text = f"""
## 当前训练状态数据

### 基本信息
- 训练会话: {ai_data.get('session_id', 'Unknown')}
- 数据采集时长: {ai_data.get('collection_duration', 0):.1f}秒
- 总数据点: {ai_data.get('total_data_points', 0)}个

### 最新指标
- 当前Epoch: {current.get('epoch', 'N/A')}
- 训练阶段: {current.get('phase', 'N/A')}
- 损失值: {current.get('loss', 'N/A')}
- 准确率: {current.get('accuracy', 'N/A')}
- 时间戳: {current.get('timestamp', 'N/A')}

### 训练趋势（最近10个数据点）
- 训练损失趋势: {trends.get('train_losses', [])}
- 验证损失趋势: {trends.get('val_losses', [])}
- 训练准确率趋势: {trends.get('train_accuracies', [])}
- 验证准确率趋势: {trends.get('val_accuracies', [])}
- Epoch序列: {trends.get('epochs', [])}

### 训练状态
- 状态: {ai_data.get('training_status', 'unknown')}
"""
        
        return analysis_text.strip()
        
    def stop_collection(self):
        """停止数据采集"""
        if self.current_training_session and self.data_file_path:
            # 更新训练状态为完成
            self._update_training_status("completed")
            print(f"✅ 训练指标采集已停止，会话: {self.current_training_session}")
            
        self.current_training_session = None
        self.data_file_path = None
        
    def _update_data_file(self, new_metric: Dict[str, Any]):
        """更新数据文件"""
        if not self.data_file_path:
            return
            
        try:
            # 读取现有数据
            if os.path.exists(self.data_file_path):
                with open(self.data_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {
                    "session_id": self.current_training_session,
                    "start_time": time.time(),
                    "metrics_history": [],
                    "current_metrics": {},
                    "training_status": "running"
                }
            
            # 添加新指标
            data["metrics_history"].append(new_metric)
            data["current_metrics"] = new_metric
            data["last_update"] = time.time()
            
            # 保持历史数据在合理范围内
            if len(data["metrics_history"]) > self.max_history:
                data["metrics_history"] = data["metrics_history"][-self.max_history:]
            
            # 写回文件
            with open(self.data_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            # 发送数据文件就绪信号
            self.analysis_data_ready.emit(self.data_file_path)
            
        except Exception as e:
            print(f"❌ 更新数据文件失败: {str(e)}")
            
    def _update_training_status(self, status: str):
        """更新训练状态"""
        if not self.data_file_path or not os.path.exists(self.data_file_path):
            return
            
        try:
            with open(self.data_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            data["training_status"] = status
            data["end_time"] = time.time()
            
            with open(self.data_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ 更新训练状态失败: {str(e)}")
            
    def cleanup_old_data(self, days_to_keep: int = 7):
        """清理旧的数据文件"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (days_to_keep * 24 * 3600)
            
            for filename in os.listdir(self.data_dir):
                if filename.endswith('_metrics.json'):
                    file_path = os.path.join(self.data_dir, filename)
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        print(f"🗑️ 已清理旧数据文件: {filename}")
                        
        except Exception as e:
            print(f"❌ 清理旧数据失败: {str(e)}")
    
    def _find_latest_metrics_file(self) -> Optional[str]:
        """查找最新的训练指标文件"""
        try:
            import glob
            pattern = os.path.join(self.data_dir, "*_metrics.json")
            files = glob.glob(pattern)
            
            if not files:
                return None
                
            # 按修改时间排序，返回最新的文件
            latest_file = max(files, key=os.path.getmtime)
            return latest_file
            
        except Exception as e:
            print(f"❌ 查找最新数据文件失败: {str(e)}")
            return None


# 全局实例
_global_collector = None

def get_global_metrics_collector() -> RealTimeMetricsCollector:
    """获取全局指标采集器实例"""
    global _global_collector
    if _global_collector is None:
        _global_collector = RealTimeMetricsCollector()
    return _global_collector


class TensorBoardInterceptor:
    """TensorBoard数据拦截器 - 非侵入式地捕获数据"""
    
    def __init__(self, original_writer: SummaryWriter):
        self.original_writer = original_writer
        self.collector = get_global_metrics_collector()
        
    def add_scalar(self, tag: str, scalar_value: float, global_step: int = None, walltime: float = None):
        """拦截add_scalar调用，复制数据到采集器"""
        # 调用原始方法
        result = self.original_writer.add_scalar(tag, scalar_value, global_step, walltime)
        
        # 复制数据到采集器
        if self.collector.current_training_session:
            self.collector.collect_scalar_metric(tag, scalar_value, global_step or 0)
            
        return result
        
    def __getattr__(self, name):
        """代理其他方法到原始writer"""
        return getattr(self.original_writer, name)


def install_tensorboard_interceptor(writer: SummaryWriter) -> TensorBoardInterceptor:
    """安装TensorBoard拦截器"""
    return TensorBoardInterceptor(writer)


# 使用示例
if __name__ == "__main__":
    # 创建采集器
    collector = RealTimeMetricsCollector()
    
    # 开始采集
    collector.start_collection("test_session")
    
    # 模拟训练数据
    for epoch in range(5):
        for phase in ['train', 'val']:
            metrics = {
                'loss': 0.5 - epoch * 0.1 + (0.1 if phase == 'val' else 0),
                'accuracy': 0.7 + epoch * 0.05 - (0.05 if phase == 'val' else 0)
            }
            collector.collect_tensorboard_metrics(epoch, phase, metrics)
            time.sleep(0.1)
    
    # 获取AI分析数据
    ai_data = collector.get_current_training_data_for_ai()
    print("AI分析数据:")
    print(json.dumps(ai_data, indent=2, ensure_ascii=False))
    
    # 获取格式化分析文本
    analysis_text = collector.get_formatted_metrics_for_analysis()
    print("\n格式化分析文本:")
    print(analysis_text)
    
    # 停止采集
    collector.stop_collection() 
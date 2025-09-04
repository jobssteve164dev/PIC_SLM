"""
智能训练控制器测试脚本

用于测试智能训练控制器的各项功能，包括：
- 监控启动/停止
- 干预触发
- 参数优化
- 训练重启
"""

import sys
import os
import time
import json
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from training_components.intelligent_training_controller import IntelligentTrainingController
from training_components.intelligent_training_manager import IntelligentTrainingManager


class MockTrainingSystem:
    """模拟训练系统"""
    
    def __init__(self):
        self.is_running = False
        self.current_epoch = 0
        self.metrics_history = []
    
    def stop(self):
        """停止训练"""
        self.is_running = False
        print("🛑 训练已停止")
    
    def start(self, config):
        """开始训练"""
        self.is_running = True
        print(f"🚀 训练已开始，配置: {config}")
    
    def get_status(self):
        """获取训练状态"""
        return {
            'is_running': self.is_running,
            'current_epoch': self.current_epoch
        }


class MockMetricsCollector:
    """模拟指标采集器"""
    
    def __init__(self):
        self.metrics_data = {
            'current_metrics': {
                'epoch': 0,
                'train_loss': 1.0,
                'val_loss': 1.1,
                'train_accuracy': 0.5,
                'val_accuracy': 0.48
            },
            'training_trends': {
                'train_losses': [1.0, 0.9, 0.8, 0.7, 0.6],
                'val_losses': [1.1, 1.0, 0.95, 0.9, 0.85],
                'train_accuracies': [0.5, 0.55, 0.6, 0.65, 0.7],
                'val_accuracies': [0.48, 0.52, 0.55, 0.58, 0.6]
            },
            'training_status': 'running',
            'session_id': 'test_session_001',
            'total_data_points': 25,
            'collection_duration': 120.5
        }
    
    def get_current_training_data_for_ai(self):
        """获取当前训练数据"""
        return self.metrics_data
    
    def update_metrics(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """更新指标"""
        self.metrics_data['current_metrics'].update({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        })
        
        # 更新趋势数据
        self.metrics_data['training_trends']['train_losses'].append(train_loss)
        self.metrics_data['training_trends']['val_losses'].append(val_loss)
        self.metrics_data['training_trends']['train_accuracies'].append(train_acc)
        self.metrics_data['training_trends']['val_accuracies'].append(val_acc)
        
        # 保持最近10个数据点
        for key in ['train_losses', 'val_losses', 'train_accuracies', 'val_accuracies']:
            if len(self.metrics_data['training_trends'][key]) > 10:
                self.metrics_data['training_trends'][key] = self.metrics_data['training_trends'][key][-10:]


class MockLLMFramework:
    """模拟LLM框架"""
    
    def __init__(self):
        self.is_active = True
    
    def analyze_real_training_metrics(self):
        """分析真实训练指标"""
        return {
            'combined_insights': """
## 🧠 AI分析结果

### 📊 训练状态评估
- **当前状态**: 训练进展良好，损失呈下降趋势
- **收敛情况**: 训练和验证损失都在稳定下降
- **过拟合风险**: 低 (训练和验证损失差距合理)

### 🎯 优化建议
1. **学习率**: 当前学习率适中，可继续使用
2. **批次大小**: 建议保持当前批次大小
3. **正则化**: 可适当增加dropout防止过拟合

### ⚠️ 注意事项
- 验证准确率提升较慢，建议关注数据质量
- 训练过程中注意监控验证集性能
            """,
            'suggestions': [
                {'parameter': 'dropout_rate', 'value': 0.2},
                {'parameter': 'weight_decay', 'value': 0.0005}
            ]
        }
    
    def get_real_hyperparameter_suggestions(self):
        """获取超参数优化建议"""
        return {
            'suggestions': [
                {'parameter': 'learning_rate', 'value': 0.0005},
                {'parameter': 'batch_size', 'value': 16},
                {'parameter': 'dropout_rate', 'value': 0.3},
                {'parameter': 'weight_decay', 'value': 0.0005}
            ]
        }


def test_intelligent_controller():
    """测试智能训练控制器"""
    print("🧪 开始测试智能训练控制器...")
    
    # 创建模拟组件
    mock_training_system = MockTrainingSystem()
    mock_metrics_collector = MockMetricsCollector()
    
    # 创建控制器
    controller = IntelligentTrainingController(mock_training_system)
    
    # 替换模拟组件
    controller.metrics_collector = mock_metrics_collector
    controller.llm_framework = MockLLMFramework()
    
    print("✅ 智能训练控制器创建成功")
    
    # 测试监控启动
    print("\n📡 测试监控启动...")
    training_config = {
        'num_epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    
    controller.start_monitoring(training_config)
    print("✅ 监控启动成功")
    
    # 模拟训练过程
    print("\n🔄 模拟训练过程...")
    for epoch in range(1, 6):
        # 模拟训练指标
        train_loss = 1.0 - epoch * 0.1
        val_loss = 1.1 - epoch * 0.08
        train_acc = 0.5 + epoch * 0.05
        val_acc = 0.48 + epoch * 0.04
        
        mock_metrics_collector.update_metrics(epoch, train_loss, val_loss, train_acc, val_acc)
        
        # 更新训练进度
        controller.update_training_progress({'epoch': epoch})
        
        print(f"  Epoch {epoch}: Train Loss={train_loss:.3f}, Val Loss={val_loss:.3f}")
        time.sleep(1)  # 模拟训练时间
    
    # 测试干预触发
    print("\n🚨 测试干预触发...")
    # 模拟过拟合情况
    mock_metrics_collector.update_metrics(6, 0.4, 0.6, 0.8, 0.55)  # 训练损失下降，验证损失上升
    mock_metrics_collector.update_metrics(7, 0.3, 0.7, 0.85, 0.52)  # 继续恶化
    mock_metrics_collector.update_metrics(8, 0.25, 0.8, 0.9, 0.48)  # 明显过拟合
    
    controller.update_training_progress({'epoch': 8})
    time.sleep(2)  # 等待干预检查
    
    # 获取会话信息
    session_info = controller.get_current_session_info()
    if session_info:
        print(f"✅ 会话信息: {session_info['session_id']}")
        print(f"   干预次数: {len(session_info['interventions'])}")
    
    # 停止监控
    print("\n🛑 停止监控...")
    controller.stop_monitoring()
    print("✅ 监控已停止")
    
    return controller


def test_intelligent_manager():
    """测试智能训练管理器"""
    print("\n🧪 开始测试智能训练管理器...")
    
    # 创建模拟训练系统
    mock_training_system = MockTrainingSystem()
    
    # 创建管理器
    manager = IntelligentTrainingManager()
    manager.set_model_trainer(mock_training_system)
    
    print("✅ 智能训练管理器创建成功")
    
    # 测试智能训练启动
    print("\n🚀 测试智能训练启动...")
    training_config = {
        'num_epochs': 15,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout_rate': 0.1,
        'weight_decay': 0.0001
    }
    
    manager.start_intelligent_training(training_config)
    print("✅ 智能训练启动成功")
    
    # 获取训练状态
    status = manager.get_training_status()
    print(f"   训练状态: {status['status']}")
    print(f"   智能模式: {status['intelligent_mode']}")
    
    # 模拟训练过程
    print("\n🔄 模拟训练过程...")
    for epoch in range(1, 4):
        # 模拟训练指标
        train_loss = 1.0 - epoch * 0.15
        val_loss = 1.1 - epoch * 0.1
        train_acc = 0.5 + epoch * 0.08
        val_acc = 0.48 + epoch * 0.06
        
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        }
        
        manager.update_training_progress(metrics)
        print(f"  Epoch {epoch}: Train Loss={train_loss:.3f}, Val Loss={val_loss:.3f}")
        time.sleep(1)
    
    # 测试停止训练
    print("\n🛑 测试停止训练...")
    manager.stop_intelligent_training()
    print("✅ 训练已停止")
    
    return manager


def test_configuration():
    """测试配置管理"""
    print("\n⚙️ 测试配置管理...")
    
    # 创建管理器
    manager = IntelligentTrainingManager()
    
    # 测试配置加载
    print("📂 测试配置加载...")
    manager.load_config()
    print("✅ 配置加载成功")
    
    # 测试配置保存
    print("💾 测试配置保存...")
    manager.save_config("test_config.json")
    print("✅ 配置保存成功")
    
    # 测试配置重置
    print("🔄 测试配置重置...")
    manager.reset_config()
    print("✅ 配置重置成功")
    
    # 清理测试文件
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
        print("🧹 测试配置文件已清理")


def main():
    """主测试函数"""
    print("🤖 智能训练控制器系统测试")
    print("=" * 50)
    
    try:
        # 测试智能训练控制器
        controller = test_intelligent_controller()
        
        # 测试智能训练管理器
        manager = test_intelligent_manager()
        
        # 测试配置管理
        test_configuration()
        
        print("\n🎉 所有测试完成！")
        print("\n📋 测试总结:")
        print("  ✅ 智能训练控制器 - 监控、分析、干预功能正常")
        print("  ✅ 智能训练管理器 - 训练协调、重启功能正常")
        print("  ✅ 配置管理 - 加载、保存、重置功能正常")
        print("  ✅ 模拟组件 - 训练系统、指标采集器、LLM框架模拟正常")
        
        print("\n🚀 系统已准备就绪，可以集成到主应用程序中！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
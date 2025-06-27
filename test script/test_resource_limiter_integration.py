#!/usr/bin/env python3
"""
测试资源限制器与训练系统的集成
验证配置传递和功能是否正常工作
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from src.utils.resource_limiter import ResourceLimits, initialize_resource_limiter
from src.training_components.resource_limited_trainer import ResourceLimitedTrainer


def test_resource_limiter_config():
    """测试资源限制器配置"""
    print("🧪 测试资源限制器配置...")
    
    # 创建资源限制配置
    limits = ResourceLimits(
        max_memory_gb=2.0,  # 2GB内存限制
        max_cpu_percent=50.0,  # 50% CPU限制
        max_disk_usage_gb=1.0,  # 1GB磁盘限制
        max_processes=2,
        max_threads=4,
        enforce_limits=True,
        auto_cleanup=True
    )
    
    # 初始化资源限制器
    resource_limiter = initialize_resource_limiter(limits)
    
    if resource_limiter:
        print("✅ 资源限制器初始化成功")
        
        # 获取当前状态
        status = resource_limiter.get_resource_status()
        print(f"📊 当前资源状态: {status}")
        
        # 测试资源检查
        try:
            resource_limiter.check_resource_before_operation("测试操作")
            print("✅ 资源检查通过")
        except Exception as e:
            print(f"⚠️ 资源检查失败: {e}")
        
        # 停止监控
        resource_limiter.stop_monitoring()
        print("🔚 资源限制器已停止")
    else:
        print("❌ 资源限制器初始化失败")


def test_training_thread_config():
    """测试训练线程配置"""
    print("\n🧪 测试训练线程配置...")
    
    # 模拟训练配置
    config = {
        'enable_resource_limits': True,  # 从训练界面启用
        'resource_limits': {
            'enforce_limits_enabled': False,  # 设置界面未启用
            'memory_absolute_limit_gb': 4.0,
            'cpu_percent_limit': 70.0,
            'temp_files_limit_gb': 2.0,
            'cpu_cores_limit': 6,
            'check_interval': 1.0,
            'auto_cleanup_enabled': True
        },
        'model_name': 'MobileNetV2',
        'num_epochs': 5,
        'batch_size': 16,
        'learning_rate': 0.001
    }
    
    # 模拟训练线程初始化
    try:
        from src.training_components.training_thread import TrainingThread
        
        # 创建训练线程实例（不启动）
        training_thread = TrainingThread(config)
        
        # 检查资源限制器是否正确初始化
        if training_thread.resource_limiter:
            print("✅ 训练线程中资源限制器初始化成功")
            print(f"📊 资源限制器状态: {training_thread.resource_limiter.get_resource_status()}")
        else:
            print("❌ 训练线程中资源限制器初始化失败")
            
    except Exception as e:
        print(f"❌ 训练线程测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_resource_limited_trainer():
    """测试资源限制训练器"""
    print("\n🧪 测试资源限制训练器...")
    
    try:
        # 创建一个简单的假训练器
        class MockTrainer:
            def train_epoch(self, epoch, train_loader, model, optimizer, criterion, device):
                print(f"  Mock training epoch {epoch}")
                return {'loss': 0.5, 'accuracy': 85.0, 'correct': 85, 'total': 100}
            
            def validate_model(self, val_loader, model, criterion, device):
                print(f"  Mock validation")
                return {'val_loss': 0.3, 'val_accuracy': 90.0, 'correct': 90, 'total': 100}
        
        # 创建资源限制的训练器
        mock_trainer = MockTrainer()
        resource_trainer = ResourceLimitedTrainer(mock_trainer)
        
        print("✅ 资源限制训练器创建成功")
        
        # 测试单个epoch训练（模拟）
        try:
            result = resource_trainer.train_epoch(
                epoch=1,
                train_loader=None,  # 模拟
                model=None,
                optimizer=None,
                criterion=None,
                device='cpu'
            )
            
            if result:
                print("✅ 资源限制训练epoch测试通过")
            else:
                print("⚠️ 训练被中断（可能是资源限制）")
                
        except Exception as e:
            print(f"❌ 资源限制训练测试失败: {e}")
            
    except Exception as e:
        print(f"❌ 资源限制训练器测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_config_scenarios():
    """测试各种配置场景"""
    print("\n🧪 测试各种配置场景...")
    
    scenarios = [
        {
            'name': '仅训练界面启用',
            'config': {
                'enable_resource_limits': True,
                'resource_limits': {'enforce_limits_enabled': False}
            }
        },
        {
            'name': '仅设置界面启用',
            'config': {
                'enable_resource_limits': False,
                'resource_limits': {'enforce_limits_enabled': True}
            }
        },
        {
            'name': '两者都启用',
            'config': {
                'enable_resource_limits': True,
                'resource_limits': {'enforce_limits_enabled': True}
            }
        },
        {
            'name': '两者都未启用',
            'config': {
                'enable_resource_limits': False,
                'resource_limits': {'enforce_limits_enabled': False}
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n  📝 场景: {scenario['name']}")
        
        # 模拟配置检查逻辑
        config = scenario['config']
        resource_limits_config = config.get('resource_limits', {})
        enable_from_ui = config.get('enable_resource_limits', False)
        enable_from_settings = resource_limits_config.get('enforce_limits_enabled', False)
        
        if enable_from_ui or enable_from_settings:
            source = "训练界面" if enable_from_ui else "设置界面"
            print(f"    ✅ 会启用资源限制 (来源: {source})")
        else:
            print(f"    ❌ 不会启用资源限制")


if __name__ == "__main__":
    print("🚀 开始测试资源限制器集成...")
    
    # 检查依赖
    try:
        import psutil
        print("✅ psutil 可用")
    except ImportError:
        print("❌ psutil 不可用，某些功能可能无法正常工作")
    
    try:
        import win32job
        print("✅ win32job 可用 (Windows)")
    except ImportError:
        print("ℹ️ win32job 不可用 (可能不是Windows系统)")
    
    # 运行测试
    test_resource_limiter_config()
    test_training_thread_config()
    test_resource_limited_trainer()
    test_config_scenarios()
    
    print("\n🎉 测试完成！") 
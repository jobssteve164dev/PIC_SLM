#!/usr/bin/env python3
"""
测试智能训练配置修复
验证界面设置的参数能够正确应用到训练编排器中
"""

import json
import os
import sys
import time

# 添加项目路径
sys.path.append('src')

def test_config_loading():
    """测试配置加载"""
    print("🔍 测试配置加载...")
    
    # 测试编排器配置加载
    from training_components.intelligent_training_orchestrator import IntelligentTrainingOrchestrator
    
    orchestrator = IntelligentTrainingOrchestrator()
    
    print(f"编排器配置:")
    print(f"  max_iterations: {orchestrator.config.get('max_iterations')}")
    print(f"  min_iteration_epochs: {orchestrator.config.get('min_iteration_epochs')}")
    print(f"  analysis_interval: {orchestrator.config.get('analysis_interval')}")
    
    return orchestrator

def test_config_update():
    """测试配置更新"""
    print("\n🔄 测试配置更新...")
    
    orchestrator = IntelligentTrainingOrchestrator()
    
    # 模拟界面设置的配置
    test_config = {
        'max_iterations': 8,
        'min_iteration_epochs': 3,
        'analysis_interval': 4,
        'convergence_threshold': 0.02,
        'improvement_threshold': 0.03
    }
    
    print(f"更新前配置: {orchestrator.config}")
    
    # 测试update_config方法
    orchestrator.update_config(test_config)
    
    print(f"更新后配置: {orchestrator.config}")
    
    # 验证配置是否更新成功
    success = True
    for key, value in test_config.items():
        if orchestrator.config.get(key) != value:
            print(f"❌ 配置更新失败: {key} = {orchestrator.config.get(key)}, 期望: {value}")
            success = False
    
    if success:
        print("✅ 配置更新测试通过")
    else:
        print("❌ 配置更新测试失败")
    
    return success

def test_update_training_config():
    """测试update_training_config方法"""
    print("\n🔧 测试update_training_config方法...")
    
    orchestrator = IntelligentTrainingOrchestrator()
    
    # 模拟训练配置
    training_config = {
        'data_dir': '/test/data',
        'model_name': 'test_model',
        'max_iterations': 10,
        'min_iteration_epochs': 5,
        'analysis_interval': 3,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    
    print(f"训练配置: {training_config}")
    
    # 测试update_training_config方法
    orchestrator.update_training_config(training_config)
    
    print(f"编排器配置: {orchestrator.config}")
    
    # 验证智能训练相关配置是否更新
    expected_values = {
        'max_iterations': 10,
        'min_iteration_epochs': 5,
        'analysis_interval': 3
    }
    
    success = True
    for key, value in expected_values.items():
        if orchestrator.config.get(key) != value:
            print(f"❌ update_training_config失败: {key} = {orchestrator.config.get(key)}, 期望: {value}")
            success = False
    
    if success:
        print("✅ update_training_config测试通过")
    else:
        print("❌ update_training_config测试失败")
    
    return success

def test_config_file_sync():
    """测试配置文件同步"""
    print("\n📁 测试配置文件同步...")
    
    # 检查主配置文件
    main_config_file = "config.json"
    intelligent_config_file = "setting/intelligent_training_config.json"
    
    if os.path.exists(main_config_file):
        with open(main_config_file, 'r', encoding='utf-8') as f:
            main_config = json.load(f)
        
        intelligent_config = main_config.get('intelligent_training', {})
        print(f"主配置文件中的智能训练配置:")
        print(f"  max_iterations: {intelligent_config.get('max_iterations')}")
        print(f"  min_iteration_epochs: {intelligent_config.get('min_iteration_epochs')}")
        print(f"  analysis_interval: {intelligent_config.get('analysis_interval')}")
    else:
        print("❌ 主配置文件不存在")
        return False
    
    if os.path.exists(intelligent_config_file):
        with open(intelligent_config_file, 'r', encoding='utf-8') as f:
            intelligent_config = json.load(f)
        
        print(f"智能训练专用配置文件:")
        print(f"  max_iterations: {intelligent_config.get('max_iterations')}")
        print(f"  min_iteration_epochs: {intelligent_config.get('min_iteration_epochs')}")
        print(f"  analysis_interval: {intelligent_config.get('analysis_interval')}")
    else:
        print("❌ 智能训练专用配置文件不存在")
        return False
    
    print("✅ 配置文件同步测试完成")
    return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 智能训练配置修复测试")
    print("=" * 60)
    
    try:
        # 测试配置加载
        orchestrator = test_config_loading()
        
        # 测试配置更新
        config_update_success = test_config_update()
        
        # 测试update_training_config方法
        training_config_success = test_update_training_config()
        
        # 测试配置文件同步
        config_sync_success = test_config_file_sync()
        
        print("\n" + "=" * 60)
        print("📊 测试结果汇总")
        print("=" * 60)
        print(f"配置加载: ✅")
        print(f"配置更新: {'✅' if config_update_success else '❌'}")
        print(f"update_training_config方法: {'✅' if training_config_success else '❌'}")
        print(f"配置文件同步: {'✅' if config_sync_success else '❌'}")
        
        if config_update_success and training_config_success and config_sync_success:
            print("\n🎉 所有测试通过！智能训练配置修复成功！")
            return True
        else:
            print("\n❌ 部分测试失败，需要进一步检查")
            return False
            
    except Exception as e:
        print(f"\n💥 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


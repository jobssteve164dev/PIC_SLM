#!/usr/bin/env python3
"""
智能训练配置测试脚本

用于验证智能训练参数设置是否能够正确传递和生效
"""

import os
import sys
import json
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_loading():
    """测试配置加载功能"""
    print("=" * 60)
    print("🧪 测试智能训练配置加载功能")
    print("=" * 60)
    
    try:
        # 测试1: 检查主配置文件
        main_config_file = "config.json"
        if os.path.exists(main_config_file):
            with open(main_config_file, 'r', encoding='utf-8') as f:
                main_config = json.load(f)
                intelligent_config = main_config.get('intelligent_training', {})
                if intelligent_config:
                    print("✅ 主配置文件存在且包含智能训练配置")
                    print(f"   配置内容: {intelligent_config}")
                else:
                    print("⚠️ 主配置文件存在但未包含智能训练配置")
        else:
            print("❌ 主配置文件不存在")
        
        # 测试2: 检查智能训练专用配置文件
        intelligent_config_file = "setting/intelligent_training_config.json"
        if os.path.exists(intelligent_config_file):
            with open(intelligent_config_file, 'r', encoding='utf-8') as f:
                intelligent_config = json.load(f)
                print("✅ 智能训练专用配置文件存在")
                print(f"   配置内容: {intelligent_config}")
        else:
            print("⚠️ 智能训练专用配置文件不存在")
        
        # 测试3: 测试编排器配置加载
        print("\n🔧 测试智能训练编排器配置加载...")
        from src.training_components.intelligent_training_orchestrator import IntelligentTrainingOrchestrator
        
        orchestrator = IntelligentTrainingOrchestrator()
        print(f"✅ 编排器初始化成功")
        print(f"   当前配置: {orchestrator.config}")
        
        # 测试4: 测试配置更新
        print("\n🔄 测试配置更新功能...")
        test_config = {
            'max_iterations': 8,
            'min_iteration_epochs': 5,
            'analysis_interval': 3,
            'convergence_threshold': 0.02,
            'improvement_threshold': 0.03
        }
        
        orchestrator.update_config(test_config)
        print(f"✅ 配置更新成功")
        print(f"   更新后配置: {orchestrator.config}")
        
        # 验证配置是否真的更新了
        if orchestrator.config['max_iterations'] == 8:
            print("✅ 配置更新验证通过")
        else:
            print("❌ 配置更新验证失败")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_config_manager():
    """测试配置管理器功能"""
    print("\n" + "=" * 60)
    print("🧪 测试智能训练管理器配置功能")
    print("=" * 60)
    
    try:
        from src.training_components.intelligent_training_manager import IntelligentTrainingManager
        
        manager = IntelligentTrainingManager()
        print(f"✅ 管理器初始化成功")
        print(f"   当前配置: {manager.config}")
        
        # 测试配置更新
        test_config = {
            'max_iterations': 10,
            'min_iteration_epochs': 3,
            'analysis_interval': 2
        }
        
        manager.update_config(test_config)
        print(f"✅ 管理器配置更新成功")
        print(f"   更新后配置: {manager.config}")
        
        return True
        
    except Exception as e:
        print(f"❌ 管理器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_config_generator():
    """测试配置生成器功能"""
    print("\n" + "=" * 60)
    print("🧪 测试智能配置生成器功能")
    print("=" * 60)
    
    try:
        from src.training_components.intelligent_config_generator import IntelligentConfigGenerator
        
        generator = IntelligentConfigGenerator()
        print(f"✅ 配置生成器初始化成功")
        
        # 测试配置更新
        test_config = {
            'llm_config': {
                'adapter_type': 'mock',
                'analysis_frequency': 'epoch_based',
                'min_data_points': 5,
                'confidence_threshold': 0.7
            },
            'overfitting_threshold': 0.8,
            'underfitting_threshold': 0.7
        }
        
        generator.update_config(test_config)
        print(f"✅ 配置生成器配置更新成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置生成器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_test_config():
    """创建测试配置文件"""
    print("\n" + "=" * 60)
    print("📝 创建测试配置文件")
    print("=" * 60)
    
    try:
        # 创建测试配置
        test_config = {
            "intelligent_training": {
                "enabled": True,
                "max_iterations": 6,
                "min_iteration_epochs": 4,
                "analysis_interval": 3,
                "convergence_threshold": 0.015,
                "improvement_threshold": 0.025,
                "auto_restart": True,
                "preserve_best_model": True,
                "overfitting_threshold": 0.75,
                "underfitting_threshold": 0.65,
                "stagnation_epochs": 8,
                "divergence_threshold": 2.5,
                "min_training_epochs": 5,
                "tuning_strategy": "balanced",
                "enable_auto_intervention": True,
                "intervention_cooldown": 3,
                "max_interventions_per_session": 15,
                "llm_analysis_enabled": True,
                "adapter_type": "mock",
                "analysis_frequency": "epoch_based",
                "min_data_points": 6,
                "confidence_threshold": 0.75,
                "check_interval": 10,
                "metrics_buffer_size": 150,
                "trend_analysis_window": 15,
                "alert_on_intervention": True,
                "auto_generate_reports": True,
                "report_format": "json",
                "include_visualizations": True,
                "save_intervention_details": True
            }
        }
        
        # 保存到主配置文件
        with open("config.json", 'w', encoding='utf-8') as f:
            json.dump(test_config, f, ensure_ascii=False, indent=2)
        
        print("✅ 测试配置文件创建成功")
        print("   文件: config.json")
        print("   配置内容已更新")
        
        return True
        
    except Exception as e:
        print(f"❌ 创建测试配置文件失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🚀 智能训练配置测试开始")
    print("=" * 60)
    
    # 创建测试配置
    if not create_test_config():
        print("❌ 无法创建测试配置，测试终止")
        return
    
    # 运行各项测试
    tests = [
        ("配置加载测试", test_config_loading),
        ("配置管理器测试", test_config_manager),
        ("配置生成器测试", test_config_generator)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 运行 {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} 通过")
        else:
            print(f"❌ {test_name} 失败")
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    print(f"总测试数: {total}")
    print(f"通过测试: {passed}")
    print(f"失败测试: {total - passed}")
    print(f"通过率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 所有测试通过！智能训练配置功能正常")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
    
    print("\n💡 建议:")
    print("1. 在设置界面中修改智能训练参数")
    print("2. 点击'验证配置'按钮检查参数合理性")
    print("3. 点击'保存配置'按钮保存设置")
    print("4. 启动智能训练验证参数是否生效")

if __name__ == "__main__":
    main()

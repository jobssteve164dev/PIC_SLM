#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强AI助手功能测试脚本
测试训练配置文件集成功能
"""

import sys
import os
import json
import time

# 添加src目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def test_enhanced_analysis_engine():
    """测试增强的分析引擎"""
    print("🧠 测试增强的分析引擎...")
    
    try:
        from llm.analysis_engine import TrainingAnalysisEngine
        from llm.model_adapters import MockLLMAdapter
        
        # 创建分析引擎
        mock_adapter = MockLLMAdapter()
        engine = TrainingAnalysisEngine(mock_adapter)
        
        print("✅ 分析引擎创建成功")
        
        # 测试训练配置查找功能
        print("\n📋 测试训练配置查找功能...")
        latest_config = engine._find_latest_training_config()
        if latest_config:
            print(f"   找到配置文件: {latest_config['file_path']}")
            print(f"   配置时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_config['timestamp']))}")
            print(f"   模型名称: {latest_config['config'].get('model_name', 'N/A')}")
        else:
            print("   ⚠️ 未找到训练配置文件")
        
        # 测试训练配置上下文生成
        print("\n📝 测试训练配置上下文生成...")
        config_context = engine._get_training_config_context()
        print(f"   配置上下文长度: {len(config_context)} 字符")
        print(f"   配置上下文预览: {config_context[:200]}...")
        
        # 测试增强的分析提示词构建
        print("\n🔧 测试增强的分析提示词构建...")
        test_metrics = {
            'epoch': 10,
            'train_loss': 0.234,
            'val_loss': 0.287,
            'train_accuracy': 0.894,
            'val_accuracy': 0.856
        }
        test_trends = {
            'train_losses': [0.5, 0.4, 0.3, 0.25, 0.234],
            'val_losses': [0.6, 0.5, 0.4, 0.3, 0.287],
            'train_accuracies': [0.7, 0.8, 0.85, 0.88, 0.894],
            'val_accuracies': [0.65, 0.75, 0.8, 0.83, 0.856]
        }
        test_real_data = {
            'session_id': 'test_session_001',
            'collection_duration': 3600.0,
            'total_data_points': 50,
            'training_status': 'training'
        }
        
        enhanced_prompt = engine._build_enhanced_analysis_prompt(test_metrics, test_trends, test_real_data)
        print(f"   增强提示词长度: {len(enhanced_prompt)} 字符")
        print(f"   提示词包含配置信息: {'是' if '训练配置信息' in enhanced_prompt else '否'}")
        print(f"   提示词包含实时数据: {'是' if '实时训练数据' in enhanced_prompt else '否'}")
        
        # 测试完整的真实数据分析
        print("\n📊 测试完整的真实数据分析...")
        analysis_result = engine.analyze_real_training_progress()
        
        if 'error' in analysis_result:
            print(f"   ⚠️ 分析失败: {analysis_result['error']}")
        else:
            print(f"   ✅ 分析成功")
            print(f"   数据来源: {analysis_result.get('data_source', 'N/A')}")
            print(f"   会话ID: {analysis_result.get('session_id', 'N/A')}")
            print(f"   综合分析长度: {len(analysis_result.get('combined_insights', ''))} 字符")
            
            # 检查是否包含配置信息
            combined_insights = analysis_result.get('combined_insights', '')
            has_config_info = '训练配置信息' in combined_insights or '模型架构' in combined_insights
            print(f"   包含配置信息: {'是' if has_config_info else '否'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_adapter_custom_prompt():
    """测试LLM适配器的自定义提示词功能"""
    print("\n🤖 测试LLM适配器自定义提示词功能...")
    
    try:
        from llm.model_adapters import MockLLMAdapter
        
        # 创建模拟适配器
        adapter = MockLLMAdapter()
        
        # 测试标准分析
        print("   测试标准分析...")
        test_metrics = {'epoch': 10, 'train_loss': 0.234}
        standard_result = adapter.analyze_metrics(test_metrics)
        print(f"   标准分析结果长度: {len(standard_result)} 字符")
        
        # 测试自定义提示词
        print("   测试自定义提示词...")
        custom_prompt = """
请基于以下训练配置进行专业分析：

## 训练配置
- 模型: ResNet50
- 学习率: 0.001
- 批次大小: 32
- 优化器: Adam

## 训练指标
- Epoch: 10
- 训练损失: 0.234

请提供针对性的优化建议。
"""
        custom_result = adapter.analyze_metrics(test_metrics, custom_prompt)
        print(f"   自定义分析结果长度: {len(custom_result)} 字符")
        print(f"   自定义分析包含配置信息: {'是' if 'ResNet50' in custom_result else '否'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_chat_functionality():
    """测试增强的聊天功能"""
    print("\n💬 测试增强的聊天功能...")
    
    try:
        from llm.llm_framework import LLMFramework
        
        # 创建LLM框架
        framework = LLMFramework('mock')
        framework.start()
        
        print("✅ LLM框架创建成功")
        
        # 测试聊天功能
        test_question = "当前训练效果如何？有什么优化建议吗？"
        print(f"   测试问题: {test_question}")
        
        chat_result = framework.chat_with_training_context(test_question)
        
        if isinstance(chat_result, dict) and 'error' in chat_result:
            print(f"   ⚠️ 聊天失败: {chat_result['error']}")
        else:
            response = chat_result.get('response', str(chat_result))
            print(f"   ✅ 聊天成功")
            print(f"   响应长度: {len(response)} 字符")
            print(f"   响应包含配置信息: {'是' if '训练配置' in response or '模型架构' in response else '否'}")
        
        framework.stop()
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_test_config_file():
    """创建测试配置文件"""
    print("\n📁 创建测试配置文件...")
    
    try:
        # 确保目录存在
        test_config_dir = "models/params/classification"
        os.makedirs(test_config_dir, exist_ok=True)
        
        # 创建测试配置文件
        test_config = {
            "data_dir": "test_dataset",
            "model_name": "ResNet50",
            "num_epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "model_save_dir": "models/saved_models",
            "task_type": "classification",
            "optimizer": "Adam",
            "use_pretrained": True,
            "metrics": ["accuracy"],
            "use_tensorboard": True,
            "weight_decay": 0.0001,
            "lr_scheduler": "StepLR",
            "use_augmentation": True,
            "early_stopping": True,
            "early_stopping_patience": 10,
            "gradient_clipping": False,
            "mixed_precision": True,
            "dropout_rate": 0.2,
            "activation_function": "ReLU",
            "use_class_weights": True,
            "weight_strategy": "balanced",
            "class_weights": {
                "class1": 1.0,
                "class2": 2.0,
                "class3": 1.5
            },
            "warmup_enabled": True,
            "warmup_steps": 1000,
            "min_lr_enabled": True,
            "min_lr": 1e-6,
            "label_smoothing_enabled": True,
            "label_smoothing": 0.1,
            "model_ema": True,
            "model_ema_decay": 0.9999,
            "model_filename": "ResNet50_test_config",
            "timestamp": time.strftime("%Y%m%d-%H%M%S")
        }
        
        # 保存配置文件
        config_file_path = os.path.join(test_config_dir, f"ResNet50_{time.strftime('%Y%m%d-%H%M%S')}_test_config.json")
        with open(config_file_path, 'w', encoding='utf-8') as f:
            json.dump(test_config, f, ensure_ascii=False, indent=4)
        
        print(f"   ✅ 测试配置文件已创建: {config_file_path}")
        return config_file_path
        
    except Exception as e:
        print(f"   ❌ 创建测试配置文件失败: {str(e)}")
        return None

def main():
    """主测试函数"""
    print("🚀 开始增强AI助手功能测试")
    print("=" * 60)
    
    # 创建测试配置文件
    test_config_file = create_test_config_file()
    
    # 运行各项测试
    tests = [
        ("增强分析引擎", test_enhanced_analysis_engine),
        ("LLM适配器自定义提示词", test_llm_adapter_custom_prompt),
        ("增强聊天功能", test_enhanced_chat_functionality)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed_tests += 1
            print(f"✅ {test_name} 测试通过")
        else:
            print(f"❌ {test_name} 测试失败")
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed_tests}/{total_tests} 通过")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！增强AI助手功能正常工作")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
    
    # 清理测试文件
    if test_config_file and os.path.exists(test_config_file):
        try:
            os.remove(test_config_file)
            print(f"🧹 已清理测试配置文件: {test_config_file}")
        except:
            pass

if __name__ == "__main__":
    main() 
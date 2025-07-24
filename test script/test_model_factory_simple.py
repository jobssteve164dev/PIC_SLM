#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版模型工厂Tab测试脚本
"""

import sys
import os

# 添加src目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def test_llm_framework():
    """测试LLM框架"""
    print("🧠 测试LLM框架...")
    
    try:
        from src.llm.llm_framework import LLMFramework
        from src.llm.model_adapters import create_llm_adapter
        
        # 创建LLM框架使用模拟适配器
        framework = LLMFramework('mock')
        
        # 启动框架
        framework.start()
        
        print("✅ LLM框架导入成功")
        
        # 测试基本功能
        test_metrics = {
            'epoch': 10,
            'train_loss': 0.234,
            'val_loss': 0.287,
            'train_accuracy': 0.894,
            'val_accuracy': 0.856
        }
        
        print("📊 测试训练指标分析...")
        analysis = framework.analyze_training_metrics(test_metrics)
        print(f"   分析结果: {analysis.get('combined_insights', 'N/A')[:100]}...")
        
        print("💡 测试超参数建议...")
        suggestions = framework.get_hyperparameter_suggestions(test_metrics)
        if isinstance(suggestions, dict):
            suggestions_text = suggestions.get('llm_suggestions', str(suggestions))
        else:
            suggestions_text = str(suggestions)
        print(f"   建议结果: {suggestions_text[:100]}...")
        
        print("🔧 测试问题诊断...")
        diagnosis = framework.diagnose_training_problems(test_metrics)
        if isinstance(diagnosis, dict):
            diagnosis_text = diagnosis.get('llm_diagnosis', str(diagnosis))
        else:
            diagnosis_text = str(diagnosis)
        print(f"   诊断结果: {diagnosis_text[:100]}...")
        
        print("📈 测试模型对比...")
        model_results = [
            {'model_name': 'ResNet50', 'accuracy': 0.89, 'val_loss': 0.23},
            {'model_name': 'EfficientNet', 'accuracy': 0.92, 'val_loss': 0.19}
        ]
        comparison = framework.compare_model_results(model_results)
        if isinstance(comparison, dict):
            comparison_text = comparison.get('analysis', str(comparison))
        else:
            comparison_text = str(comparison)
        print(f"   对比结果: {comparison_text[:100]}...")
        
        print("💬 测试对话功能...")
        response = framework.chat_with_training_context("训练状态如何？")
        if isinstance(response, dict):
            response_text = response.get('response', str(response))
        else:
            response_text = str(response)
        print(f"   对话响应: {response_text[:100]}...")
        
        # 获取统计信息
        stats = framework.get_framework_stats()
        print(f"📊 框架统计: 总请求 {stats.get('total_requests', 0)}, 成功率 {stats.get('success_rate', 0):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM框架测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_components():
    """测试UI组件（无GUI）"""
    print("\n🎨 测试UI组件...")
    
    try:
        # 测试模型工厂Tab类的导入
        from src.ui.model_factory_tab import ModelFactoryTab, LLMChatWidget, AnalysisPanelWidget
        print("✅ 模型工厂Tab组件导入成功")
        
        # 测试基类导入
        from src.ui.base_tab import BaseTab
        print("✅ BaseTab基类导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ UI组件测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """测试集成功能"""
    print("\n🔗 测试集成功能...")
    
    try:
        # 测试主窗口导入
        from src.ui.main_window import MainWindow
        print("✅ 主窗口导入成功")
        
        # 检查模型工厂Tab是否已集成
        # 这里我们只能检查导入，不能实际创建GUI
        print("✅ 集成测试通过（静态检查）")
        
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🏭 第三阶段：AI模型工厂 - 功能测试")
    print("=" * 60)
    
    results = []
    
    # 测试LLM框架
    results.append(test_llm_framework())
    
    # 测试UI组件
    results.append(test_ui_components())
    
    # 测试集成功能
    results.append(test_integration())
    
    print("\n" + "=" * 60)
    print("📋 测试结果汇总")
    print("=" * 60)
    
    test_names = ["LLM框架", "UI组件", "集成功能"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{i+1}. {name}: {status}")
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n🎯 总体结果: {success_count}/{total_count} 项测试通过")
    
    if success_count == total_count:
        print("🎉 所有测试通过！第三阶段开发成功！")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
        return False

if __name__ == "__main__":
    main() 
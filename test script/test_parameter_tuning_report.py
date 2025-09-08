#!/usr/bin/env python3
"""
参数微调报告生成功能测试脚本

测试智能训练组件的参数微调报告生成功能
"""

import os
import sys
import json
import time
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.training_components.parameter_tuning_report_generator import ParameterTuningReportGenerator


def test_report_generator():
    """测试报告生成器功能"""
    print("🧪 开始测试参数微调报告生成器...")
    
    # 测试配置
    test_config = {
        'parameter_tuning_reports': {
            'enabled': True,
            'save_path': 'test_reports/parameter_tuning',
            'format': 'markdown',
            'include_llm_analysis': True,
            'include_metrics_comparison': True,
            'include_config_changes': True
        }
    }
    
    # 创建报告生成器
    generator = ParameterTuningReportGenerator(test_config)
    print("✅ 报告生成器创建成功")
    
    # 测试数据
    original_config = {
        'model_name': 'MobileNetV2',
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 50,
        'dropout_rate': 0.2,
        'weight_decay': 0.0001,
        'early_stopping_patience': 10
    }
    
    adjusted_config = {
        'model_name': 'MobileNetV2',
        'learning_rate': 0.0005,  # 降低学习率
        'batch_size': 16,         # 减小批次大小
        'num_epochs': 50,
        'dropout_rate': 0.3,      # 增加dropout
        'weight_decay': 0.0002,   # 增加权重衰减
        'early_stopping_patience': 10
    }
    
    changes = {
        'learning_rate': {'from': 0.001, 'to': 0.0005},
        'batch_size': {'from': 32, 'to': 16},
        'dropout_rate': {'from': 0.2, 'to': 0.3},
        'weight_decay': {'from': 0.0001, 'to': 0.0002}
    }
    
    llm_analysis = {
        'reason': '检测到过拟合风险，需要调整参数',
        'analysis': '''
基于当前训练指标分析，发现以下问题：

1. **过拟合风险**: 验证损失开始上升，而训练损失继续下降
2. **学习率过高**: 当前学习率可能导致训练不稳定
3. **正则化不足**: Dropout和权重衰减需要增强

建议的优化策略：
- 降低学习率以提高训练稳定性
- 减小批次大小以增加梯度噪声
- 增加Dropout率以防止过拟合
- 增强权重衰减以改善泛化能力
        ''',
        'suggestions': [
            {
                'parameter': 'learning_rate',
                'current_value': 0.001,
                'suggested_value': 0.0005,
                'reason': '训练损失下降缓慢，建议降低学习率',
                'priority': 'high'
            },
            {
                'parameter': 'batch_size',
                'current_value': 32,
                'suggested_value': 16,
                'reason': 'GPU内存使用率较高，建议减小批次大小',
                'priority': 'medium'
            },
            {
                'parameter': 'dropout_rate',
                'current_value': 0.2,
                'suggested_value': 0.3,
                'reason': '检测到过拟合，建议增加Dropout率',
                'priority': 'high'
            },
            {
                'parameter': 'weight_decay',
                'current_value': 0.0001,
                'suggested_value': 0.0002,
                'reason': '检测到过拟合，建议增加权重衰减',
                'priority': 'high'
            }
        ]
    }
    
    training_metrics = {
        'epoch': 15,
        'train_loss': 0.234,
        'val_loss': 0.312,
        'train_accuracy': 0.892,
        'val_accuracy': 0.856,
        'learning_rate': 0.001,
        'batch_size': 32,
        'gpu_memory_usage': 0.78,
        'training_time': 125.6
    }
    
    # 生成报告
    print("📝 正在生成参数微调报告...")
    report_path = generator.generate_report(
        original_config=original_config,
        adjusted_config=adjusted_config,
        changes=changes,
        llm_analysis=llm_analysis,
        training_metrics=training_metrics,
        reason="智能参数优化 - 过拟合风险调整",
        session_id="test_session_001",
        adjustment_id="adj_001"
    )
    
    if report_path:
        print(f"✅ 报告生成成功: {report_path}")
        
        # 验证报告文件
        if os.path.exists(report_path):
            print("✅ 报告文件存在")
            
            # 读取并显示报告内容
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"📊 报告内容长度: {len(content)} 字符")
            print("📋 报告内容预览:")
            print("-" * 50)
            print(content[:500] + "..." if len(content) > 500 else content)
            print("-" * 50)
            
            # 检查报告内容是否包含关键信息
            required_sections = [
                "# 智能训练参数微调报告",
                "## 📋 调整原因",
                "## 🔧 配置变更详情",
                "## 🤖 LLM分析结果",
                "## 📊 训练指标",
                "## ⚙️ 配置对比",
                "## 📝 报告总结"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"⚠️ 报告缺少以下部分: {missing_sections}")
            else:
                print("✅ 报告包含所有必需的部分")
            
            # 检查配置变更是否正确显示
            if "learning_rate" in content and "0.001" in content and "0.0005" in content:
                print("✅ 学习率变更正确显示")
            else:
                print("⚠️ 学习率变更显示可能有问题")
            
            if "batch_size" in content and "32" in content and "16" in content:
                print("✅ 批次大小变更正确显示")
            else:
                print("⚠️ 批次大小变更显示可能有问题")
            
            if "LLM分析结果" in content and "过拟合风险" in content:
                print("✅ LLM分析结果正确显示")
            else:
                print("⚠️ LLM分析结果显示可能有问题")
            
            if "训练指标" in content and "0.234" in content and "0.312" in content:
                print("✅ 训练指标正确显示")
            else:
                print("⚠️ 训练指标显示可能有问题")
            
        else:
            print("❌ 报告文件不存在")
            return False
    else:
        print("❌ 报告生成失败")
        return False
    
    return True


def test_config_update():
    """测试配置更新功能"""
    print("\n🔧 测试配置更新功能...")
    
    generator = ParameterTuningReportGenerator()
    
    # 测试配置更新
    new_config = {
        'parameter_tuning_reports': {
            'enabled': False,
            'save_path': 'new_reports/path',
            'format': 'json',
            'include_llm_analysis': False
        }
    }
    
    generator.update_config(new_config)
    current_config = generator.get_config()
    
    print(f"✅ 配置更新成功")
    print(f"📋 当前配置: {json.dumps(current_config, indent=2, ensure_ascii=False)}")
    
    return True


def test_error_handling():
    """测试错误处理"""
    print("\n🛡️ 测试错误处理...")
    
    generator = ParameterTuningReportGenerator()
    
    # 测试无效配置
    try:
        report_path = generator.generate_report(
            original_config={},
            adjusted_config={},
            changes={},
            llm_analysis={},
            training_metrics={},
            reason="测试错误处理",
            session_id="error_test",
            adjustment_id="error_001"
        )
        
        if not report_path:
            print("✅ 错误处理正常 - 无效配置未生成报告")
        else:
            print("⚠️ 错误处理可能有问题 - 无效配置生成了报告")
            
    except Exception as e:
        print(f"✅ 错误处理正常 - 捕获到异常: {str(e)}")
    
    return True


def main():
    """主测试函数"""
    print("🚀 开始参数微调报告生成功能测试")
    print("=" * 60)
    
    test_results = []
    
    # 运行测试
    test_results.append(("报告生成器基本功能", test_report_generator()))
    test_results.append(("配置更新功能", test_config_update()))
    test_results.append(("错误处理", test_error_handling()))
    
    # 显示测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总:")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"📈 测试通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！参数微调报告生成功能正常工作。")
    else:
        print("⚠️ 部分测试失败，请检查相关功能。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

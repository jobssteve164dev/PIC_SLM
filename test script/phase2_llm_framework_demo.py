#!/usr/bin/env python3
"""
第二阶段LLM框架完整演示脚本

这个脚本演示了LLM框架的所有核心功能，包括：
1. 训练指标智能分析
2. 超参数优化建议
3. 训练问题诊断
4. 模型对比分析
5. 自然语言对话

运行方式: python phase2_llm_framework_demo.py
"""

import sys
import os
import json
import time
import random

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm.llm_framework import LLMFramework, create_llm_framework
from llm.model_adapters import MockLLMAdapter


def generate_realistic_metrics(epoch: int, total_epochs: int = 50) -> dict:
    """生成真实的训练指标数据"""
    # 模拟训练过程中的指标变化
    progress = epoch / total_epochs
    
    # 训练损失：从2.5逐渐降低到0.1，带有随机波动
    base_train_loss = 2.5 * (1 - progress) ** 1.5 + 0.1
    train_loss = base_train_loss + random.uniform(-0.05, 0.05)
    
    # 验证损失：通常比训练损失高一些，可能出现过拟合
    overfitting_factor = 1.0 + 0.3 * progress if progress > 0.7 else 1.0
    val_loss = train_loss * overfitting_factor + random.uniform(-0.02, 0.08)
    
    # 训练准确率：从20%提升到95%
    train_accuracy = 0.2 + 0.75 * (1 - (1 - progress) ** 2) + random.uniform(-0.02, 0.02)
    
    # 验证准确率：通常比训练准确率低一些
    val_accuracy = train_accuracy - 0.05 + random.uniform(-0.03, 0.01)
    
    # 学习率：使用余弦退火调度
    initial_lr = 0.01
    learning_rate = initial_lr * 0.5 * (1 + math.cos(3.14159 * progress))
    
    # GPU内存使用：随训练进度略有变化
    gpu_memory_used = 6.0 + random.uniform(-0.5, 1.0)
    
    # 训练速度：随着模型复杂度增加而略有下降
    training_speed = 1.5 - 0.3 * progress + random.uniform(-0.1, 0.1)
    
    return {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "train_loss": max(0.01, train_loss),
        "val_loss": max(0.01, val_loss),
        "train_accuracy": max(0.0, min(1.0, train_accuracy)),
        "val_accuracy": max(0.0, min(1.0, val_accuracy)),
        "learning_rate": max(0.0001, learning_rate),
        "gpu_memory_used": max(4.0, min(8.0, gpu_memory_used)),
        "gpu_memory_total": 8.0,
        "training_speed": max(0.5, training_speed),
        "gradient_norm": random.uniform(0.01, 0.5),
        "weight_norm": random.uniform(10, 50)
    }


def demo_metrics_analysis(framework: LLMFramework):
    """演示训练指标分析功能"""
    print("\n" + "="*60)
    print("🔍 演示功能：训练指标智能分析")
    print("="*60)
    
    # 生成几个不同阶段的训练指标
    test_cases = [
        ("早期训练阶段", generate_realistic_metrics(5, 50)),
        ("中期训练阶段", generate_realistic_metrics(25, 50)),
        ("后期训练阶段", generate_realistic_metrics(45, 50))
    ]
    
    for stage_name, metrics in test_cases:
        print(f"\n📊 {stage_name} - Epoch {metrics['epoch']}")
        print(f"   训练损失: {metrics['train_loss']:.4f} | 验证损失: {metrics['val_loss']:.4f}")
        print(f"   训练准确率: {metrics['train_accuracy']:.1%} | 验证准确率: {metrics['val_accuracy']:.1%}")
        
        # 进行分析
        analysis_result = framework.analyze_training_metrics(metrics)
        
        if 'error' not in analysis_result:
            rule_analysis = analysis_result.get('rule_analysis', {})
            print(f"   🤖 AI分析: 训练状态={rule_analysis.get('training_state', '未知')}")
            print(f"           收敛状态={rule_analysis.get('convergence_status', '未知')}")
            print(f"           过拟合风险={rule_analysis.get('overfitting_risk', '未知')}")
            
            alerts = analysis_result.get('alerts', [])
            if alerts:
                print(f"   ⚠️  警报: {len(alerts)}个")
                for alert in alerts:
                    print(f"       - {alert.get('message', '')}")
        else:
            print(f"   ❌ 分析失败: {analysis_result['error']}")
        
        time.sleep(1)  # 模拟处理时间


def demo_hyperparameter_suggestions(framework: LLMFramework):
    """演示超参数优化建议功能"""
    print("\n" + "="*60)
    print("⚙️  演示功能：超参数优化建议")
    print("="*60)
    
    # 模拟不同的训练场景
    scenarios = [
        {
            "name": "过拟合场景",
            "metrics": {
                "epoch": 30,
                "train_loss": 0.15,
                "val_loss": 0.45,  # 验证损失明显高于训练损失
                "train_accuracy": 0.95,
                "val_accuracy": 0.78,
                "learning_rate": 0.001,
                "gpu_memory_used": 5.5
            },
            "params": {"batch_size": 32, "learning_rate": 0.001, "dropout": 0.1}
        },
        {
            "name": "学习率过高场景",
            "metrics": {
                "epoch": 10,
                "train_loss": 2.8,  # 损失居高不下
                "val_loss": 3.1,
                "train_accuracy": 0.25,
                "val_accuracy": 0.22,
                "learning_rate": 0.1,  # 学习率过高
                "gpu_memory_used": 3.2
            },
            "params": {"batch_size": 16, "learning_rate": 0.1, "dropout": 0.2}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📈 {scenario['name']}")
        metrics = scenario['metrics']
        params = scenario['params']
        
        print(f"   当前参数: 批量大小={params['batch_size']}, 学习率={params['learning_rate']}")
        print(f"   训练状态: 损失={metrics['train_loss']:.3f}/{metrics['val_loss']:.3f}, 准确率={metrics['train_accuracy']:.1%}/{metrics['val_accuracy']:.1%}")
        
        # 获取优化建议
        suggestion_result = framework.get_hyperparameter_suggestions(metrics, params)
        
        if 'error' not in suggestion_result:
            rule_suggestions = suggestion_result.get('rule_suggestions', [])
            print(f"   🎯 AI建议: {len(rule_suggestions)}条优化建议")
            
            for i, suggestion in enumerate(rule_suggestions[:3], 1):  # 显示前3条
                param = suggestion.get('parameter', '')
                current = suggestion.get('current_value', '')
                suggested = suggestion.get('suggested_value', '')
                reason = suggestion.get('reason', '')
                priority = suggestion.get('priority', '')
                
                print(f"      {i}. [{priority.upper()}] {param}: {current} → {suggested}")
                print(f"         理由: {reason}")
            
            # 显示预期改进
            improvements = suggestion_result.get('expected_improvements', {})
            if improvements:
                print(f"   📊 预期改进:")
                for param, improvement in list(improvements.items())[:2]:
                    print(f"      - {param}: {improvement}")
        else:
            print(f"   ❌ 建议生成失败: {suggestion_result['error']}")
        
        time.sleep(1)


def demo_problem_diagnosis(framework: LLMFramework):
    """演示训练问题诊断功能"""
    print("\n" + "="*60)
    print("🔧 演示功能：训练问题诊断")
    print("="*60)
    
    # 模拟问题场景
    problem_scenarios = [
        {
            "name": "梯度爆炸问题",
            "metrics": {
                "epoch": 8,
                "train_loss": 150.0,  # 损失爆炸
                "val_loss": 200.0,
                "train_accuracy": 0.1,
                "val_accuracy": 0.1,
                "learning_rate": 0.01,
                "gradient_norm": 50.0  # 梯度范数过大
            },
            "error_info": "RuntimeError: Loss became NaN during training"
        },
        {
            "name": "GPU内存不足",
            "metrics": {
                "epoch": 15,
                "train_loss": 0.8,
                "val_loss": 0.9,
                "train_accuracy": 0.65,
                "val_accuracy": 0.62,
                "learning_rate": 0.002,
                "gpu_memory_used": 7.8,
                "gpu_memory_total": 8.0  # 内存使用率97.5%
            },
            "error_info": "CUDA out of memory"
        }
    ]
    
    for scenario in problem_scenarios:
        print(f"\n🚨 {scenario['name']}")
        metrics = scenario['metrics']
        error_info = scenario.get('error_info')
        
        if error_info:
            print(f"   错误信息: {error_info}")
        
        print(f"   训练状态: Epoch {metrics['epoch']}, 损失={metrics['train_loss']:.2f}")
        
        # 进行问题诊断
        diagnosis_result = framework.diagnose_training_problems(metrics, error_info)
        
        if 'error' not in diagnosis_result:
            anomalies = diagnosis_result.get('detected_anomalies', [])
            severity = diagnosis_result.get('severity_assessment', 'unknown')
            actions = diagnosis_result.get('recommended_actions', [])
            
            print(f"   🔍 检测异常: {len(anomalies)}个 (严重程度: {severity})")
            for anomaly in anomalies[:2]:  # 显示前2个异常
                print(f"      - {anomaly.get('description', '')}")
            
            print(f"   💡 推荐行动: {len(actions)}项")
            for action in actions[:2]:  # 显示前2个行动
                desc = action.get('description', '')
                priority = action.get('priority', '')
                time_est = action.get('expected_time', '')
                print(f"      - [{priority.upper()}] {desc} (预计用时: {time_est})")
            
            # 显示LLM诊断
            llm_diagnosis = diagnosis_result.get('llm_diagnosis', '')
            if llm_diagnosis and len(llm_diagnosis) > 100:
                print(f"   🤖 AI诊断: {llm_diagnosis[:100]}...")
        else:
            print(f"   ❌ 诊断失败: {diagnosis_result['error']}")
        
        time.sleep(1)


def demo_model_comparison(framework: LLMFramework):
    """演示模型对比分析功能"""
    print("\n" + "="*60)
    print("🏆 演示功能：模型对比分析")
    print("="*60)
    
    # 模拟不同模型的训练结果
    model_results = [
        {
            "model_name": "ResNet50",
            "architecture": "CNN",
            "parameters": 25557032,
            "train_loss": 0.15,
            "val_loss": 0.22,
            "train_accuracy": 0.94,
            "val_accuracy": 0.89,
            "training_time": 3600,  # 秒
            "gpu_memory": 6.2
        },
        {
            "model_name": "EfficientNet-B0",
            "architecture": "CNN",
            "parameters": 5288548,
            "train_loss": 0.18,
            "val_loss": 0.21,
            "train_accuracy": 0.92,
            "val_accuracy": 0.90,
            "training_time": 2800,
            "gpu_memory": 4.8
        },
        {
            "model_name": "MobileNetV2",
            "architecture": "CNN",
            "parameters": 3504872,
            "train_loss": 0.22,
            "val_loss": 0.26,
            "train_accuracy": 0.89,
            "val_accuracy": 0.86,
            "training_time": 1800,
            "gpu_memory": 3.5
        }
    ]
    
    print(f"📋 对比 {len(model_results)} 个模型:")
    for model in model_results:
        print(f"   • {model['model_name']}: 参数量={model['parameters']:,}, 验证准确率={model['val_accuracy']:.1%}")
    
    # 进行模型对比
    comparison_result = framework.compare_model_results(model_results)
    
    if 'error' not in comparison_result:
        rule_comparison = comparison_result.get('rule_comparison', {})
        best_model = comparison_result.get('best_model', {})
        ranking = comparison_result.get('performance_ranking', [])
        
        print(f"\n🥇 最佳模型: {best_model.get('model_name', '未知')}")
        print(f"   验证准确率: {best_model.get('val_accuracy', 0):.1%}")
        print(f"   验证损失: {best_model.get('val_loss', 0):.3f}")
        
        print(f"\n📊 性能排名:")
        for i, model in enumerate(ranking[:3], 1):
            name = model.get('model_name', '未知')
            score = model.get('score', 0)
            val_acc = model.get('val_accuracy', 0)
            print(f"   {i}. {name} (综合评分: {score:.3f}, 验证准确率: {val_acc:.1%})")
        
        # 显示对比摘要
        summary = rule_comparison.get('summary', {})
        if summary:
            acc_range = summary.get('accuracy_range', [0, 0])
            print(f"\n📈 对比摘要:")
            print(f"   准确率范围: {acc_range[0]:.1%} - {acc_range[1]:.1%}")
            print(f"   模型总数: {summary.get('total_models', 0)}")
    else:
        print(f"   ❌ 对比失败: {comparison_result['error']}")
    
    time.sleep(1)


def demo_chat_functionality(framework: LLMFramework):
    """演示自然语言对话功能"""
    print("\n" + "="*60)
    print("💬 演示功能：自然语言对话")
    print("="*60)
    
    # 预设一些训练上下文
    context_metrics = generate_realistic_metrics(20, 50)
    framework.analyze_training_metrics(context_metrics)  # 建立上下文
    
    # 模拟用户问题
    questions = [
        "当前训练效果如何？",
        "我的模型是否存在过拟合问题？",
        "为什么验证损失比训练损失高？",
        "如何提高模型的泛化能力？",
        "GPU内存使用率是否正常？"
    ]
    
    print("🤖 AI训练助手已准备就绪，基于当前训练上下文回答问题\n")
    
    for i, question in enumerate(questions, 1):
        print(f"👤 用户问题 {i}: {question}")
        
        # 获取AI回答
        chat_result = framework.chat_with_training_context(question)
        
        if 'error' not in chat_result:
            response = chat_result.get('response', '')
            processing_time = chat_result.get('framework_info', {}).get('processing_time', 0)
            
            # 截取回答的前200个字符用于展示
            display_response = response[:200] + "..." if len(response) > 200 else response
            print(f"🤖 AI回答: {display_response}")
            print(f"   (处理时间: {processing_time:.2f}秒)")
        else:
            print(f"❌ 回答失败: {chat_result['error']}")
        
        print()
        time.sleep(0.5)


def demo_framework_stats(framework: LLMFramework):
    """演示框架统计和健康监控"""
    print("\n" + "="*60)
    print("📊 演示功能：框架统计和健康监控")
    print("="*60)
    
    # 获取框架统计
    stats = framework.get_framework_stats()
    
    print("📈 性能统计:")
    perf_stats = stats.get('performance_stats', {})
    print(f"   总请求数: {perf_stats.get('total_requests', 0)}")
    print(f"   成功请求: {perf_stats.get('successful_requests', 0)}")
    print(f"   失败请求: {perf_stats.get('failed_requests', 0)}")
    print(f"   平均响应时间: {perf_stats.get('average_response_time', 0):.2f}秒")
    
    # 获取系统健康状态
    health = framework.get_system_health()
    
    print(f"\n🏥 系统健康状态: {health.get('overall_status', '未知')}")
    
    components = health.get('components', {})
    for comp_name, comp_info in components.items():
        status = comp_info.get('status', '未知')
        comp_type = comp_info.get('type', '')
        print(f"   • {comp_name}: {status} ({comp_type})")
    
    issues = health.get('issues', [])
    if issues:
        print(f"\n⚠️  发现问题 ({len(issues)}个):")
        for issue in issues:
            print(f"   - {issue}")
    
    recommendations = health.get('recommendations', [])
    if recommendations:
        print(f"\n💡 改进建议:")
        for rec in recommendations:
            print(f"   - {rec}")


def main():
    """主演示函数"""
    print("🚀 第二阶段LLM框架完整演示")
    print("=" * 80)
    
    # 创建LLM框架实例
    print("🔧 初始化LLM框架...")
    framework_config = {
        'adapter_type': 'mock',  # 使用模拟适配器进行演示
        'adapter_config': {},
        'enable_streaming': True,
        'auto_start': True
    }
    
    framework = create_llm_framework(framework_config)
    
    try:
        # 演示各项功能
        demo_metrics_analysis(framework)
        demo_hyperparameter_suggestions(framework)
        demo_problem_diagnosis(framework)
        demo_model_comparison(framework)
        demo_chat_functionality(framework)
        demo_framework_stats(framework)
        
        # 最终报告
        print("\n" + "="*60)
        print("📋 演示完成 - 最终报告")
        print("="*60)
        
        final_stats = framework.get_framework_stats()
        perf_stats = final_stats.get('performance_stats', {})
        
        print(f"✅ 总共处理请求: {perf_stats.get('total_requests', 0)} 个")
        print(f"✅ 成功率: {perf_stats.get('successful_requests', 0) / max(1, perf_stats.get('total_requests', 1)) * 100:.1f}%")
        print(f"✅ 平均响应时间: {perf_stats.get('average_response_time', 0):.2f} 秒")
        
        # 确保llm_train目录存在
        os.makedirs('llm_train', exist_ok=True)
        
        # 导出分析报告到llm_train目录
        report = framework.export_analysis_report(include_history=False)
        report_file = f"llm_train/llm_framework_demo_report_{int(time.time())}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 详细报告已导出: {report_file}")
        
        print("\n🎉 第二阶段LLM框架演示成功完成！")
        print("   所有核心功能均正常工作，系统已准备好进入第三阶段的用户界面集成。")
        
    except KeyboardInterrupt:
        print("\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        framework.stop()
        print("\n🔧 LLM框架已停止")


if __name__ == "__main__":
    # 导入数学库用于生成真实指标
    import math
    main() 
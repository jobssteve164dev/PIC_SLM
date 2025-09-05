#!/usr/bin/env python3
"""
测试修复后的配置
"""

import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fixed_config():
    """测试修复后的配置"""
    print("🔍 测试修复后的配置...")
    
    # 测试智能训练编排器配置
    try:
        from src.training_components.intelligent_training_orchestrator import IntelligentTrainingOrchestrator
        orchestrator = IntelligentTrainingOrchestrator()
        config = orchestrator.config
        
        print(f"✅ 智能训练编排器配置:")
        print(f"  - 最小迭代轮数: {config.get('min_iteration_epochs', 'N/A')}")
        print(f"  - 分析间隔: {config.get('analysis_interval', 'N/A')}")
        print(f"  - 最大迭代次数: {config.get('max_iterations', 'N/A')}")
        
        # 验证配置是否正确
        if config.get('min_iteration_epochs') == 2 and config.get('analysis_interval') == 2:
            print("✅ 调试配置正确设置")
        else:
            print("❌ 调试配置设置错误")
            
    except Exception as e:
        print(f"❌ 测试智能训练编排器配置失败: {str(e)}")
    
    # 测试LLM配置加载
    try:
        from src.training_components.intelligent_config_generator import IntelligentConfigGenerator
        generator = IntelligentConfigGenerator()
        llm_config = generator._load_llm_config()
        
        print(f"✅ LLM配置:")
        print(f"  - 适配器类型: {llm_config.get('adapter_type', 'N/A')}")
        print(f"  - 适配器配置: {llm_config.get('adapter_config', {})}")
        
        if llm_config.get('adapter_type') == 'deepseek':
            print("✅ DeepSeek配置正确加载")
        else:
            print(f"⚠️ 当前使用适配器: {llm_config.get('adapter_type')}")
            
    except Exception as e:
        print(f"❌ 测试LLM配置失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 配置测试完成")

if __name__ == "__main__":
    test_fixed_config()

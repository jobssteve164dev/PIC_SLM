#!/usr/bin/env python3
"""
测试调试配置
"""

import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_debug_config():
    """测试调试配置"""
    print("🔍 测试调试配置...")
    
    # 测试AI配置文件
    ai_config_file = "setting/ai_config.json"
    if os.path.exists(ai_config_file):
        with open(ai_config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            default_adapter = config.get('general', {}).get('default_adapter', 'N/A')
            print(f"✅ AI配置 - 默认适配器: {default_adapter}")
            
            if default_adapter == 'deepseek':
                deepseek_config = config.get('deepseek', {})
                print(f"✅ DeepSeek配置: {deepseek_config}")
    
    # 测试智能训练配置文件
    intelligent_config_file = "setting/intelligent_training_config.json"
    if os.path.exists(intelligent_config_file):
        with open(intelligent_config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            min_training_epochs = config.get('intervention_thresholds', {}).get('min_training_epochs', 'N/A')
            min_data_points = config.get('llm_analysis', {}).get('min_data_points', 'N/A')
            print(f"✅ 智能训练配置 - 最小训练轮数: {min_training_epochs}")
            print(f"✅ 智能训练配置 - 最小数据点: {min_data_points}")
    
    # 测试配置加载
    try:
        from src.training_components.intelligent_config_generator import IntelligentConfigGenerator
        generator = IntelligentConfigGenerator()
        config = generator._load_llm_config()
        print(f"✅ 最终LLM配置: {config}")
        
        # 测试LLM框架初始化
        from src.llm.llm_framework import LLMFramework
        llm_framework = LLMFramework(
            adapter_type=config['adapter_type'],
            adapter_config=config['adapter_config']
        )
        print(f"✅ LLM框架初始化成功，适配器: {type(llm_framework.llm_adapter).__name__}")
        
    except Exception as e:
        print(f"❌ 配置测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug_config()

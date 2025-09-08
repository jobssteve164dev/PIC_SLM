#!/usr/bin/env python3
"""
测试配置加载
"""

import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_loading():
    """测试配置加载"""
    print("🔍 测试LLM配置加载...")
    
    # 测试AI配置文件
    ai_config_file = "setting/ai_config.json"
    if os.path.exists(ai_config_file):
        print(f"✅ 找到AI配置文件: {ai_config_file}")
        with open(ai_config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            print(f"📋 AI配置内容:")
            print(f"  - 默认适配器: {config.get('general', {}).get('default_adapter', 'N/A')}")
            print(f"  - DeepSeek配置: {config.get('deepseek', {})}")
    else:
        print(f"❌ AI配置文件不存在: {ai_config_file}")
    
    # 测试智能训练配置文件
    intelligent_config_file = "setting/intelligent_training_config.json"
    if os.path.exists(intelligent_config_file):
        print(f"✅ 找到智能训练配置文件: {intelligent_config_file}")
        with open(intelligent_config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            llm_config = config.get('llm_config', {})
            if llm_config:
                print(f"📋 智能训练LLM配置: {llm_config}")
            else:
                print("⚠️ 智能训练配置中未找到llm_config")
    else:
        print(f"❌ 智能训练配置文件不存在: {intelligent_config_file}")
    
    # 测试配置加载逻辑
    try:
        from src.training_components.intelligent_config_generator import IntelligentConfigGenerator
        generator = IntelligentConfigGenerator()
        config = generator._load_llm_config()
        print(f"🎯 最终加载的配置: {config}")
    except Exception as e:
        print(f"❌ 配置加载失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config_loading()


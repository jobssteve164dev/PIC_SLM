#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单的AI配置测试脚本
"""

import json
import os

def test_basic_config():
    """测试基本配置操作"""
    print("🔧 测试AI配置文件基本操作...")
    
    # 测试配置
    config = {
        'openai': {
            'api_key': 'sk-test12345',
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 1000
        },
        'ollama': {
            'base_url': 'http://localhost:11434',
            'model': 'llama2',
            'temperature': 0.7,
            'num_predict': 1000
        },
        'general': {
            'default_adapter': 'openai',
            'request_timeout': 60,
            'max_retries': 3,
            'enable_cache': True,
            'enable_streaming': False
        }
    }
    
    # 确保目录存在
    os.makedirs("setting", exist_ok=True)
    
    # 保存配置
    config_file = "setting/ai_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 配置已保存到: {config_file}")
    
    # 读取验证
    with open(config_file, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    
    print(f"✅ 配置读取成功")
    print(f"   默认适配器: {loaded['general']['default_adapter']}")
    print(f"   OpenAI模型: {loaded['openai']['model']}")
    print(f"   Ollama模型: {loaded['ollama']['model']}")
    
    return True

if __name__ == "__main__":
    try:
        test_basic_config()
        print("🎉 基本配置测试通过！")
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc() 
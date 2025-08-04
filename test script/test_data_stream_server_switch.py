#!/usr/bin/env python3
"""
测试数据流服务器开关功能

验证AI设置中的数据流服务器开关是否能够正确控制训练过程中的数据流服务器启动。
"""

import json
import os
import sys
import time
import tempfile
import shutil

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_stream_server_switch():
    """测试数据流服务器开关功能"""
    print("=== 测试数据流服务器开关功能 ===\n")
    
    # 1. 备份原始配置文件
    config_file = "setting/ai_config.json"
    backup_file = "setting/ai_config_backup.json"
    
    print("1. 备份原始配置文件")
    if os.path.exists(config_file):
        shutil.copy2(config_file, backup_file)
        print(f"✅ 已备份到: {backup_file}")
    else:
        print("❌ 原始配置文件不存在")
        return False
    
    try:
        # 2. 测试启用状态
        print("\n2. 测试启用状态")
        test_enabled_config = {
            "openai": {
                "api_key": "sk-test12345",
                "base_url": "",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1003
            },
            "deepseek": {
                "api_key": "sk-63363d60338d444f9ad3709736609cd2",
                "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat",
                "temperature": 0.7,
                "max_tokens": 3000
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "deepseek-r1:8b",
                "temperature": 0.7,
                "num_predict": 1000,
                "timeout": 120
            },
            "custom_api": {
                "name": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "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",
                "provider_type": "OpenAI兼容",
                "model": "deepseek/deepseek-r1-0528:free",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "general": {
                "default_adapter": "deepseek",
                "request_timeout": 180,
                "max_retries": 3,
                "enable_cache": True,
                "enable_streaming": True,
                "enable_data_stream_server": True
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(test_enabled_config, f, indent=2, ensure_ascii=False)
        
        print("✅ 已设置启用状态")
        
        # 3. 测试禁用状态
        print("\n3. 测试禁用状态")
        test_disabled_config = test_enabled_config.copy()
        test_disabled_config['general']['enable_data_stream_server'] = False
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(test_disabled_config, f, indent=2, ensure_ascii=False)
        
        print("✅ 已设置禁用状态")
        
        # 4. 测试配置加载功能
        print("\n4. 测试配置加载功能")
        from src.training_components.training_thread import TrainingThread
        
        # 创建临时训练配置
        temp_config = {
            'data_dir': 'test_data',
            'model_name': 'test_model',
            'num_epochs': 1,
            'batch_size': 2,
            'learning_rate': 0.001,
            'task_type': 'classification'
        }
        
        # 创建训练线程实例
        training_thread = TrainingThread(temp_config)
        
        # 测试配置加载
        ai_config = training_thread._load_ai_config()
        enable_server = ai_config.get('general', {}).get('enable_data_stream_server', True)
        
        print(f"加载的配置: enable_data_stream_server = {enable_server}")
        
        if enable_server == False:
            print("✅ 禁用状态配置加载正确")
        else:
            print("❌ 禁用状态配置加载错误")
            return False
        
        # 5. 测试启用状态配置加载
        print("\n5. 测试启用状态配置加载")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(test_enabled_config, f, indent=2, ensure_ascii=False)
        
        ai_config = training_thread._load_ai_config()
        enable_server = ai_config.get('general', {}).get('enable_data_stream_server', True)
        
        print(f"加载的配置: enable_data_stream_server = {enable_server}")
        
        if enable_server == True:
            print("✅ 启用状态配置加载正确")
        else:
            print("❌ 启用状态配置加载错误")
            return False
        
        # 6. 测试默认值处理
        print("\n6. 测试默认值处理")
        test_default_config = test_enabled_config.copy()
        del test_default_config['general']['enable_data_stream_server']
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(test_default_config, f, indent=2, ensure_ascii=False)
        
        ai_config = training_thread._load_ai_config()
        enable_server = ai_config.get('general', {}).get('enable_data_stream_server', True)
        
        print(f"加载的配置: enable_data_stream_server = {enable_server}")
        
        if enable_server == True:
            print("✅ 默认值处理正确")
        else:
            print("❌ 默认值处理错误")
            return False
        
        print("\n🎉 所有测试通过！数据流服务器开关功能正常工作。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {str(e)}")
        return False
    
    finally:
        # 7. 恢复原始配置文件
        print("\n7. 恢复原始配置文件")
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, config_file)
            os.remove(backup_file)
            print("✅ 已恢复原始配置文件")
        else:
            print("⚠️ 备份文件不存在，无法恢复")

if __name__ == "__main__":
    success = test_data_stream_server_switch()
    sys.exit(0 if success else 1) 
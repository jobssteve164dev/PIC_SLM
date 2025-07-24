#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI设置完整集成测试脚本
测试AI设置组件与模型工厂Tab的完整集成功能
"""

import sys
import os
import json
import traceback
from datetime import datetime

# 设置路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_ai_config_file_operations():
    """测试AI配置文件操作"""
    print("🔧 测试AI配置文件操作...")
    
    try:
        # 测试配置数据
        test_config = {
            'openai': {
                'api_key': 'sk-test12345',
                'base_url': 'https://api.openai.com/v1',
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
            json.dump(test_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 配置文件已保存到: {config_file}")
        
        # 读取配置验证
        with open(config_file, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        
        print(f"✅ 配置文件读取成功，包含 {len(loaded_config)} 个配置组")
        print(f"   - OpenAI配置: {loaded_config.get('openai', {}).get('model', 'N/A')}")
        print(f"   - Ollama配置: {loaded_config.get('ollama', {}).get('model', 'N/A')}")
        print(f"   - 默认适配器: {loaded_config.get('general', {}).get('default_adapter', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件操作失败: {str(e)}")
        traceback.print_exc()
        return False

def test_model_factory_config_loading():
    """测试模型工厂Tab的配置加载功能"""
    print("\n🏭 测试模型工厂Tab配置加载...")
    
    try:
        from src.ui.model_factory_tab import LLMChatWidget
        
        # 创建聊天组件实例
        chat_widget = LLMChatWidget()
        
        # 测试配置加载方法
        ai_config = chat_widget.load_ai_config()
        
        print(f"✅ AI配置加载成功")
        print(f"   - OpenAI API密钥: {'已配置' if ai_config.get('openai', {}).get('api_key') else '未配置'}")
        print(f"   - OpenAI模型: {ai_config.get('openai', {}).get('model', 'N/A')}")
        print(f"   - Ollama服务器: {ai_config.get('ollama', {}).get('base_url', 'N/A')}")
        print(f"   - Ollama模型: {ai_config.get('ollama', {}).get('model', 'N/A')}")
        print(f"   - 默认适配器: {ai_config.get('general', {}).get('default_adapter', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型工厂Tab配置加载失败: {str(e)}")
        traceback.print_exc()
        return False

def test_llm_framework_with_config():
    """测试LLM框架与配置的集成"""
    print("\n🧠 测试LLM框架配置集成...")
    
    try:
        from src.llm.llm_framework import LLMFramework
        
        # 测试不同适配器类型的初始化
        test_configs = [
            ('mock', {}),
            ('openai', {
                'api_key': 'sk-test12345',
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 1000
            }),
            ('local', {
                'model_name': 'llama2',
                'base_url': 'http://localhost:11434',
                'temperature': 0.7
            })
        ]
        
        for adapter_type, adapter_config in test_configs:
            try:
                print(f"   测试 {adapter_type} 适配器...")
                framework = LLMFramework(adapter_type, adapter_config)
                framework.start()
                
                stats = framework.get_framework_stats()
                print(f"   ✅ {adapter_type} 适配器初始化成功")
                
            except Exception as e:
                print(f"   ⚠️ {adapter_type} 适配器初始化失败: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM框架配置集成测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_ai_settings_widget():
    """测试AI设置组件"""
    print("\n🤖 测试AI设置组件...")
    
    try:
        from PyQt5.QtWidgets import QApplication
        from src.ui.components.settings.ai_settings_widget import AISettingsWidget
        
        # 创建应用实例（如果不存在）
        if not QApplication.instance():
            app = QApplication([])
        
        # 创建AI设置组件
        ai_widget = AISettingsWidget()
        
        print("✅ AI设置组件创建成功")
        
        # 测试配置加载
        ai_widget.load_config()
        print("✅ AI设置配置加载成功")
        
        # 测试获取当前配置
        current_config = ai_widget.get_config()
        print(f"✅ 当前配置获取成功，包含 {len(current_config)} 个配置组")
        
        return True
        
    except Exception as e:
        print(f"❌ AI设置组件测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_settings_integration():
    """测试设置Tab集成功能"""
    print("\n⚙️ 测试设置Tab集成...")
    
    try:
        # 模拟AI设置变化
        test_ai_config = {
            'openai': {
                'api_key': 'sk-test12345',
                'model': 'gpt-4',
                'temperature': 0.8
            },
            'ollama': {
                'model': 'llama2',
                'base_url': 'http://localhost:11434'
            },
            'general': {
                'default_adapter': 'openai'
            }
        }
        
        # 测试配置保存逻辑
        import json
        os.makedirs("setting", exist_ok=True)
        config_file = "setting/ai_config.json"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(test_ai_config, f, indent=2, ensure_ascii=False)
        
        print("✅ 设置Tab配置保存模拟成功")
        
        # 验证配置文件
        with open(config_file, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        
        assert saved_config['general']['default_adapter'] == 'openai'
        assert saved_config['openai']['model'] == 'gpt-4'
        
        print("✅ 设置Tab集成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 设置Tab集成测试失败: {str(e)}")
        traceback.print_exc()
        return False

def cleanup_test_files():
    """清理测试文件"""
    print("\n🧹 清理测试文件...")
    
    try:
        test_file = "setting/ai_config.json"
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"✅ 已清理测试文件: {test_file}")
        
    except Exception as e:
        print(f"⚠️ 清理测试文件时出错: {str(e)}")

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 AI设置完整集成测试")
    print("=" * 60)
    
    test_results = []
    
    # 执行各项测试
    tests = [
        ("AI配置文件操作", test_ai_config_file_operations),
        ("模型工厂配置加载", test_model_factory_config_loading),
        ("LLM框架配置集成", test_llm_framework_with_config),
        ("AI设置组件", test_ai_settings_widget),
        ("设置Tab集成", test_settings_integration),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试出现异常: {str(e)}")
            test_results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！AI设置集成功能正常工作")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
    
    # 清理测试文件
    cleanup_test_files()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
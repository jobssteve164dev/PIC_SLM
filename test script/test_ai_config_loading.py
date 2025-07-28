#!/usr/bin/env python3
"""
测试AI配置加载和初始化过程
"""

import json
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ai_config_loading():
    """测试AI配置加载"""
    print("=== 测试AI配置加载 ===\n")
    
    # 1. 检查配置文件是否存在
    config_file = "setting/ai_config.json"
    print(f"1. 检查配置文件: {config_file}")
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    print(f"✅ 配置文件存在")
    
    # 2. 读取配置文件
    print(f"\n2. 读取配置文件")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✅ 配置文件读取成功")
    except Exception as e:
        print(f"❌ 配置文件读取失败: {str(e)}")
        return False
    
    # 3. 检查配置结构
    print(f"\n3. 检查配置结构")
    
    # 检查general部分
    general_config = config.get('general', {})
    default_adapter = general_config.get('default_adapter', 'mock')
    print(f"默认适配器: {default_adapter}")
    
    if default_adapter != 'custom':
        print(f"❌ 默认适配器不是'custom'，而是'{default_adapter}'")
        return False
    
    print("✅ 默认适配器设置正确")
    
    # 4. 检查custom_api配置
    print(f"\n4. 检查custom_api配置")
    custom_config = config.get('custom_api', {})
    
    api_key = custom_config.get('api_key', '')
    base_url = custom_config.get('base_url', '')
    model = custom_config.get('model', '')
    name = custom_config.get('name', '')
    
    print(f"API密钥: {'已设置' if api_key else '未设置'}")
    print(f"Base URL: {base_url}")
    print(f"模型: {model}")
    print(f"名称: {name}")
    
    if not api_key:
        print("❌ API密钥未设置")
        return False
    
    if not base_url:
        print("❌ Base URL未设置")
        return False
    
    if not model:
        print("❌ 模型未设置")
        return False
    
    print("✅ custom_api配置完整")
    
    # 5. 模拟初始化逻辑
    print(f"\n5. 模拟初始化逻辑")
    
    # 模拟init_llm_framework中的逻辑
    if default_adapter == 'custom':
        if api_key and base_url:
            print("✅ 配置检查通过，应该使用自定义API")
            print(f"   适配器类型: custom")
            print(f"   模型: {model}")
            print(f"   API地址: {base_url}")
            print(f"   API名称: {name}")
        else:
            print("❌ 配置检查失败，会回退到模拟适配器")
            print(f"   API密钥存在: {bool(api_key)}")
            print(f"   Base URL存在: {bool(base_url)}")
            return False
    else:
        print(f"❌ 默认适配器不是custom，而是{default_adapter}")
        return False
    
    return True

def test_llm_framework_creation():
    """测试LLM框架创建"""
    print(f"\n=== 测试LLM框架创建 ===\n")
    
    try:
        # 导入LLM框架
        from src.llm.llm_framework import LLMFramework
        
        # 读取配置
        with open("setting/ai_config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        custom_config = config.get('custom_api', {})
        
        print("尝试创建自定义API适配器...")
        framework = LLMFramework('custom', custom_config)
        
        print(f"✅ LLM框架创建成功")
        print(f"   适配器类型: {framework.adapter_type}")
        print(f"   适配器类: {type(framework.llm_adapter).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM框架创建失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始测试AI配置加载和初始化...\n")
    
    # 测试配置加载
    config_ok = test_ai_config_loading()
    
    if config_ok:
        # 测试LLM框架创建
        framework_ok = test_llm_framework_creation()
        
        if framework_ok:
            print("\n🎉 所有测试通过！配置和初始化都正常")
        else:
            print("\n❌ LLM框架创建失败")
    else:
        print("\n❌ 配置加载失败")
    
    print("\n建议:")
    print("1. 如果配置正常但界面仍显示模拟适配器，可能是应用启动时机问题")
    print("2. 尝试重启应用或重新加载AI配置")
    print("3. 检查应用日志中是否有初始化错误") 
#!/usr/bin/env python3
"""
验证OpenRouter API修复
"""

import requests
import json

def test_openrouter_fix():
    """测试OpenRouter API修复"""
    
    api_key = "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"
    base_url = "https://openrouter.ai/api/v1"
    model = "deepseek/deepseek-r1-0528:free"
    
    print("=== 验证OpenRouter API修复 ===\n")
    
    # 测试1: 获取模型列表
    print("测试1: 获取模型列表")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            models_data = response.json()
            print(f"✅ 成功获取 {len(models_data.get('data', []))} 个模型")
        else:
            print(f"❌ 失败: {response.text}")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试2: 聊天完成请求
    print("测试2: 聊天完成请求")
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello, please respond with just 'OK'"}],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✅ 成功! 响应: {content}")
        else:
            print(f"❌ 失败: {response.text}")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试3: 验证配置文件中的设置
    print("测试3: 验证配置文件设置")
    try:
        with open("setting/ai_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        custom_config = config.get('custom_api', {})
        config_base_url = custom_config.get('base_url', '')
        config_model = custom_config.get('model', '')
        config_api_key = custom_config.get('api_key', '')
        
        print(f"配置文件base_url: {config_base_url}")
        print(f"配置文件model: {config_model}")
        print(f"配置文件api_key: {'已设置' if config_api_key else '未设置'}")
        
        if config_base_url == base_url:
            print("✅ base_url配置正确")
        else:
            print(f"❌ base_url配置错误，应该是: {base_url}")
            
        if config_model == model:
            print("✅ model配置正确")
        else:
            print(f"❌ model配置错误，应该是: {model}")
            
    except Exception as e:
        print(f"❌ 读取配置文件失败: {str(e)}")

if __name__ == "__main__":
    test_openrouter_fix() 
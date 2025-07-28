#!/usr/bin/env python3
"""
基于OpenRouter官方文档的认证测试
参考: https://openrouter.ai/docs
"""

import requests
import json

def test_openrouter_official():
    """基于OpenRouter官方文档的认证测试"""
    
    api_key = "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"
    base_url = "https://openrouter.ai/api/v1"
    model = "deepseek/deepseek-r1-0528:free"
    
    print("=== 基于OpenRouter官方文档的认证测试 ===\n")
    
    # 根据OpenRouter官方文档，需要以下认证头
    # 1. Authorization: Bearer <api_key> (必需)
    # 2. HTTP-Referer: <your-site> (推荐)
    # 3. X-Title: <your-app-name> (推荐)
    
    # 测试1: 官方推荐的最小配置
    print("测试1: 官方推荐的最小配置")
    headers1 = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "AI Training Assistant"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    try:
        response1 = requests.post(
            f"{base_url}/chat/completions",
            headers=headers1,
            json=data,
            timeout=10
        )
        print(f"状态码: {response1.status_code}")
        if response1.status_code == 200:
            print("✅ 成功!")
        else:
            print(f"❌ 失败: {response1.text}")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试2: 检查API密钥状态
    print("测试2: 检查API密钥状态")
    try:
        response = requests.get(f"{base_url}/auth/key", headers=headers1, timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            key_info = response.json()
            print(f"✅ API密钥有效: {key_info}")
        else:
            print(f"❌ API密钥检查失败: {response.text}")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试3: 获取模型列表（验证认证）
    print("测试3: 获取模型列表")
    try:
        response = requests.get(f"{base_url}/models", headers=headers1, timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            models_data = response.json()
            print(f"✅ 成功获取 {len(models_data.get('data', []))} 个模型")
        else:
            print(f"❌ 失败: {response.text}")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试4: 检查用户信息
    print("测试4: 检查用户信息")
    try:
        response = requests.get(f"{base_url}/auth/key", headers=headers1, timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ 用户信息: {user_info}")
        else:
            print(f"❌ 失败: {response.text}")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试5: 使用不同的模型
    print("测试5: 使用不同的模型")
    alternative_models = [
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-haiku",
        "meta-llama/llama-3.1-8b-instruct"
    ]
    
    for alt_model in alternative_models:
        print(f"尝试模型: {alt_model}")
        data["model"] = alt_model
        
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers1,
                json=data,
                timeout=10
            )
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                print(f"✅ 模型 {alt_model} 可用!")
                break
            else:
                print(f"❌ 模型 {alt_model} 失败: {response.text}")
        except Exception as e:
            print(f"❌ 异常: {str(e)}")
        
        print()

if __name__ == "__main__":
    test_openrouter_official() 
#!/usr/bin/env python3
"""
简单的OpenRouter API测试
"""

import requests
import json

def test_openrouter():
    """测试OpenRouter API"""
    
    api_key = "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"
    base_url = "https://openrouter.ai/api/v1"
    model = "deepseek/deepseek-r1-0528:free"
    
    print("=== 测试OpenRouter API ===\n")
    
    # 测试1: 只使用Authorization头
    print("测试1: 只使用Authorization头")
    headers1 = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
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
        if response1.status_code != 200:
            print(f"错误: {response1.text}")
        else:
            print("✅ 成功!")
    except Exception as e:
        print(f"异常: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试2: 只使用X-API-Key头
    print("测试2: 只使用X-API-Key头")
    headers2 = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response2 = requests.post(
            f"{base_url}/chat/completions",
            headers=headers2,
            json=data,
            timeout=10
        )
        print(f"状态码: {response2.status_code}")
        if response2.status_code != 200:
            print(f"错误: {response2.text}")
        else:
            print("✅ 成功!")
    except Exception as e:
        print(f"异常: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试3: 同时使用两种头
    print("测试3: 同时使用Authorization和X-API-Key头")
    headers3 = {
        "Authorization": f"Bearer {api_key}",
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        response3 = requests.post(
            f"{base_url}/chat/completions",
            headers=headers3,
            json=data,
            timeout=10
        )
        print(f"状态码: {response3.status_code}")
        if response3.status_code != 200:
            print(f"错误: {response3.text}")
        else:
            print("✅ 成功!")
    except Exception as e:
        print(f"异常: {str(e)}")

if __name__ == "__main__":
    test_openrouter() 
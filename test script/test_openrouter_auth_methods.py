#!/usr/bin/env python3
"""
测试OpenRouter的不同认证方法
"""

import requests
import json

def test_openrouter_auth_methods():
    """测试OpenRouter的不同认证方法"""
    
    api_key = "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"
    base_url = "https://openrouter.ai/api/v1"
    model = "deepseek/deepseek-r1-0528:free"
    
    # 测试数据
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    print("=== 测试OpenRouter认证方法 ===\n")
    
    # 方法1: 标准Bearer认证
    print("方法1: 标准Bearer认证")
    headers1 = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
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
    
    # 方法2: 添加可选头信息
    print("方法2: 添加可选头信息")
    headers2 = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "AI Training Assistant"
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
    
    # 方法3: 使用X-API-Key头
    print("方法3: 使用X-API-Key头")
    headers3 = {
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
    
    print("\n" + "="*50 + "\n")
    
    # 方法4: 同时使用两种认证头
    print("方法4: 同时使用两种认证头")
    headers4 = {
        "Authorization": f"Bearer {api_key}",
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "AI Training Assistant"
    }
    
    try:
        response4 = requests.post(
            f"{base_url}/chat/completions",
            headers=headers4,
            json=data,
            timeout=10
        )
        print(f"状态码: {response4.status_code}")
        if response4.status_code != 200:
            print(f"错误: {response4.text}")
        else:
            print("✅ 成功!")
    except Exception as e:
        print(f"异常: {str(e)}")

if __name__ == "__main__":
    test_openrouter_auth_methods() 
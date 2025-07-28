#!/usr/bin/env python3
"""
详细的OpenRouter认证测试
"""

import requests
import json

def test_openrouter_detailed_auth():
    """详细的OpenRouter认证测试"""
    
    api_key = "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"
    base_url = "https://openrouter.ai/api/v1"
    model = "deepseek/deepseek-r1-0528:free"
    
    print("=== 详细OpenRouter认证测试 ===\n")
    
    # 测试数据
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    # 测试1: 只使用Authorization头
    print("测试1: 只使用Authorization头")
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
        if response1.status_code == 200:
            print("✅ 成功!")
        else:
            print(f"❌ 失败: {response1.text}")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
    
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
        if response2.status_code == 200:
            print("✅ 成功!")
        else:
            print(f"❌ 失败: {response2.text}")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
    
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
        if response3.status_code == 200:
            print("✅ 成功!")
        else:
            print(f"❌ 失败: {response3.text}")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试4: 添加HTTP-Referer头（OpenRouter推荐）
    print("测试4: 添加HTTP-Referer头")
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
        if response4.status_code == 200:
            print("✅ 成功!")
        else:
            print(f"❌ 失败: {response4.text}")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试5: 检查API密钥格式
    print("测试5: 检查API密钥格式")
    print(f"API密钥长度: {len(api_key)}")
    print(f"API密钥前缀: {api_key[:10]}...")
    print(f"API密钥是否以'sk-or-v1-'开头: {api_key.startswith('sk-or-v1-')}")
    
    # 测试6: 获取用户信息
    print("\n测试6: 获取用户信息")
    try:
        response = requests.get(f"{base_url}/auth/key", headers=headers3, timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ 用户信息: {user_info}")
        else:
            print(f"❌ 失败: {response.text}")
    except Exception as e:
        print(f"❌ 异常: {str(e)}")

if __name__ == "__main__":
    test_openrouter_detailed_auth() 
#!/usr/bin/env python3
"""
最终的OpenRouter修复验证
"""

import requests
import json

def test_openrouter_final():
    """最终的OpenRouter修复验证"""
    
    api_key = "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"
    base_url = "https://openrouter.ai/api/v1"
    model = "deepseek/deepseek-r1-0528:free"
    
    print("=== 最终的OpenRouter修复验证 ===\n")
    
    # 使用修复后的认证头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-training-assistant",
        "X-Title": "AI Training Assistant"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello, please respond with just 'OK'"}],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    print("测试聊天完成请求...")
    print(f"URL: {base_url}/chat/completions")
    print(f"模型: {model}")
    print(f"认证头: {headers}")
    
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        
        print(f"\n状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✅ 成功! 响应: {content}")
            return True
        else:
            print(f"❌ 失败: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_openrouter_final()
    if success:
        print("\n🎉 OpenRouter API修复成功!")
    else:
        print("\n❌ OpenRouter API仍有问题，需要进一步调查") 
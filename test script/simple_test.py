"""
简单测试脚本 - 验证数据流基础功能
"""

import requests
import json
import time

def test_basic_functionality():
    """测试基本功能"""
    base_url = "http://127.0.0.1:5000"
    
    print("🧪 开始测试数据流基础功能")
    print("=" * 50)
    
    # 测试健康检查
    try:
        response = requests.get(f"{base_url}/api/system/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查: 通过 - {data['service']}")
        else:
            print(f"❌ 健康检查: 失败 (状态码: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ 健康检查: 异常 - {str(e)}")
        return False
    
    # 测试系统信息
    try:
        response = requests.get(f"{base_url}/api/system/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 系统信息: 通过 - 指标数量: {data['data']['metrics_count']}")
        else:
            print(f"❌ 系统信息: 失败")
            return False
    except Exception as e:
        print(f"❌ 系统信息: 异常 - {str(e)}")
        return False
    
    # 测试当前指标
    try:
        response = requests.get(f"{base_url}/api/metrics/current", timeout=5)
        if response.status_code == 200:
            data = response.json()
            metrics = data['data']
            if metrics:
                print(f"✅ 当前指标: 通过 - Epoch: {metrics.get('epoch', 'N/A')}, Loss: {metrics.get('train_loss', 'N/A')}")
            else:
                print("✅ 当前指标: 通过 - 暂无数据")
        else:
            print(f"❌ 当前指标: 失败")
            return False
    except Exception as e:
        print(f"❌ 当前指标: 异常 - {str(e)}")
        return False
    
    # 测试SSE数据流 (简单测试)
    try:
        response = requests.get(f"{base_url}/api/stream/metrics", stream=True, timeout=10)
        if response.status_code == 200:
            print("✅ SSE数据流: 开始接收...")
            
            # 读取前几条消息
            messages_received = 0
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        try:
                            data = json.loads(decoded_line[5:])  # 去掉 "data:" 前缀
                            print(f"  📊 接收数据: {data.get('type', 'unknown')} - {data.get('data', {}).get('epoch', 'N/A')}")
                            messages_received += 1
                            if messages_received >= 5:  # 只接收前5条消息
                                break
                        except json.JSONDecodeError:
                            continue
            
            print(f"✅ SSE数据流: 通过 - 接收了 {messages_received} 条消息")
        else:
            print(f"❌ SSE数据流: 失败")
            return False
    except Exception as e:
        print(f"❌ SSE数据流: 异常 - {str(e)}")
        return False
    
    print("\n🎉 所有基础功能测试通过！")
    return True

def main():
    """主函数"""
    print("等待服务器启动...")
    time.sleep(5)  # 等待服务器启动
    
    if test_basic_functionality():
        print("\n✅ 数据流基础设施测试成功！")
        print("📊 第一阶段核心功能已实现：")
        print("  • SSE实时数据流")
        print("  • REST API接口")
        print("  • 训练指标模拟")
        print("  • 健康检查机制")
    else:
        print("\n❌ 测试失败，请检查服务器状态")

if __name__ == "__main__":
    main() 
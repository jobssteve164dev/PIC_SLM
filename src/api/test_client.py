"""
数据流测试客户端 - 用于测试SSE、WebSocket和REST API接口

提供完整的测试套件来验证数据流服务器的功能。
"""

import requests
import asyncio
import websockets
import json
import time
import threading
from typing import Dict, Any


class DataStreamTestClient:
    """数据流测试客户端"""
    
    def __init__(self):
        self.base_urls = {
            'sse': 'http://127.0.0.1:8888',
            'websocket': 'ws://127.0.0.1:8889',
            'rest_api': 'http://127.0.0.1:8890'
        }
        self.test_results = {}
        
    def test_all_services(self):
        """测试所有服务"""
        print("🚀 开始测试数据流服务...")
        print("=" * 60)
        
        # 测试REST API
        self.test_rest_api()
        
        # 测试SSE
        self.test_sse_stream()
        
        # 测试WebSocket
        asyncio.run(self.test_websocket())
        
        # 打印测试结果
        self.print_test_results()
    
    def test_rest_api(self):
        """测试REST API接口"""
        print("📡 测试REST API接口...")
        
        try:
            # 测试健康检查
            response = requests.get(f"{self.base_urls['rest_api']}/api/system/health", timeout=5)
            if response.status_code == 200:
                print("✅ 健康检查: 通过")
                self.test_results['rest_health'] = True
            else:
                print(f"❌ 健康检查: 失败 (状态码: {response.status_code})")
                self.test_results['rest_health'] = False
        except Exception as e:
            print(f"❌ 健康检查: 异常 ({str(e)})")
            self.test_results['rest_health'] = False
        
        try:
            # 测试系统信息
            response = requests.get(f"{self.base_urls['rest_api']}/api/system/info", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 系统信息: 通过 (请求数: {data['data']['total_requests']})")
                self.test_results['rest_info'] = True
            else:
                print(f"❌ 系统信息: 失败")
                self.test_results['rest_info'] = False
        except Exception as e:
            print(f"❌ 系统信息: 异常 ({str(e)})")
            self.test_results['rest_info'] = False
        
        try:
            # 测试当前指标
            response = requests.get(f"{self.base_urls['rest_api']}/api/metrics/current", timeout=5)
            if response.status_code == 200:
                print("✅ 当前指标: 通过")
                self.test_results['rest_metrics'] = True
            else:
                print(f"❌ 当前指标: 失败")
                self.test_results['rest_metrics'] = False
        except Exception as e:
            print(f"❌ 当前指标: 异常 ({str(e)})")
            self.test_results['rest_metrics'] = False
        
        try:
            # 测试训练状态
            response = requests.get(f"{self.base_urls['rest_api']}/api/training/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 训练状态: 通过 (训练中: {data['data']['is_training']})")
                self.test_results['rest_training'] = True
            else:
                print(f"❌ 训练状态: 失败")
                self.test_results['rest_training'] = False
        except Exception as e:
            print(f"❌ 训练状态: 异常 ({str(e)})")
            self.test_results['rest_training'] = False
    
    def test_sse_stream(self):
        """测试SSE数据流"""
        print("\n📺 测试SSE数据流...")
        
        try:
            # 测试SSE状态
            response = requests.get(f"{self.base_urls['sse']}/api/stream/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ SSE状态: 通过 (活跃客户端: {data['active_clients']})")
                self.test_results['sse_status'] = True
            else:
                print(f"❌ SSE状态: 失败")
                self.test_results['sse_status'] = False
        except Exception as e:
            print(f"❌ SSE状态: 异常 ({str(e)})")
            self.test_results['sse_status'] = False
        
        try:
            # 测试SSE历史数据
            response = requests.get(f"{self.base_urls['sse']}/api/stream/history?limit=5", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ SSE历史: 通过 (记录数: {data['total_count']})")
                self.test_results['sse_history'] = True
            else:
                print(f"❌ SSE历史: 失败")
                self.test_results['sse_history'] = False
        except Exception as e:
            print(f"❌ SSE历史: 异常 ({str(e)})")
            self.test_results['sse_history'] = False
        
        # 测试SSE实时流（短时间连接）
        def test_sse_connection():
            try:
                response = requests.get(f"{self.base_urls['sse']}/api/stream/metrics", 
                                      stream=True, timeout=10)
                
                if response.status_code == 200:
                    # 读取前几条消息
                    messages_received = 0
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith('data:'):
                                messages_received += 1
                                if messages_received >= 3:  # 接收3条消息后断开
                                    break
                    
                    print(f"✅ SSE实时流: 通过 (接收消息: {messages_received}条)")
                    self.test_results['sse_stream'] = True
                else:
                    print(f"❌ SSE实时流: 失败")
                    self.test_results['sse_stream'] = False
                    
            except Exception as e:
                print(f"❌ SSE实时流: 异常 ({str(e)})")
                self.test_results['sse_stream'] = False
        
        # 在独立线程中测试SSE连接
        sse_thread = threading.Thread(target=test_sse_connection)
        sse_thread.start()
        sse_thread.join(timeout=15)  # 最多等待15秒
    
    async def test_websocket(self):
        """测试WebSocket连接"""
        print("\n🔌 测试WebSocket连接...")
        
        try:
            async with websockets.connect(self.base_urls['websocket']) as websocket:
                # 等待连接确认
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(response)
                
                if data.get('type') == 'connection' and data.get('status') == 'connected':
                    print(f"✅ WebSocket连接: 通过 (客户端ID: {data.get('client_id')})")
                    self.test_results['ws_connection'] = True
                    
                    # 测试ping-pong
                    await websocket.send(json.dumps({'type': 'ping'}))
                    pong_response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    pong_data = json.loads(pong_response)
                    
                    if pong_data.get('type') == 'pong':
                        print("✅ WebSocket心跳: 通过")
                        self.test_results['ws_ping'] = True
                    else:
                        print("❌ WebSocket心跳: 失败")
                        self.test_results['ws_ping'] = False
                    
                    # 测试订阅
                    await websocket.send(json.dumps({'type': 'subscribe', 'subscription': 'metrics'}))
                    sub_response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    sub_data = json.loads(sub_response)
                    
                    if sub_data.get('type') == 'subscription_confirmed':
                        print("✅ WebSocket订阅: 通过")
                        self.test_results['ws_subscribe'] = True
                    else:
                        print("❌ WebSocket订阅: 失败")
                        self.test_results['ws_subscribe'] = False
                    
                    # 测试获取历史数据
                    await websocket.send(json.dumps({'type': 'get_history', 'limit': 5}))
                    history_response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    history_data = json.loads(history_response)
                    
                    if history_data.get('type') == 'history_data':
                        print(f"✅ WebSocket历史: 通过 (记录数: {history_data.get('count', 0)})")
                        self.test_results['ws_history'] = True
                    else:
                        print("❌ WebSocket历史: 失败")
                        self.test_results['ws_history'] = False
                else:
                    print("❌ WebSocket连接: 失败")
                    self.test_results['ws_connection'] = False
                    
        except asyncio.TimeoutError:
            print("❌ WebSocket连接: 超时")
            self.test_results['ws_connection'] = False
        except Exception as e:
            print(f"❌ WebSocket连接: 异常 ({str(e)})")
            self.test_results['ws_connection'] = False
    
    def test_training_control(self):
        """测试训练控制接口"""
        print("\n🎮 测试训练控制接口...")
        
        try:
            # 测试获取配置
            response = requests.post(
                f"{self.base_urls['rest_api']}/api/training/control",
                json={'action': 'get_config'},
                timeout=5
            )
            
            if response.status_code == 200:
                print("✅ 训练控制: 通过")
                self.test_results['training_control'] = True
            else:
                print(f"❌ 训练控制: 失败 (状态码: {response.status_code})")
                self.test_results['training_control'] = False
                
        except Exception as e:
            print(f"❌ 训练控制: 异常 ({str(e)})")
            self.test_results['training_control'] = False
    
    def send_test_metrics(self):
        """发送测试指标数据"""
        print("\n📊 发送测试指标数据...")
        
        # 模拟训练指标
        test_metrics = {
            'epoch': 1,
            'train_loss': 0.5234,
            'val_loss': 0.4876,
            'train_accuracy': 0.8234,
            'val_accuracy': 0.8456,
            'learning_rate': 0.001,
            'batch_size': 32,
            'model_name': 'ResNet50'
        }
        
        try:
            # 这里应该通过训练系统发送指标，但在测试环境中我们直接模拟
            print(f"📤 模拟发送指标: {test_metrics}")
            print("✅ 测试指标发送: 完成")
            
        except Exception as e:
            print(f"❌ 测试指标发送: 异常 ({str(e)})")
    
    def print_test_results(self):
        """打印测试结果汇总"""
        print("\n" + "=" * 60)
        print("📋 测试结果汇总")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"总测试数: {total_tests}")
        print(f"通过数: {passed_tests}")
        print(f"失败数: {total_tests - passed_tests}")
        print(f"成功率: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n详细结果:")
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {test_name}: {status}")
        
        if passed_tests == total_tests:
            print("\n🎉 所有测试通过！数据流服务器运行正常。")
        else:
            print("\n⚠️ 部分测试失败，请检查服务器状态。")


def main():
    """主函数"""
    print("🧪 数据流服务测试客户端")
    print("=" * 60)
    
    client = DataStreamTestClient()
    
    # 等待服务器启动
    print("⏳ 等待服务器启动...")
    time.sleep(3)
    
    # 运行所有测试
    client.test_all_services()
    
    # 测试训练控制
    client.test_training_control()
    
    # 发送测试指标
    client.send_test_metrics()


if __name__ == "__main__":
    main() 
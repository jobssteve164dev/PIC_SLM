#!/usr/bin/env python3
"""
数据流监控组件修复验证测试脚本

用于测试修复后的数据流监控组件是否能正确连接到数据流服务器。
"""

import sys
import os
import time
import requests
import asyncio
import websockets
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ui.components.model_analysis.real_time_stream_monitor import RealTimeStreamMonitor


class TestWindow(QMainWindow):
    """测试窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据流监控组件修复测试")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)
        
        # 创建监控组件
        self.monitor = RealTimeStreamMonitor()
        layout.addWidget(self.monitor)
        
        # 创建定时器，定期测试连接
        self.test_timer = QTimer()
        self.test_timer.timeout.connect(self.periodic_test)
        self.test_timer.start(10000)  # 每10秒测试一次
        
        # 启动监控
        self.monitor.start_monitoring()


def test_server_endpoints():
    """测试服务器端点是否可用"""
    print("🔍 测试服务器端点...")
    
    endpoints = {
        'SSE': 'http://127.0.0.1:8888/api/stream/metrics',
        'WebSocket': 'ws://127.0.0.1:8889',
        'REST API': 'http://127.0.0.1:8890/api/metrics/current'
    }
    
    results = {}
    
    # 测试SSE端点
    try:
        response = requests.get(endpoints['SSE'], timeout=5)
        results['SSE'] = f"✅ 可用 (HTTP {response.status_code})"
    except Exception as e:
        results['SSE'] = f"❌ 不可用: {type(e).__name__}"
    
    # 测试REST API端点
    try:
        response = requests.get(endpoints['REST API'], timeout=5)
        results['REST API'] = f"✅ 可用 (HTTP {response.status_code})"
    except Exception as e:
        results['REST API'] = f"❌ 不可用: {type(e).__name__}"
    
    # 测试WebSocket端点
    async def test_websocket():
        try:
            # 修复：使用asyncio.wait_for来控制超时
            websocket = await asyncio.wait_for(
                websockets.connect(endpoints['WebSocket']), 
                timeout=5
            )
            async with websocket:
                await websocket.ping()
                return True
        except Exception as e:
            return str(e)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(test_websocket())
        loop.close()
        
        if result is True:
            results['WebSocket'] = "✅ 可用"
        else:
            results['WebSocket'] = f"❌ 不可用: {result}"
    except Exception as e:
        results['WebSocket'] = f"❌ 测试失败: {type(e).__name__}"
    
    # 打印结果
    print("\n📊 服务器端点测试结果:")
    for endpoint, result in results.items():
        print(f"  {endpoint}: {result}")
    
    return results


def main():
    """主函数"""
    print("🚀 数据流监控组件修复验证测试")
    print("=" * 50)
    
    # 测试服务器端点
    server_results = test_server_endpoints()
    
    # 检查是否有可用的端点
    available_endpoints = [k for k, v in server_results.items() if '✅' in v]
    
    if not available_endpoints:
        print("\n❌ 没有可用的数据流服务器端点")
        print("请确保训练系统已启动并运行数据流服务器")
        return
    
    print(f"\n✅ 发现 {len(available_endpoints)} 个可用端点: {', '.join(available_endpoints)}")
    
    # 启动GUI测试
    print("\n🖥️ 启动GUI测试...")
    app = QApplication(sys.argv)
    
    # 创建测试窗口
    window = TestWindow()
    window.show()
    
    print("✅ GUI测试已启动，请观察监控组件的连接状态")
    print("💡 提示:")
    print("  - 点击'🔍 测试连接'按钮可以手动测试连接")
    print("  - 点击'▶️ 开始监控'按钮开始自动监控")
    print("  - 观察状态标签的颜色变化")
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
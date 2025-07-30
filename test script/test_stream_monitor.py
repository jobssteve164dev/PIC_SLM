#!/usr/bin/env python3
"""
实时数据流监控控件测试脚本

用于测试实时数据流监控控件的功能，包括SSE、WebSocket和REST API连接。
"""

import sys
import os
import time
import json
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_stream_monitor():
    """测试实时数据流监控控件"""
    app = QApplication(sys.argv)
    
    # 创建主窗口
    main_window = QMainWindow()
    main_window.setWindowTitle("实时数据流监控测试")
    main_window.setGeometry(100, 100, 1200, 800)
    
    # 创建中央控件
    central_widget = QWidget()
    main_window.setCentralWidget(central_widget)
    
    # 创建布局
    layout = QVBoxLayout(central_widget)
    
    try:
        # 导入实时数据流监控控件
        from src.ui.components.model_analysis.real_time_stream_monitor import RealTimeStreamMonitor
        stream_monitor = RealTimeStreamMonitor()
        layout.addWidget(stream_monitor)
        
        print("✅ 实时数据流监控控件加载成功")
        
        # 添加测试说明
        from PyQt5.QtWidgets import QLabel
        info_label = QLabel("""
测试说明：
1. 点击"开始监控"按钮开始监控数据流
2. 确保训练系统正在运行并启动了数据流服务器
3. 观察SSE、WebSocket和REST API的连接状态
4. 查看各个标签页中的数据流内容
5. 使用"导出数据"功能保存监控数据
        """)
        info_label.setStyleSheet("background-color: #f8f9fa; padding: 10px; border: 1px solid #dee2e6; border-radius: 5px;")
        layout.addWidget(info_label)
        
    except ImportError as e:
        print(f"❌ 实时数据流监控控件导入失败: {e}")
        from PyQt5.QtWidgets import QLabel
        error_label = QLabel(f"实时数据流监控控件导入失败: {str(e)}")
        error_label.setStyleSheet("color: #dc3545; padding: 20px; border: 1px solid #dc3545; border-radius: 5px;")
        layout.addWidget(error_label)
    
    # 显示窗口
    main_window.show()
    
    print("🚀 实时数据流监控测试窗口已启动")
    print("💡 请确保训练系统正在运行并启动了数据流服务器")
    
    return app.exec_()

def test_data_stream_servers():
    """测试数据流服务器是否正在运行"""
    import requests
    import websockets
    import asyncio
    
    print("\n🔍 检查数据流服务器状态...")
    
    # 测试SSE服务器
    try:
        response = requests.get("http://127.0.0.1:8888/api/stream/status", timeout=3)
        if response.status_code == 200:
            print("✅ SSE服务器正在运行")
        else:
            print("❌ SSE服务器响应异常")
    except Exception as e:
        print(f"❌ SSE服务器未运行: {e}")
    
    # 测试REST API服务器
    try:
        response = requests.get("http://127.0.0.1:8890/api/system/health", timeout=3)
        if response.status_code == 200:
            print("✅ REST API服务器正在运行")
        else:
            print("❌ REST API服务器响应异常")
    except Exception as e:
        print(f"❌ REST API服务器未运行: {e}")
    
    # 测试WebSocket服务器
    async def test_websocket():
        try:
            async with websockets.connect("ws://127.0.0.1:8889") as websocket:
                print("✅ WebSocket服务器正在运行")
        except Exception as e:
            print(f"❌ WebSocket服务器未运行: {e}")
    
    try:
        asyncio.run(test_websocket())
    except Exception as e:
        print(f"❌ WebSocket测试失败: {e}")

if __name__ == "__main__":
    print("🧪 实时数据流监控控件测试")
    print("=" * 50)
    
    # 首先检查数据流服务器状态
    test_data_stream_servers()
    
    print("\n" + "=" * 50)
    print("🎯 启动实时数据流监控测试界面...")
    
    # 启动测试界面
    sys.exit(test_stream_monitor()) 
#!/usr/bin/env python3
"""
只测试WebSocket服务器

测试WebSocket服务器是否能正常启动和停止。
"""

import sys
import os
import time
import threading

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_websocket_only():
    """只测试WebSocket服务器"""
    print("🧪 只测试WebSocket服务器")
    print("=" * 50)
    
    try:
        # 导入WebSocket处理器
        from src.api.websocket_handler import WebSocketHandler
        
        print("✅ 成功导入WebSocket处理器")
        
        # 第一次启动
        print("\n📡 第一次启动WebSocket服务器")
        ws1 = WebSocketHandler(host='127.0.0.1', port=8889)
        print(f"WebSocket1状态: {'运行中' if ws1.is_running else '未运行'}")
        
        # 启动服务器
        ws_thread1 = ws1.start_server_thread()
        print("WebSocket服务器线程已启动")
        
        # 等待服务器启动
        time.sleep(5)
        
        # 检查服务器状态
        if ws1.is_running:
            print("✅ 第一次启动成功")
        else:
            print("❌ 第一次启动失败")
            return
        
        # 停止服务器
        print("\n📡 停止WebSocket服务器")
        ws1.stop_server_sync()
        time.sleep(3)
        
        # 检查服务器是否已停止
        if not ws1.is_running:
            print("✅ 服务器已正确停止")
        else:
            print("❌ 服务器未能正确停止")
        
        # 第二次启动
        print("\n📡 第二次启动WebSocket服务器")
        ws2 = WebSocketHandler(host='127.0.0.1', port=8889)
        print(f"WebSocket2状态: {'运行中' if ws2.is_running else '未运行'}")
        
        # 启动服务器
        ws_thread2 = ws2.start_server_thread()
        print("WebSocket服务器线程已启动")
        
        # 等待服务器启动
        time.sleep(5)
        
        # 检查服务器状态
        if ws2.is_running:
            print("✅ 第二次启动成功")
        else:
            print("❌ 第二次启动失败")
        
        # 清理
        ws2.stop_server_sync()
        
        print("\n✅ WebSocket测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 WebSocket服务器测试")
    print("=" * 60)
    
    test_websocket_only()
    
    print("\n" + "=" * 60)
    print("🎯 测试完成") 
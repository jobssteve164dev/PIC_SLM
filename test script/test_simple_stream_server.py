#!/usr/bin/env python3
"""
简单的数据流服务器测试

测试修复后的数据流服务器是否能正常启动和停止。
"""

import sys
import os
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple_stream_server():
    """简单测试数据流服务器"""
    print("🧪 简单数据流服务器测试")
    print("=" * 50)
    
    try:
        # 导入数据流服务器管理器
        from src.api.stream_server_manager import get_stream_server, release_stream_server, is_stream_server_running
        
        print("✅ 成功导入数据流服务器管理器")
        
        # 第一次启动
        print("\n📡 第一次启动数据流服务器")
        server1 = get_stream_server()
        print(f"服务器1状态: {'运行中' if server1.is_running else '未运行'}")
        
        # 等待服务器启动
        time.sleep(5)
        
        # 检查服务器状态
        if is_stream_server_running():
            print("✅ 第一次启动成功")
        else:
            print("❌ 第一次启动失败")
            return
        
        # 释放引用
        print("\n📡 释放服务器引用")
        release_stream_server()
        time.sleep(3)
        
        # 检查服务器是否已停止
        if not is_stream_server_running():
            print("✅ 服务器已正确停止")
        else:
            print("❌ 服务器未能正确停止")
        
        # 第二次启动
        print("\n📡 第二次启动数据流服务器")
        server2 = get_stream_server()
        print(f"服务器2状态: {'运行中' if server2.is_running else '未运行'}")
        
        # 等待服务器启动
        time.sleep(5)
        
        # 检查服务器状态
        if is_stream_server_running():
            print("✅ 第二次启动成功")
        else:
            print("❌ 第二次启动失败")
        
        # 清理
        release_stream_server()
        
        print("\n✅ 简单测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 简单数据流服务器测试")
    print("=" * 60)
    
    test_simple_stream_server()
    
    print("\n" + "=" * 60)
    print("🎯 测试完成") 
#!/usr/bin/env python3
"""
测试数据流服务器启动修复
"""

import sys
import os
import time
import logging

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from api.stream_server_manager import get_stream_server

def test_stream_server_startup():
    """测试数据流服务器启动"""
    print("🧪 测试数据流服务器启动修复...")
    print("=" * 60)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s:%(message)s'
    )
    
    try:
        # 创建数据流服务器配置
        stream_config = {
            'sse_host': '127.0.0.1',
            'sse_port': 8888,
            'websocket_host': '127.0.0.1',
            'websocket_port': 8889,
            'rest_api_host': '127.0.0.1',
            'rest_api_port': 8890,
            'buffer_size': 1000,
            'debug_mode': False
        }
        
        print("🔧 获取数据流服务器实例...")
        
        # 获取数据流服务器实例
        stream_server = get_stream_server(
            training_system=None,
            config=stream_config
        )
        
        print("⏳ 等待服务器启动完成...")
        time.sleep(3)
        
        # 检查服务器状态
        if stream_server.is_running:
            print("✅ 数据流服务器启动成功!")
            
            # 获取服务器信息
            server_info = stream_server.get_server_info()
            print(f"📊 服务器运行状态: {server_info['is_running']}")
            print(f"🔗 端点信息:")
            for name, url in server_info['endpoints'].items():
                print(f"   • {name}: {url}")
            
            # 获取API端点列表
            endpoints = stream_server.get_api_endpoints()
            print(f"📡 API端点数量: {len(endpoints)}")
            
            print("\n🎉 数据流服务器启动测试通过!")
            
        else:
            print("❌ 数据流服务器启动失败!")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return False
    
    finally:
        # 清理资源
        try:
            from api.stream_server_manager import release_stream_server
            release_stream_server()
            print("🧹 已清理数据流服务器资源")
        except Exception as e:
            print(f"⚠️ 清理资源时出错: {str(e)}")
    
    return True

if __name__ == "__main__":
    success = test_stream_server_startup()
    sys.exit(0 if success else 1) 
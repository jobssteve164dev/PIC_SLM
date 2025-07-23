"""
演示服务器 - 用于测试数据流基础设施

独立运行数据流服务器，模拟训练指标数据。
"""

import time
import threading
import random
from stream_server import TrainingStreamServer


class MockTrainingSystem:
    """模拟训练系统"""
    
    def __init__(self):
        self.is_training = False
        self.current_epoch = 0
        self.total_epochs = 20
        
    def start_mock_training(self, stream_server):
        """开始模拟训练"""
        self.is_training = True
        
        for epoch in range(1, self.total_epochs + 1):
            if not self.is_training:
                break
                
            self.current_epoch = epoch
            
            # 模拟训练指标
            train_loss = max(0.1, 2.0 - epoch * 0.08 + random.uniform(-0.1, 0.1))
            val_loss = max(0.1, 2.2 - epoch * 0.075 + random.uniform(-0.1, 0.1))
            train_acc = min(0.95, 0.3 + epoch * 0.03 + random.uniform(-0.02, 0.02))
            val_acc = min(0.92, 0.25 + epoch * 0.032 + random.uniform(-0.02, 0.02))
            
            # 训练阶段指标
            train_metrics = {
                'epoch': epoch,
                'phase': 'train',
                'train_loss': round(train_loss, 4),
                'train_accuracy': round(train_acc, 4),
                'learning_rate': 0.001 * (0.9 ** (epoch // 5)),
                'batch_size': 32,
                'model_name': 'ResNet50',
                'gpu_memory_used': round(random.uniform(5.5, 7.2), 2),
                'gpu_memory_total': 8.0,
                'training_speed': round(random.uniform(1.8, 2.5), 2),
                'timestamp': time.time()
            }
            
            print(f"📊 Epoch {epoch}/{self.total_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            stream_server.broadcast_metrics(train_metrics)
            
            time.sleep(2)  # 模拟训练时间
            
            # 验证阶段指标
            val_metrics = {
                'epoch': epoch,
                'phase': 'val',
                'val_loss': round(val_loss, 4),
                'val_accuracy': round(val_acc, 4),
                'learning_rate': 0.001 * (0.9 ** (epoch // 5)),
                'batch_size': 32,
                'model_name': 'ResNet50',
                'gpu_memory_used': round(random.uniform(5.5, 7.2), 2),
                'gpu_memory_total': 8.0,
                'training_speed': round(random.uniform(1.8, 2.5), 2),
                'timestamp': time.time()
            }
            
            print(f"📊 Epoch {epoch}/{self.total_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            stream_server.broadcast_metrics(val_metrics)
            
            time.sleep(1)  # 验证时间
        
        self.is_training = False
        print("🎉 模拟训练完成！")
    
    def stop_training(self):
        """停止训练"""
        self.is_training = False


def main():
    """主函数"""
    print("🚀 启动数据流演示服务器")
    print("=" * 60)
    
    # 创建模拟训练系统
    mock_system = MockTrainingSystem()
    
    # 创建数据流服务器
    server_config = {
        'sse_host': '127.0.0.1',
        'sse_port': 8888,
        'websocket_host': '127.0.0.1',
        'websocket_port': 8889,
        'rest_api_host': '127.0.0.1',
        'rest_api_port': 8890,
        'buffer_size': 1000,
        'debug_mode': False
    }
    
    stream_server = TrainingStreamServer(
        training_system=mock_system,
        config=server_config
    )
    
    try:
        # 启动所有服务器
        print("🔧 启动数据流服务器...")
        stream_server.start_all_servers()
        
        # 等待服务器启动完成
        time.sleep(3)
        
        # 显示服务器信息
        server_info = stream_server.get_server_info()
        print("\n📡 服务器信息:")
        print(f"• 运行状态: {'✅ 运行中' if server_info['is_running'] else '❌ 已停止'}")
        print(f"• SSE端点: {server_info['endpoints']['sse']}")
        print(f"• WebSocket端点: {server_info['endpoints']['websocket']}")
        print(f"• REST API端点: {server_info['endpoints']['rest_api']}")
        
        # 显示API端点列表
        endpoints = stream_server.get_api_endpoints()
        print("\n🔗 API端点列表:")
        for name, url in endpoints.items():
            print(f"• {name}: {url}")
        
        print("\n💡 提示:")
        print("• 运行 'python src/api/test_client.py' 来测试所有接口")
        print("• 在浏览器中访问 REST API 端点查看数据")
        print("• 使用 SSE 客户端连接实时数据流")
        print("• 按 Ctrl+C 停止服务器")
        
        # 在独立线程中开始模拟训练
        training_thread = threading.Thread(
            target=mock_system.start_mock_training,
            args=(stream_server,),
            daemon=True
        )
        training_thread.start()
        
        # 保持服务器运行
        try:
            while True:
                time.sleep(1)
                
                # 检查服务器状态
                if not stream_server.is_running:
                    print("⚠️ 服务器已停止")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 收到停止信号...")
            
    except Exception as e:
        print(f"❌ 服务器启动失败: {str(e)}")
        
    finally:
        # 停止服务器
        print("🔧 正在停止服务器...")
        mock_system.stop_training()
        stream_server.stop_all_servers()
        print("✅ 服务器已停止")


if __name__ == "__main__":
    main() 
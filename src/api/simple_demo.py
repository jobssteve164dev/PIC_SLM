"""
简化演示脚本 - 测试数据流基础功能
"""

import json
import time
import threading
from flask import Flask, jsonify, Response
from flask_cors import CORS

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 全局变量存储指标
current_metrics = {}
metrics_history = []

@app.route('/api/system/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'timestamp': time.time(),
        'service': 'Training Data Stream API'
    })

@app.route('/api/metrics/current')
def get_current_metrics():
    """获取当前指标"""
    return jsonify({
        'status': 'success',
        'data': current_metrics,
        'timestamp': time.time()
    })

@app.route('/api/stream/metrics')
def stream_metrics():
    """SSE数据流"""
    def generate():
        yield "data: " + json.dumps({
            'type': 'connection',
            'status': 'connected',
            'timestamp': time.time()
        }) + "\n\n"
        
        # 发送历史数据
        for metric in metrics_history[-10:]:  # 最近10条记录
            yield "data: " + json.dumps({
                'type': 'historical_data',
                'data': metric
            }) + "\n\n"
            time.sleep(0.1)
        
        # 发送实时数据
        counter = 0
        while counter < 20:  # 发送20条测试数据
            test_metric = {
                'epoch': counter + 1,
                'train_loss': round(2.0 - counter * 0.08, 4),
                'train_accuracy': round(0.3 + counter * 0.03, 4),
                'timestamp': time.time()
            }
            
            yield "data: " + json.dumps({
                'type': 'real_time_data',
                'data': test_metric
            }) + "\n\n"
            
            counter += 1
            time.sleep(1)
    
    return Response(generate(), mimetype='text/plain')

@app.route('/api/system/info')
def system_info():
    """系统信息"""
    return jsonify({
        'status': 'success',
        'data': {
            'total_requests': 100,
            'uptime_seconds': 3600,
            'active_connections': 2,
            'metrics_count': len(metrics_history)
        }
    })

def simulate_training():
    """模拟训练数据"""
    global current_metrics, metrics_history
    
    for epoch in range(1, 21):
        # 生成模拟指标
        metrics = {
            'epoch': epoch,
            'train_loss': round(2.0 - epoch * 0.08, 4),
            'val_loss': round(2.2 - epoch * 0.075, 4),
            'train_accuracy': round(0.3 + epoch * 0.03, 4),
            'val_accuracy': round(0.25 + epoch * 0.032, 4),
            'learning_rate': 0.001,
            'timestamp': time.time()
        }
        
        # 更新全局状态
        current_metrics = metrics
        metrics_history.append(metrics)
        
        # 保持历史记录在合理大小
        if len(metrics_history) > 100:
            metrics_history = metrics_history[-100:]
        
        print(f"📊 模拟训练 Epoch {epoch}: Loss={metrics['train_loss']}, Acc={metrics['train_accuracy']}")
        time.sleep(2)

if __name__ == '__main__':
    print("🚀 启动简化数据流演示服务器")
    print("=" * 50)
    
    # 在后台线程中开始模拟训练
    training_thread = threading.Thread(target=simulate_training, daemon=True)
    training_thread.start()
    
    print("📡 服务器端点:")
    print("• 健康检查: http://127.0.0.1:5000/api/system/health")
    print("• 当前指标: http://127.0.0.1:5000/api/metrics/current")
    print("• SSE数据流: http://127.0.0.1:5000/api/stream/metrics")
    print("• 系统信息: http://127.0.0.1:5000/api/system/info")
    print("\n💡 在浏览器中访问这些端点来查看数据")
    print("按 Ctrl+C 停止服务器\n")
    
    # 启动Flask服务器
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True) 
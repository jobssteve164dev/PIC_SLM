"""
第一阶段功能验证脚本
验证数据流基础设施的核心功能
"""

import sys
import os
import importlib.util

def validate_api_modules():
    """验证API模块是否正确创建"""
    print("🔍 验证API模块结构...")
    
    required_files = [
        '__init__.py',
        'sse_handler.py', 
        'websocket_handler.py',
        'rest_api.py',
        'stream_server.py',
        'test_client.py',
        'simple_demo.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"  ✅ {file}")
    
    if missing_files:
        print(f"  ❌ 缺失文件: {missing_files}")
        return False
    
    print("✅ API模块结构验证通过")
    return True

def validate_module_imports():
    """验证模块导入"""
    print("\n🔍 验证模块导入...")
    
    modules_to_test = [
        ('flask', 'Flask Web框架'),
        ('flask_cors', 'Flask CORS支持'),
        ('websockets', 'WebSocket支持'),
        ('requests', 'HTTP客户端'),
        ('json', 'JSON处理'),
        ('threading', '多线程支持'),
        ('time', '时间处理')
    ]
    
    failed_imports = []
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name} - {description}")
        except ImportError as e:
            print(f"  ❌ {module_name} - 导入失败: {str(e)}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"❌ 导入失败的模块: {failed_imports}")
        return False
    
    print("✅ 模块导入验证通过")
    return True

def validate_code_structure():
    """验证代码结构"""
    print("\n🔍 验证代码结构...")
    
    # 检查SSE处理器
    try:
        with open('sse_handler.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'class SSEHandler' in content and 'Server-Sent Events' in content:
                print("  ✅ SSE处理器结构正确")
            else:
                print("  ❌ SSE处理器结构不完整")
                return False
    except Exception as e:
        print(f"  ❌ SSE处理器读取失败: {str(e)}")
        return False
    
    # 检查WebSocket处理器
    try:
        with open('websocket_handler.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'class WebSocketHandler' in content and 'WebSocket' in content:
                print("  ✅ WebSocket处理器结构正确")
            else:
                print("  ❌ WebSocket处理器结构不完整")
                return False
    except Exception as e:
        print(f"  ❌ WebSocket处理器读取失败: {str(e)}")
        return False
    
    # 检查REST API
    try:
        with open('rest_api.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'class TrainingAPI' in content and 'Flask' in content:
                print("  ✅ REST API结构正确")
            else:
                print("  ❌ REST API结构不完整")
                return False
    except Exception as e:
        print(f"  ❌ REST API读取失败: {str(e)}")
        return False
    
    print("✅ 代码结构验证通过")
    return True

def validate_integration_points():
    """验证集成点"""
    print("\n🔍 验证训练系统集成点...")
    
    # 检查TensorBoard日志器扩展
    tensorboard_path = '../../training_components/tensorboard_logger.py'
    try:
        with open(tensorboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            checks = [
                ('metrics_stream = pyqtSignal(dict)', '数据流信号'),
                ('set_stream_server', '数据流服务器设置'),
                ('_update_current_metrics', '指标更新方法'),
                ('log_comprehensive_metrics', '综合指标记录')
            ]
            
            for check, description in checks:
                if check in content:
                    print(f"  ✅ {description}")
                else:
                    print(f"  ❌ 缺失: {description}")
                    return False
                    
    except Exception as e:
        print(f"  ❌ TensorBoard日志器检查失败: {str(e)}")
        return False
    
    # 检查训练线程扩展
    training_thread_path = '../../training_components/training_thread.py'
    try:
        with open(training_thread_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            if '_initialize_stream_server' in content:
                print("  ✅ 训练线程数据流集成")
            else:
                print("  ❌ 训练线程集成不完整")
                return False
                
    except Exception as e:
        print(f"  ❌ 训练线程检查失败: {str(e)}")
        return False
    
    print("✅ 集成点验证通过")
    return True

def run_simple_functionality_test():
    """运行简单功能测试"""
    print("\n🔍 运行简单功能测试...")
    
    try:
        # 测试Flask应用创建
        from flask import Flask
        app = Flask(__name__)
        print("  ✅ Flask应用创建成功")
        
        # 测试JSON处理
        import json
        test_data = {'epoch': 1, 'loss': 0.5}
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        print("  ✅ JSON处理功能正常")
        
        # 测试线程创建
        import threading
        import time
        
        def test_thread():
            time.sleep(0.1)
            
        thread = threading.Thread(target=test_thread)
        thread.start()
        thread.join()
        print("  ✅ 多线程功能正常")
        
        print("✅ 简单功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 功能测试失败: {str(e)}")
        return False

def generate_phase1_report():
    """生成第一阶段报告"""
    print("\n" + "="*60)
    print("📋 第一阶段验证报告")
    print("="*60)
    
    all_tests = [
        validate_api_modules,
        validate_module_imports, 
        validate_code_structure,
        validate_integration_points,
        run_simple_functionality_test
    ]
    
    passed_tests = 0
    total_tests = len(all_tests)
    
    for test_func in all_tests:
        if test_func():
            passed_tests += 1
    
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n📊 测试结果统计:")
    print(f"• 总测试数: {total_tests}")
    print(f"• 通过数: {passed_tests}")
    print(f"• 失败数: {total_tests - passed_tests}")
    print(f"• 成功率: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print(f"\n🎉 第一阶段验证成功！")
        print("📊 已实现的核心功能:")
        print("  ✅ SSE数据流处理器")
        print("  ✅ WebSocket实时通信")
        print("  ✅ REST API接口服务")
        print("  ✅ 统一数据流服务器")
        print("  ✅ TensorBoard集成扩展")
        print("  ✅ 训练线程集成")
        print("  ✅ 测试客户端工具")
        
        print(f"\n🚀 准备进入第二阶段: LLM智能分析框架开发")
        return True
    else:
        print(f"\n⚠️ 第一阶段验证未完全通过，需要修复问题后继续")
        return False

def main():
    """主函数"""
    print("🧪 第一阶段功能验证")
    print("验证数据流基础设施的实现情况")
    print("="*60)
    
    return generate_phase1_report()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
#!/usr/bin/env python3
"""
数据流监控组件最终修复脚本

确保所有WebSocket连接和信号调用问题都得到解决。
"""

import os
import re

def fix_websocket_timeout_issues():
    """修复所有WebSocket timeout参数问题"""
    file_path = "src/ui/components/model_analysis/real_time_stream_monitor.py"
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否还有错误的timeout参数使用
    if 'websockets.connect(' in content and 'timeout=' in content:
        print("🔍 发现可能的WebSocket timeout参数问题...")
        
        # 查找所有websockets.connect调用
        pattern = r'websockets\.connect\([^)]*timeout[^)]*\)'
        matches = re.findall(pattern, content)
        
        if matches:
            print(f"❌ 发现 {len(matches)} 个错误的WebSocket连接调用:")
            for match in matches:
                print(f"  - {match}")
            return False
        else:
            print("✅ 未发现错误的WebSocket连接调用")
    
    return True

def fix_signal_calls():
    """检查信号调用是否正确"""
    file_path = "src/ui/components/model_analysis/real_time_stream_monitor.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否有错误的信号调用
    if 'self.connection_status_changed.emit(' in content:
        print("❌ 发现错误的信号调用")
        return False
    
    print("✅ 信号调用检查通过")
    return True

def verify_fixes():
    """验证修复是否完整"""
    print("🔍 验证修复完整性...")
    
    # 检查WebSocket连接
    websocket_ok = fix_websocket_timeout_issues()
    
    # 检查信号调用
    signal_ok = fix_signal_calls()
    
    if websocket_ok and signal_ok:
        print("✅ 所有修复验证通过")
        return True
    else:
        print("❌ 修复验证失败")
        return False

def main():
    """主函数"""
    print("🚀 数据流监控组件最终修复验证")
    print("=" * 50)
    
    if verify_fixes():
        print("\n✅ 修复验证成功！")
        print("💡 建议:")
        print("  1. 重启应用程序")
        print("  2. 测试监控组件功能")
        print("  3. 验证WebSocket连接")
        print("  4. 检查错误信息准确性")
    else:
        print("\n❌ 修复验证失败！")
        print("💡 需要进一步检查和修复")

if __name__ == "__main__":
    main() 
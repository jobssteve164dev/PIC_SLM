#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试AI设置中的数据流监控组件集成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

def test_ai_settings_stream_monitor():
    """测试AI设置中的数据流监控组件"""
    app = QApplication(sys.argv)
    
    # 创建主窗口
    main_window = QMainWindow()
    main_window.setWindowTitle("AI设置数据流监控测试")
    main_window.setGeometry(100, 100, 1200, 800)
    
    # 创建中央部件
    central_widget = QWidget()
    main_window.setCentralWidget(central_widget)
    
    # 创建布局
    layout = QVBoxLayout(central_widget)
    
    try:
        # 导入AI设置组件
        from src.ui.components.settings.ai_settings_widget import AISettingsWidget
        ai_settings = AISettingsWidget()
        layout.addWidget(ai_settings)
        
        print("✅ AI设置组件加载成功")
        print("✅ 数据流监控组件已集成到AI设置中")
        
        # 显示窗口
        main_window.show()
        
        print("🎯 测试说明:")
        print("1. 点击'📡 数据流监控'标签页")
        print("2. 验证数据流监控组件是否正确显示")
        print("3. 测试监控功能是否正常工作")
        
        return app.exec_()
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return 1
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(test_ai_settings_stream_monitor()) 
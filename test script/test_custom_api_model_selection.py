#!/usr/bin/env python3
"""
测试自定义API模型选择功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from src.ui.components.settings.ai_settings_widget import AISettingsWidget

def test_custom_api_model_selection():
    """测试自定义API模型选择功能"""
    app = QApplication(sys.argv)
    
    # 创建主窗口
    main_window = QMainWindow()
    main_window.setWindowTitle("自定义API模型选择测试")
    main_window.setGeometry(100, 100, 800, 600)
    
    # 创建中央部件
    central_widget = QWidget()
    main_window.setCentralWidget(central_widget)
    
    # 创建布局
    layout = QVBoxLayout(central_widget)
    
    # 创建AI设置组件
    ai_settings = AISettingsWidget()
    layout.addWidget(ai_settings)
    
    # 显示窗口
    main_window.show()
    
    print("测试说明:")
    print("1. 切换到'自定义API'标签页")
    print("2. 输入API密钥和基础URL")
    print("3. 点击'🔍 测试连接'按钮")
    print("4. 应该会弹出模型选择对话框")
    print("5. 选择一个模型进行测试")
    
    return app.exec_()

if __name__ == "__main__":
    test_custom_api_model_selection() 
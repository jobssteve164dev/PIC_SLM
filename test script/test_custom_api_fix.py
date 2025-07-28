#!/usr/bin/env python3
"""
测试自定义API测试修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
from src.ui.components.settings.ai_settings_widget import CustomAPITestThread

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("自定义API测试修复验证")
        self.setGeometry(100, 100, 600, 400)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)
        
        # 测试按钮
        self.test_btn = QPushButton("测试自定义API连接")
        self.test_btn.clicked.connect(self.test_custom_api)
        layout.addWidget(self.test_btn)
        
        # 状态标签
        self.status_label = QLabel("点击按钮开始测试")
        layout.addWidget(self.status_label)
        
        # 测试线程
        self.test_thread = None
    
    def test_custom_api(self):
        """测试自定义API连接"""
        self.test_btn.setEnabled(False)
        self.status_label.setText("正在测试...")
        
        # 使用OpenRouter作为测试
        api_key = "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"
        base_url = "https://openrouter.ai/api/v1"
        
        self.test_thread = CustomAPITestThread(api_key, base_url, "OpenAI兼容")
        self.test_thread.test_completed.connect(self.on_test_completed)
        self.test_thread.model_selection_needed.connect(self.on_model_selection_needed)
        self.test_thread.start()
    
    def on_model_selection_needed(self, models):
        """需要选择模型"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
        
        print(f"获取到 {len(models)} 个模型，需要用户选择")
        
        # 创建模型选择对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择测试模型")
        dialog.setModal(True)
        dialog.setFixedSize(400, 150)
        
        layout = QVBoxLayout(dialog)
        
        # 说明文字
        info_label = QLabel(f"已获取到 {len(models)} 个可用模型，请选择一个进行连接测试：")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 模型选择下拉框
        model_combo = QComboBox()
        model_combo.addItems(models)
        layout.addWidget(model_combo)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 取消按钮
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        # 测试按钮
        test_btn = QPushButton("开始测试")
        test_btn.setDefault(True)
        button_layout.addWidget(test_btn)
        
        layout.addLayout(button_layout)
        
        # 连接测试按钮信号
        def start_test():
            selected_model = model_combo.currentText()
            dialog.accept()
            print(f"用户选择了模型: {selected_model}")
            self.test_thread.set_selected_model(selected_model, models)
            self.status_label.setText("正在测试连接...")
        
        test_btn.clicked.connect(start_test)
        
        # 显示对话框
        if dialog.exec_() == QDialog.Accepted:
            pass
        else:
            # 用户取消
            self.test_btn.setEnabled(True)
            self.status_label.setText("测试已取消")
    
    def on_test_completed(self, success, message, models):
        """测试完成"""
        self.test_btn.setEnabled(True)
        if success:
            self.status_label.setText(f"✅ {message}")
        else:
            self.status_label.setText(f"❌ {message}")
        print(f"测试完成: {message}")

def main():
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    return app.exec_()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试模型评估组件的悬停功能和表格显示修复
"""

import sys
import os
sys.path.append('src')

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
    from PyQt5.QtCore import Qt
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import numpy as np
    
    print("✓ 所有必要的库导入成功")
    
    class TestWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.init_ui()
            
        def init_ui(self):
            layout = QVBoxLayout(self)
            
            # 创建测试图表
            self.figure = Figure(figsize=(10, 6))
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas)
            
            # 创建测试数据
            self.create_test_chart()
            
        def create_test_chart(self):
            """创建测试图表"""
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # 测试数据
            model_names = ['DenseNet201_test1', 'MobileNetV2_test2']
            values = [0.85, 0.78]
            
            bars = ax.bar(range(len(model_names)), values)
            ax.set_title('测试悬停功能')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45)
            
            # 添加悬停功能测试
            def on_hover(event):
                if event.inaxes == ax and event.xdata is not None:
                    for i, bar in enumerate(bars):
                        contains, info = bar.contains(event)
                        if contains:
                            print(f"悬停检测成功: 模型 {model_names[i]}, 值 {values[i]:.3f}")
                            return
            
            # 连接事件
            self.canvas.mpl_connect('motion_notify_event', on_hover)
            self.canvas.draw()
    
    def main():
        app = QApplication(sys.argv)
        
        window = QMainWindow()
        widget = TestWidget()
        window.setCentralWidget(widget)
        window.setWindowTitle('悬停功能测试')
        window.resize(800, 600)
        window.show()
        
        print("✓ 测试窗口创建成功")
        print("请移动鼠标到柱状图上测试悬停功能")
        print("按Ctrl+C退出测试")
        
        try:
            sys.exit(app.exec_())
        except KeyboardInterrupt:
            print("\n测试结束")
            
    if __name__ == '__main__':
        main()
        
except ImportError as e:
    print(f"✗ 导入失败: {e}")
except Exception as e:
    print(f"✗ 测试失败: {e}") 
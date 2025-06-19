#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强模型评估功能演示脚本

这个脚本展示了新增的增强模型评估功能，包括：
1. 精确率 (Precision)
2. 召回率 (Recall) 
3. F1分数 (F1-Score)
4. AUC分数 (Area Under Curve)
5. 混淆矩阵 (Confusion Matrix)
6. 详细分类报告
7. 每个类别的详细指标

使用方法：
1. 确保已有训练好的模型文件
2. 确保有验证数据集
3. 运行此脚本启动增强评估界面
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

# 添加src目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from ui.components.evaluation.widgets.enhanced_model_evaluation_widget import EnhancedModelEvaluationWidget
    print("✓ 增强模型评估组件导入成功")
except ImportError as e:
    print(f"✗ 导入增强模型评估组件失败: {e}")
    sys.exit(1)


class EnhancedEvaluationDemo(QMainWindow):
    """增强评估功能演示窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("增强模型评估功能演示 - 图片分类模型训练系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)
        
        # 创建增强评估组件
        self.enhanced_eval_widget = EnhancedModelEvaluationWidget()
        
        # 连接状态更新信号
        self.enhanced_eval_widget.status_updated.connect(self.update_status)
        
        layout.addWidget(self.enhanced_eval_widget)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("增强模型评估功能已就绪")
        
        print("增强评估演示窗口初始化完成")
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_bar.showMessage(message)
        print(f"状态更新: {message}")


def main():
    """主函数"""
    print("="*60)
    print("增强模型评估功能演示")
    print("="*60)
    print()
    
    print("功能特点:")
    print("• 精确率、召回率、F1分数等详细指标")
    print("• 混淆矩阵可视化")
    print("• 每个类别的详细性能分析")
    print("• AUC和平均精度分数")
    print("• 完整的分类报告")
    print("• 多线程评估，支持进度显示")
    print()
    
    print("使用说明:")
    print("1. 点击'浏览...'选择包含.pth模型文件的目录")
    print("2. 点击'刷新'更新模型列表")
    print("3. 选择要评估的模型")
    print("4. 点击'评估选中模型'开始评估")
    print("5. 在不同标签页查看详细结果")
    print()
    
    print("注意事项:")
    print("• 确保config.json中配置了正确的数据集路径")
    print("• 确保存在类别信息文件(class_info.json)")
    print("• 评估过程可能需要一些时间，请耐心等待")
    print()
    
    # 创建应用
    app = QApplication(sys.argv)
    
    # 创建主窗口
    demo_window = EnhancedEvaluationDemo()
    demo_window.show()
    
    print("演示窗口已启动，您可以开始使用增强评估功能")
    print("关闭窗口退出演示")
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 
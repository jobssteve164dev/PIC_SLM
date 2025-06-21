#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建应用程序图标
"""

import os
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QBrush
from PyQt5.QtCore import Qt

def create_app_icon():
    """创建应用程序图标"""
    # 创建QApplication实例（如果不存在）
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # 创建64x64的pixmap
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    
    # 创建画笔
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # 绘制背景圆形
    margin = 4
    painter.setPen(QPen(QColor(30, 90, 140), 2))
    painter.setBrush(QBrush(QColor(70, 130, 180)))
    painter.drawEllipse(margin, margin, size-margin*2, size-margin*2)
    
    # 绘制相机图标
    # 相机主体
    cam_x = size // 4
    cam_y = size // 3
    cam_w = size // 2
    cam_h = size // 3
    painter.setPen(QPen(QColor(200, 200, 200), 1))
    painter.setBrush(QBrush(QColor(255, 255, 255)))
    painter.drawRect(cam_x, cam_y, cam_w, cam_h)
    
    # 相机镜头
    lens_center_x = size // 2
    lens_center_y = size // 2
    lens_radius = size // 6
    painter.setPen(QPen(QColor(30, 30, 30), 1))
    painter.setBrush(QBrush(QColor(50, 50, 50)))
    painter.drawEllipse(lens_center_x - lens_radius, lens_center_y - lens_radius,
                       lens_radius * 2, lens_radius * 2)
    
    # 镜头中心点
    center_radius = 3
    painter.setBrush(QBrush(QColor(100, 100, 100)))
    painter.drawEllipse(lens_center_x - center_radius, lens_center_y - center_radius,
                       center_radius * 2, center_radius * 2)
    
    painter.end()
    
    # 保存图标
    icon_dir = os.path.join(os.path.dirname(__file__), 'icons')
    os.makedirs(icon_dir, exist_ok=True)
    
    # 保存为PNG文件（ICO需要额外的库）
    icon_path = os.path.join(icon_dir, 'app.png')
    pixmap.save(icon_path, 'PNG')
    
    print(f"图标已创建: {icon_path}")
    return icon_path

if __name__ == '__main__':
    try:
        create_app_icon()
        print("图标创建成功！")
    except Exception as e:
        print(f"创建图标失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
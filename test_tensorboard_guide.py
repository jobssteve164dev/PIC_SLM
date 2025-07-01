#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append('src')

try:
    from PyQt5.QtWidgets import QApplication
    from ui.components.evaluation.widgets.tensorboard_manager_widget import TensorBoardParameterGuideWidget, TensorBoardManagerWidget
    
    app = QApplication(sys.argv)
    
    # 测试参数指南组件
    guide_widget = TensorBoardParameterGuideWidget()
    print('✓ TensorBoard参数指南组件创建成功')
    
    # 测试管理组件
    manager_widget = TensorBoardManagerWidget()
    print('✓ TensorBoard管理组件创建成功')
    
    print('✓ 所有组件测试通过')
    print('✓ TensorBoard参数监控说明功能已成功集成')
    
except Exception as e:
    print(f'✗ 测试失败: {e}')
    import traceback
    traceback.print_exc() 
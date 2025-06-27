from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                           QHBoxLayout, QStackedWidget)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os
import sys

# 添加src目录到路径以便导入组件
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from .base_tab import BaseTab
from .components.evaluation import (
    TrainingCurveWidget,
    TensorBoardManagerWidget,
    EnhancedModelEvaluationWidget,
    ParamsComparisonWidget,
    VisualizationContainerWidget
)


class EvaluationTab(BaseTab):
    """重构后的评估标签页，使用组件化架构"""
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.main_window = main_window
        
        # 初始化各个组件
        self.training_curve_widget = None
        self.tensorboard_widget = None
        self.enhanced_model_eval_widget = None
        self.params_compare_widget = None
        self.visualization_container = None
        
        self.init_ui()
        
        # 使用新的智能配置系统
        config = self.get_config_from_manager()
        if config:
            self.apply_config(config)
        
    def init_ui(self):
        """初始化UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self.scroll_content)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 添加标题
        title_label = QLabel("模型评估与可视化")
        title_label.setFont(QFont('微软雅黑', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 创建切换按钮组
        switch_layout = QHBoxLayout()
        
        # 实时训练曲线按钮
        self.training_curve_btn = QPushButton("实时训练曲线")
        self.training_curve_btn.setCheckable(True)
        self.training_curve_btn.setChecked(True)
        self.training_curve_btn.clicked.connect(lambda: self.switch_view(0))
        switch_layout.addWidget(self.training_curve_btn)
        
        # TensorBoard可视化按钮
        self.tb_btn = QPushButton("TensorBoard可视化")
        self.tb_btn.setCheckable(True)
        self.tb_btn.clicked.connect(lambda: self.switch_view(1))
        switch_layout.addWidget(self.tb_btn)
        
        # 训练参数对比按钮
        self.params_compare_btn = QPushButton("训练参数对比")
        self.params_compare_btn.setCheckable(True)
        self.params_compare_btn.clicked.connect(lambda: self.switch_view(2))
        switch_layout.addWidget(self.params_compare_btn)
        
        # 模型评估按钮
        self.eval_btn = QPushButton("模型评估")
        self.eval_btn.setCheckable(True)
        self.eval_btn.clicked.connect(lambda: self.switch_view(3))
        self.eval_btn.setToolTip("完整的模型性能评估，包含准确率、精确率、召回率、F1分数、混淆矩阵等专业指标")
        switch_layout.addWidget(self.eval_btn)
        
        # 模型结构可视化按钮
        self.model_structure_btn = QPushButton("模型结构")
        self.model_structure_btn.setCheckable(True)
        self.model_structure_btn.clicked.connect(lambda: self.switch_view(4))
        switch_layout.addWidget(self.model_structure_btn)
        
        main_layout.addLayout(switch_layout)
        
        # 创建堆叠小部件用于切换视图
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # 初始化各个组件并添加到堆叠小部件
        self.init_components()
        
        # 添加弹性空间
        main_layout.addStretch()
    
    def init_components(self):
        """初始化所有组件"""
        try:
            # 创建实时训练曲线组件
            self.training_curve_widget = TrainingCurveWidget()
            self.training_curve_widget.status_updated.connect(self.update_status)
            self.stacked_widget.addWidget(self.training_curve_widget)
            
            # 创建TensorBoard管理组件
            self.tensorboard_widget = TensorBoardManagerWidget(main_window=self.main_window)
            self.tensorboard_widget.status_updated.connect(self.update_status)
            self.stacked_widget.addWidget(self.tensorboard_widget)
            
            # 创建训练参数对比组件
            self.params_compare_widget = ParamsComparisonWidget(main_window=self.main_window)
            self.params_compare_widget.status_updated.connect(self.update_status)
            self.stacked_widget.addWidget(self.params_compare_widget)
            
            # 创建模型评估组件
            self.enhanced_model_eval_widget = EnhancedModelEvaluationWidget(main_window=self.main_window)
            self.enhanced_model_eval_widget.status_updated.connect(self.update_status)
            self.stacked_widget.addWidget(self.enhanced_model_eval_widget)
            
            # 创建可视化容器组件（仅用于模型结构可视化）
            self.visualization_container = VisualizationContainerWidget()
            self.visualization_container.status_updated.connect(self.update_status)
            # 添加模型结构可视化组件
            self.stacked_widget.addWidget(self.visualization_container.get_model_structure_widget())
            
        except Exception as e:
            import traceback
            print(f"初始化组件时出错: {str(e)}")
            print(traceback.format_exc())
            self.update_status(f"初始化组件失败: {str(e)}")
    
    def switch_view(self, index):
        """切换视图"""
        # 取消所有按钮的选中状态
        buttons = [
            self.training_curve_btn, self.tb_btn, self.params_compare_btn,
            self.eval_btn, self.model_structure_btn
        ]
        
        for btn in buttons:
            btn.setChecked(False)
        
        # 根据索引选中相应按钮
        if 0 <= index < len(buttons):
            buttons[index].setChecked(True)
        
        # 切换视图
        self.stacked_widget.setCurrentIndex(index)
        
        # 当切换到参数对比页面时，刷新参数列表
        if index == 2 and self.params_compare_widget:
            # 触发显示事件来刷新参数列表
            if hasattr(self.params_compare_widget, 'showEvent'):
                from PyQt5.QtGui import QShowEvent
                self.params_compare_widget.showEvent(QShowEvent())
    
    def update_training_visualization(self, data):
        """更新训练可视化（兼容原有接口）"""
        if self.training_curve_widget:
            self.training_curve_widget.update_training_visualization(data)
    
    def reset_training_visualization(self):
        """重置训练可视化（兼容原有接口）"""
        if self.training_curve_widget:
            self.training_curve_widget.reset_training_visualization()
    
    def setup_trainer(self, trainer):
        """设置训练器并连接信号（兼容原有接口）"""
        if self.training_curve_widget:
            return self.training_curve_widget.setup_trainer(trainer)
        return False
    
    def set_model(self, model, class_names=None):
        """设置模型，用于模型结构可视化组件"""
        if self.visualization_container:
            self.visualization_container.set_model(model, class_names)
    
    def _do_apply_config(self, config):
        """实现具体的配置应用逻辑 - 智能配置系统"""
        print(f"EvaluationTab: 智能应用配置，包含 {len(config)} 个配置项")
        
        try:
            # 为各个组件应用配置
            if self.enhanced_model_eval_widget:
                self.enhanced_model_eval_widget.apply_config(config)
                
            if self.tensorboard_widget:
                self.tensorboard_widget.apply_config(config)
                
            if self.params_compare_widget:
                self.params_compare_widget.apply_config(config)
                
            if self.visualization_container:
                self.visualization_container.apply_config(config)
                
            print("EvaluationTab: 智能配置应用完成")
            
        except Exception as e:
            import traceback
            print(f"应用配置时出错: {str(e)}")
            print(traceback.format_exc())
    
    def go_to_params_compare_tab(self):
        """切换到训练参数对比视图（兼容原有接口）"""
        self.switch_view(2)
    
    def select_models_dir(self):
        """选择模型目录（兼容原有接口）"""
        if self.enhanced_model_eval_widget:
            self.enhanced_model_eval_widget.select_models_dir()
    
    def refresh_model_list(self):
        """刷新模型列表（兼容原有接口）"""
        if self.enhanced_model_eval_widget:
            self.enhanced_model_eval_widget.refresh_model_list()
    
    def compare_models(self):
        """比较选中的模型（兼容原有接口）"""
        if self.enhanced_model_eval_widget:
            self.enhanced_model_eval_widget.compare_models()
    
    def select_log_dir(self):
        """选择TensorBoard日志目录（兼容原有接口）"""
        if self.tensorboard_widget:
            self.tensorboard_widget.select_log_dir()
    
    def start_tensorboard(self):
        """启动TensorBoard（兼容原有接口）"""
        if self.tensorboard_widget:
            self.tensorboard_widget.start_tensorboard()
    
    def stop_tensorboard(self):
        """停止TensorBoard（兼容原有接口）"""
        if self.tensorboard_widget:
            self.tensorboard_widget.stop_tensorboard()
    
    def browse_param_dir(self):
        """浏览参数目录（兼容原有接口）"""
        if self.params_compare_widget:
            self.params_compare_widget.browse_param_dir()
    
    def load_model_configs(self):
        """加载模型配置文件（兼容原有接口）"""
        if self.params_compare_widget:
            self.params_compare_widget.load_model_configs()
    
    def select_all_models(self):
        """选择所有模型（兼容原有接口）"""
        if self.params_compare_widget:
            self.params_compare_widget.select_all_models()
    
    def deselect_all_models(self):
        """取消选择所有模型（兼容原有接口）"""
        if self.params_compare_widget:
            self.params_compare_widget.deselect_all_models()
    
    def compare_params(self):
        """比较参数（兼容原有接口）"""
        if self.params_compare_widget:
            self.params_compare_widget.compare_params()
    
    def showEvent(self, event):
        """当标签页显示时的事件处理（兼容原有接口）"""
        super().showEvent(event)
        
        # 获取当前标签页索引
        current_index = self.stacked_widget.currentIndex()
        
        # 如果当前是参数对比页面，刷新参数列表
        if current_index == 2 and self.params_compare_widget:
            self.params_compare_widget.showEvent(event)
    
    def closeEvent(self, event):
        """窗口关闭事件（兼容原有接口）"""
        # 确保在关闭窗口时停止TensorBoard进程
        if self.tensorboard_widget:
            self.tensorboard_widget.closeEvent(event)
        super().closeEvent(event)
        
    def __del__(self):
        """析构方法（兼容原有接口）"""
        try:
            if self.tensorboard_widget:
                self.tensorboard_widget.__del__()
        except:
            # 在析构时忽略异常
            pass
    
    # 为了完全兼容原有接口，添加一些属性访问器
    @property
    def models_dir(self):
        """获取模型目录"""
        if self.enhanced_model_eval_widget:
            return self.enhanced_model_eval_widget.models_dir
        return ""
    
    @models_dir.setter
    def models_dir(self, value):
        """设置模型目录"""
        if self.enhanced_model_eval_widget:
            self.enhanced_model_eval_widget.models_dir = value
    
    @property
    def models_list(self):
        """获取模型列表"""
        if self.enhanced_model_eval_widget:
            return self.enhanced_model_eval_widget.models_list
        return []
    
    @property
    def log_dir(self):
        """获取日志目录"""
        if self.tensorboard_widget:
            return self.tensorboard_widget.log_dir
        return ""
    
    @log_dir.setter
    def log_dir(self, value):
        """设置日志目录"""
        if self.tensorboard_widget:
            self.tensorboard_widget.log_dir = value
    
    @property
    def tensorboard_process(self):
        """获取TensorBoard进程"""
        if self.tensorboard_widget:
            return self.tensorboard_widget.tensorboard_process
        return None
    
    @property
    def model_dir(self):
        """获取参数目录"""
        if self.params_compare_widget:
            return self.params_compare_widget.model_dir
        return ""
    
    @model_dir.setter
    def model_dir(self, value):
        """设置参数目录"""
        if self.params_compare_widget:
            self.params_compare_widget.model_dir = value
    
    @property
    def model_configs(self):
        """获取模型配置"""
        if self.params_compare_widget:
            return self.params_compare_widget.model_configs
        return []
    
    # 可视化组件的访问器 - 只保留模型结构组件
    @property
    def model_structure_widget(self):
        """获取模型结构组件"""
        if self.visualization_container:
            return self.visualization_container.get_model_structure_widget()
        return None
    
    @property
    def training_visualization(self):
        """获取训练可视化组件"""
        if self.training_curve_widget:
            return self.training_curve_widget.training_visualization
        return None 
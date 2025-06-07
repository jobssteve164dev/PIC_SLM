"""
类别权重配置组件 - 负责管理类别权重配置
"""

from typing import Dict, List, Tuple
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                           QLabel, QComboBox, QTableWidget, QTableWidgetItem,
                           QPushButton, QDoubleSpinBox, QInputDialog, QMessageBox,
                           QFileDialog, QFrame, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import pyqtSignal
from .weight_strategy import WeightStrategy
from .config_manager import ConfigManager


class ClassWeightWidget(QWidget):
    """类别权重配置组件"""
    
    # 定义信号
    classes_changed = pyqtSignal(list)  # 类别列表变化
    weights_changed = pyqtSignal(dict)  # 权重字典变化
    strategy_changed = pyqtSignal(object)  # 权重策略变化
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.default_classes = []
        self.class_weights = {}
        self.current_strategy = WeightStrategy.BALANCED
        self.config_manager = ConfigManager()
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建类别权重配置组
        classes_group = QGroupBox("默认缺陷类别与权重配置")
        classes_layout = QVBoxLayout()
        classes_layout.setContentsMargins(10, 20, 10, 10)
        
        # 添加权重策略选择
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("权重策略:"))
        
        self.weight_strategy_combo = QComboBox()
        self.weight_strategy_combo.addItems(WeightStrategy.get_all_display_names())
        self.weight_strategy_combo.setCurrentText(WeightStrategy.BALANCED.display_name)
        self.weight_strategy_combo.currentTextChanged.connect(self.on_weight_strategy_changed)
        strategy_layout.addWidget(self.weight_strategy_combo)
        strategy_layout.addStretch()
        
        classes_layout.addLayout(strategy_layout)
        
        # 添加说明标签
        info_label = QLabel("说明: balanced自动平衡权重, inverse逆频率权重, log_inverse对数逆频率权重, custom使用自定义权重, none不使用权重")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        classes_layout.addWidget(info_label)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        classes_layout.addWidget(line)
        
        # 添加类别权重表格
        self.class_weight_table = QTableWidget()
        self.class_weight_table.setColumnCount(2)
        self.class_weight_table.setHorizontalHeaderLabels(["类别名称", "权重值"])
        self.class_weight_table.horizontalHeader().setStretchLastSection(True)
        self.class_weight_table.setMinimumHeight(200)
        self.class_weight_table.setAlternatingRowColors(True)
        classes_layout.addWidget(self.class_weight_table)
        
        # 添加按钮组
        btn_layout = QHBoxLayout()
        
        add_class_btn = QPushButton("添加类别")
        add_class_btn.clicked.connect(self.add_defect_class)
        btn_layout.addWidget(add_class_btn)
        
        remove_class_btn = QPushButton("删除类别")
        remove_class_btn.clicked.connect(self.remove_defect_class)
        btn_layout.addWidget(remove_class_btn)
        
        # 添加重置权重按钮
        reset_weights_btn = QPushButton("重置权重")
        reset_weights_btn.clicked.connect(self.reset_class_weights)
        btn_layout.addWidget(reset_weights_btn)
        
        btn_layout.addSpacerItem(QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # 添加保存到文件按钮
        save_to_file_btn = QPushButton("保存到文件")
        save_to_file_btn.clicked.connect(self.save_classes_to_file)
        btn_layout.addWidget(save_to_file_btn)
        
        # 添加从文件加载按钮
        load_from_file_btn = QPushButton("从文件加载")
        load_from_file_btn.clicked.connect(self.load_classes_from_file)
        btn_layout.addWidget(load_from_file_btn)
        
        classes_layout.addLayout(btn_layout)
        classes_group.setLayout(classes_layout)
        layout.addWidget(classes_group)
    
    def add_defect_class(self):
        """添加缺陷类别"""
        class_name, ok = QInputDialog.getText(self, "添加缺陷类别", "请输入缺陷类别名称:")
        if ok and class_name:
            # 检查是否已存在
            if class_name in self.default_classes:
                QMessageBox.warning(self, "警告", f"类别 '{class_name}' 已存在!")
                return
                
            self.default_classes.append(class_name)
            # 为新类别设置默认权重
            self.class_weights[class_name] = 1.0
            
            # 添加到表格
            self._add_class_to_table(class_name, 1.0)
            self._update_weight_widgets_state()
            
            # 发送信号
            self.classes_changed.emit(self.default_classes)
            self.weights_changed.emit(self.class_weights)
    
    def remove_defect_class(self):
        """删除缺陷类别"""
        current_row = self.class_weight_table.currentRow()
        if current_row >= 0:
            class_name_item = self.class_weight_table.item(current_row, 0)
            if class_name_item:
                class_name = class_name_item.text()
                
                # 从列表和权重字典中移除
                if class_name in self.default_classes:
                    self.default_classes.remove(class_name)
                if class_name in self.class_weights:
                    del self.class_weights[class_name]
                
                # 从表格中移除
                self.class_weight_table.removeRow(current_row)
                
                # 发送信号
                self.classes_changed.emit(self.default_classes)
                self.weights_changed.emit(self.class_weights)
    
    def _add_class_to_table(self, class_name: str, weight_value: float):
        """添加类别到表格"""
        row_count = self.class_weight_table.rowCount()
        self.class_weight_table.insertRow(row_count)
        self.class_weight_table.setItem(row_count, 0, QTableWidgetItem(class_name))
        
        # 创建权重输入框
        weight_spinbox = QDoubleSpinBox()
        weight_spinbox.setMinimum(0.01)
        weight_spinbox.setMaximum(100.0)
        weight_spinbox.setSingleStep(0.1)
        weight_spinbox.setDecimals(2)
        weight_spinbox.setValue(weight_value)
        weight_spinbox.valueChanged.connect(lambda value, name=class_name: self.on_weight_changed(name, value))
        
        self.class_weight_table.setCellWidget(row_count, 1, weight_spinbox)
    
    def on_weight_changed(self, class_name: str, value: float):
        """处理权重值变化"""
        self.class_weights[class_name] = value
        self.weights_changed.emit(self.class_weights)
    
    def on_weight_strategy_changed(self):
        """处理权重策略选择变化"""
        strategy_text = self.weight_strategy_combo.currentText()
        self.current_strategy = WeightStrategy.from_display_name(strategy_text)
        self._update_weight_widgets_state()
        
        # 如果选择了自定义权重策略，显示提示
        if self.current_strategy.is_custom():
            QMessageBox.information(
                self, 
                "自定义权重", 
                "您选择了自定义权重策略。\n请在下表中设置每个类别的权重值。\n较高的权重值会让模型更关注该类别的样本。"
            )
        
        # 发送信号
        self.strategy_changed.emit(self.current_strategy)
    
    def _update_weight_widgets_state(self):
        """根据权重策略更新权重输入框的状态"""
        is_custom = self.current_strategy.is_custom()
        
        # 启用或禁用权重输入框
        for row in range(self.class_weight_table.rowCount()):
            weight_widget = self.class_weight_table.cellWidget(row, 1)
            if weight_widget:
                weight_widget.setEnabled(is_custom)
    
    def reset_class_weights(self):
        """重置类别权重"""
        reply = QMessageBox.question(
            self, 
            "重置权重", 
            "确定要重置所有类别权重为1.0吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 重置所有权重为1.0
            for class_name in self.default_classes:
                self.class_weights[class_name] = 1.0
            
            # 更新表格中的权重显示
            for row in range(self.class_weight_table.rowCount()):
                weight_widget = self.class_weight_table.cellWidget(row, 1)
                if weight_widget:
                    weight_widget.setValue(1.0)
                    
            QMessageBox.information(self, "完成", "已重置所有类别权重为1.0")
            
            # 发送信号
            self.weights_changed.emit(self.class_weights)
    
    def save_classes_to_file(self):
        """保存类别配置到文件"""
        try:
            # 从表格收集最新数据
            self._collect_weights_from_table()
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "保存类别配置文件", 
                "defect_classes_config.json", 
                "JSON文件 (*.json)"
            )
            
            if file_path:
                success = self.config_manager.save_classes_config_to_file(
                    self.default_classes,
                    self.class_weights,
                    self.current_strategy,
                    file_path
                )
                
                if success:
                    QMessageBox.information(
                        self, 
                        "保存成功", 
                        f"类别配置已保存到:\n{file_path}\n\n"
                        f"包含 {len(self.default_classes)} 个类别\n"
                        f"权重策略: {self.current_strategy.value}"
                    )
                else:
                    QMessageBox.critical(self, "保存失败", "保存类别配置文件时出错")
                
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存类别配置文件时出错:\n{str(e)}")

    def load_classes_from_file(self):
        """从文件加载类别配置"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "加载类别配置文件", 
                "", 
                "文本文件 (*.txt);;JSON文件 (*.json);;所有文件 (*)"
            )
            
            if not file_path:
                return
            
            # 询问是否替换现有类别
            replace_existing = True
            if self.default_classes:
                reply = QMessageBox.question(
                    self, 
                    "加载确认", 
                    "是否替换现有的类别配置？\n选择'是'将替换所有现有类别，选择'否'将添加到现有类别。",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                
                if reply == QMessageBox.Cancel:
                    return
                    
                replace_existing = reply == QMessageBox.Yes
            
            # 加载类别配置
            loaded_classes, loaded_weights, loaded_strategy = self.config_manager.load_classes_from_file(file_path)
            
            # 根据用户选择处理现有类别
            if replace_existing:
                # 替换现有类别
                self.default_classes.clear()
                self.class_weights.clear()
                self.class_weight_table.setRowCount(0)
            
            # 添加新类别
            added_count = 0
            for class_name in loaded_classes:
                if class_name not in self.default_classes:
                    self.default_classes.append(class_name)
                    # 使用加载的权重或默认权重
                    weight_value = loaded_weights.get(class_name, 1.0)
                    self.class_weights[class_name] = weight_value
                    
                    # 添加到表格
                    self._add_class_to_table(class_name, weight_value)
                    added_count += 1
            
            # 设置权重策略
            self.current_strategy = loaded_strategy
            self.weight_strategy_combo.setCurrentText(loaded_strategy.display_name)
            
            # 更新权重输入框状态
            self._update_weight_widgets_state()
            
            action_text = "替换" if replace_existing else "添加"
            QMessageBox.information(
                self, 
                "加载成功", 
                f"成功{action_text}了 {added_count} 个类别\n"
                f"权重策略: {loaded_strategy.value}\n"
                f"当前总类别数: {len(self.default_classes)}"
            )
            
            # 发送信号
            self.classes_changed.emit(self.default_classes)
            self.weights_changed.emit(self.class_weights)
            self.strategy_changed.emit(self.current_strategy)
                
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载类别配置文件时出错:\n{str(e)}")
    
    def _collect_weights_from_table(self):
        """从表格收集权重数据"""
        self.class_weights.clear()
        self.default_classes.clear()
        
        for row in range(self.class_weight_table.rowCount()):
            class_name_item = self.class_weight_table.item(row, 0)
            if class_name_item:
                class_name = class_name_item.text()
                self.default_classes.append(class_name)
                
                # 获取权重值
                weight_widget = self.class_weight_table.cellWidget(row, 1)
                if weight_widget:
                    weight_value = weight_widget.value()
                    self.class_weights[class_name] = weight_value
    
    def get_classes_config(self) -> Tuple[List[str], Dict[str, float], WeightStrategy]:
        """获取类别配置"""
        self._collect_weights_from_table()
        return self.default_classes, self.class_weights, self.current_strategy
    
    def set_classes_config(self, classes: List[str], weights: Dict[str, float], strategy: WeightStrategy):
        """设置类别配置"""
        # 清空表格
        self.class_weight_table.setRowCount(0)
        
        # 设置数据
        self.default_classes = classes.copy()
        self.class_weights = weights.copy()
        self.current_strategy = strategy
        
        # 更新UI
        self.weight_strategy_combo.setCurrentText(strategy.display_name)
        
        # 填充表格
        for class_name in self.default_classes:
            weight_value = self.class_weights.get(class_name, 1.0)
            self._add_class_to_table(class_name, weight_value)
        
        # 更新权重输入框状态
        self._update_weight_widgets_state()
    
    def clear_config(self):
        """清空配置"""
        self.default_classes.clear()
        self.class_weights.clear()
        self.class_weight_table.setRowCount(0)
        self.current_strategy = WeightStrategy.BALANCED
        self.weight_strategy_combo.setCurrentText(WeightStrategy.BALANCED.display_name) 
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QListWidget, QGroupBox, QGridLayout,
                           QLineEdit, QInputDialog, QMessageBox, QComboBox, QScrollArea,
                           QSplitter, QCheckBox, QDoubleSpinBox, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os
import glob
from .annotation_canvas import AnnotationCanvas

class DetectionWidget(QWidget):
    """目标检测标注组件"""
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.detection_classes = []  # 目标检测类别列表
        self.detection_folder = ""  # 目标检测图像文件夹
        self.image_files = []  # 确保图像文件列表一开始就被初始化
        self.current_index = -1  # 当前图像索引初始化
        self.annotation_format = 'voc'  # 默认VOC格式
        
        self.init_ui()
        
    def init_ui(self):
        """初始化目标检测标注界面"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 图像文件夹选择组
        folder_group = QGroupBox("图像文件夹")
        folder_layout = QGridLayout()
        
        self.detection_path_edit = QLineEdit()
        self.detection_path_edit.setReadOnly(True)
        self.detection_path_edit.setPlaceholderText("请选择图像文件夹")
        
        folder_btn = QPushButton("浏览...")
        folder_btn.clicked.connect(self.select_detection_folder)
        
        folder_layout.addWidget(QLabel("文件夹:"), 0, 0)
        folder_layout.addWidget(self.detection_path_edit, 0, 1)
        folder_layout.addWidget(folder_btn, 0, 2)
        
        folder_group.setLayout(folder_layout)
        left_layout.addWidget(folder_group)
        
        # 添加训练数据准备组
        training_prep_group = QGroupBox("训练数据准备")
        training_prep_layout = QVBoxLayout()
        
        # 添加说明标签
        help_label = QLabel("生成目标检测训练所需的文件结构:")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #333; font-size: 12px;")
        training_prep_layout.addWidget(help_label)
        
        # 文件结构说明
        structure_label = QLabel(
            "将在默认输出文件夹中创建如下结构:\n"
            "detection_data/\n"
            "├── images/\n"
            "│   ├── train/\n"
            "│   └── val/\n"
            "└── labels/\n"
            "    ├── train/\n"
            "    └── val/"
        )
        structure_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; font-family: monospace;")
        training_prep_layout.addWidget(structure_label)
        
        # 训练/验证比例
        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("训练集比例:"))
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.5, 0.95)
        self.train_ratio_spin.setValue(0.8)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.setDecimals(2)
        ratio_layout.addWidget(self.train_ratio_spin)
        training_prep_layout.addLayout(ratio_layout)
        
        # 生成按钮
        generate_btn = QPushButton("生成训练数据结构")
        generate_btn.clicked.connect(self.generate_detection_dataset)
        training_prep_layout.addWidget(generate_btn)
        
        training_prep_group.setLayout(training_prep_layout)
        left_layout.addWidget(training_prep_group)
        
        # 图像列表
        self.image_list_widget = QListWidget()
        self.image_list_widget.setSelectionMode(QListWidget.SingleSelection)
        left_layout.addWidget(QLabel("图像列表:"))
        left_layout.addWidget(self.image_list_widget)
        
        # 标签选择
        left_layout.addWidget(QLabel("标签选择:"))
        self.label_combo = QComboBox()
        left_layout.addWidget(self.label_combo)
        
        # 导航按钮
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一张")
        self.next_btn = QPushButton("下一张")
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        left_layout.addLayout(nav_layout)
        
        # 操作按钮
        op_layout = QHBoxLayout()
        self.undo_btn = QPushButton("撤销操作")
        self.reset_view_btn = QPushButton("重置视图")
        op_layout.addWidget(self.undo_btn)
        op_layout.addWidget(self.reset_view_btn)
        left_layout.addLayout(op_layout)
        
        # 添加保存标注按钮（单独一行）
        self.save_btn = QPushButton("保存标注")
        self.save_btn.setMinimumHeight(30)  # 设置按钮更高一些
        left_layout.addWidget(self.save_btn)
        
        # 格式选择
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("保存格式:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["VOC", "YOLO"])
        format_layout.addWidget(self.format_combo)
        left_layout.addLayout(format_layout)
        
        # 添加类别管理组
        class_group = QGroupBox("目标类别")
        class_layout = QVBoxLayout()
        
        # 添加类别列表
        self.detection_class_list = QListWidget()
        self.detection_class_list.setMinimumHeight(100)
        class_layout.addWidget(self.detection_class_list)
        
        # 添加按钮组
        class_btn_layout = QHBoxLayout()
        
        add_class_btn = QPushButton("添加类别")
        add_class_btn.clicked.connect(self.add_detection_class)
        class_btn_layout.addWidget(add_class_btn)
        
        remove_class_btn = QPushButton("删除类别")
        remove_class_btn.clicked.connect(self.remove_detection_class)
        class_btn_layout.addWidget(remove_class_btn)
        
        class_layout.addLayout(class_btn_layout)
        class_group.setLayout(class_layout)
        left_layout.addWidget(class_group)
        
        # 添加左侧面板到分割器
        splitter.addWidget(left_panel)
        
        # 右侧标注画布
        self.annotation_canvas = AnnotationCanvas()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.annotation_canvas)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(600)
        scroll_area.setMinimumHeight(400)
        scroll_area.setAlignment(Qt.AlignCenter)
        splitter.addWidget(scroll_area)
        
        # 设置分割器比例
        splitter.setSizes([200, 800])
        
        # 添加分割器到主布局
        main_layout.addWidget(splitter)
        
        # 添加保存选项组
        save_options_group = QGroupBox("保存选项")
        save_options_layout = QVBoxLayout()
        
        # 添加保存到特定标注文件夹的勾选框
        self.save_to_annotations_folder_checkbox = QCheckBox("保存标注文件到标注文件夹")
        self.save_to_annotations_folder_checkbox.setChecked(True)  # 默认选中
        self.save_to_annotations_folder_checkbox.setToolTip("勾选后，标注文件将保存到默认输出文件夹中的annotations子文件夹，否则保存在图片所在目录")
        save_options_layout.addWidget(self.save_to_annotations_folder_checkbox)
        
        # 添加说明文本
        save_info_label = QLabel("勾选：标注文件将保存到默认输出文件夹的annotations子文件夹\n取消勾选：标注文件将保存在图片所在的目录")
        save_info_label.setStyleSheet("color: #666; font-size: 11px;")
        save_options_layout.addWidget(save_info_label)
        
        save_options_group.setLayout(save_options_layout)
        main_layout.addWidget(save_options_group)
        
        # 状态标签
        self.detection_status_label = QLabel("就绪")
        main_layout.addWidget(self.detection_status_label)
        
        # 绑定事件
        self.image_list_widget.currentRowChanged.connect(self.load_image)
        self.label_combo.currentTextChanged.connect(self.on_label_changed)
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.save_btn.clicked.connect(self.save_annotations)
        self.undo_btn.clicked.connect(self.undo_operation)
        self.format_combo.currentTextChanged.connect(self.set_annotation_format)
        self.reset_view_btn.clicked.connect(self.reset_view)
        
    def select_detection_folder(self):
        """选择目标检测图像文件夹"""
        try:
            folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
            if folder:
                self.detection_path_edit.setText(folder)
                self.detection_folder = folder
                self.load_image_files(folder)
                self.check_detection_ready()
        except Exception as e:
            print(f"选择目标检测文件夹时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def check_detection_ready(self):
        """检查是否可以开始目标检测标注"""
        try:
            has_image_folder = bool(self.detection_path_edit.text())
            has_classes = self.detection_class_list.count() > 0
            
            # 安全检查image_files
            if not hasattr(self, 'image_files'):
                self.image_files = []
            
            # 更新标签下拉框
            if has_classes:
                self.update_label_combo()
                
            # 更新按钮状态
            self.save_btn.setEnabled(has_image_folder and has_classes and self.current_index >= 0)
            self.prev_btn.setEnabled(has_image_folder and self.current_index > 0)
            self.next_btn.setEnabled(has_image_folder and self.current_index < len(self.image_files) - 1)
        except Exception as e:
            print(f"检查标注准备状态时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def add_detection_class(self):
        """添加目标检测类别"""
        class_name, ok = QInputDialog.getText(self, "添加目标类别", "请输入目标类别名称:")
        if ok and class_name:
            # 检查是否已存在
            if class_name in self.detection_classes:
                QMessageBox.warning(self, "警告", f"类别 '{class_name}' 已存在!")
                return
            self.detection_classes.append(class_name)
            self.detection_class_list.addItem(class_name)
            self.update_label_combo()
            self.check_detection_ready()
            
    def remove_detection_class(self):
        """删除目标检测类别"""
        current_item = self.detection_class_list.currentItem()
        if current_item:
            class_name = current_item.text()
            self.detection_classes.remove(class_name)
            self.detection_class_list.takeItem(self.detection_class_list.row(current_item))
            self.update_label_combo()
            self.check_detection_ready()
            
    def load_image_files(self, folder):
        """加载图像文件列表"""
        if not folder or not os.path.exists(folder):
            print(f"文件夹不存在或无效: {folder}")
            self.update_detection_status("错误: 文件夹不存在或无效")
            return
        
        try:
            self.image_files = []
            
            # 支持的图像格式（不区分大小写）
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            
            # 获取文件夹中的所有文件
            all_files = os.listdir(folder)
            
            # 过滤出图像文件
            for file in all_files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extensions:
                    full_path = os.path.join(folder, file)
                    self.image_files.append(full_path)
            
            print(f"找到 {len(self.image_files)} 个图像文件")
            
            # 更新图像列表
            self.image_list_widget.clear()
            for image_file in self.image_files:
                self.image_list_widget.addItem(os.path.basename(image_file))
                
            # 如果有图像，选择第一张
            if self.image_files:
                self.current_index = 0
                self.image_list_widget.setCurrentRow(0)
                self.update_detection_status(f"已加载 {len(self.image_files)} 张图像")
            else:
                self.current_index = -1
                self.update_detection_status("未找到图像文件")
        except Exception as e:
            print(f"加载图像文件时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.update_detection_status(f"加载图像文件出错: {str(e)}")
            
    def load_image(self, index):
        """加载指定索引的图像"""
        try:
            # 检查image_files是否已初始化
            if not hasattr(self, 'image_files') or self.image_files is None:
                self.image_files = []
                self.update_detection_status("图像列表尚未初始化")
                return
                
            # 检查索引是否有效
            if index < 0 or index >= len(self.image_files):
                print(f"索引无效: {index}，有效范围: 0-{len(self.image_files)-1 if self.image_files else -1}")
                return
                
            self.current_index = index
            image_path = self.image_files[index]
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                self.update_detection_status(f"图像文件不存在: {os.path.basename(image_path)}")
                return
                
            # 加载图像到画布
            if self.annotation_canvas.set_image(image_path):
                self.update_detection_status(f"已加载图像: {os.path.basename(image_path)}")
                # 只更新按钮状态，不更新标签
                self.save_btn.setEnabled(True)
                self.prev_btn.setEnabled(self.current_index > 0)
                self.next_btn.setEnabled(self.current_index < len(self.image_files) - 1)
            else:
                self.update_detection_status(f"加载图像失败: {os.path.basename(image_path)}")
        except Exception as e:
            print(f"加载图像时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.update_detection_status(f"加载图像出错: {str(e)}")
            
    def prev_image(self):
        """加载上一张图像"""
        if self.current_index > 0:
            self.image_list_widget.setCurrentRow(self.current_index - 1)
            
    def next_image(self):
        """加载下一张图像"""
        if self.current_index < len(self.image_files) - 1:
            self.image_list_widget.setCurrentRow(self.current_index + 1)
            
    def update_label_combo(self):
        """更新标签下拉框"""
        try:
            # 检查是否需要更新
            current_items = [self.label_combo.itemText(i) for i in range(self.label_combo.count())]
            detection_items = [self.detection_class_list.item(i).text() for i in range(self.detection_class_list.count())]
            
            # 只有当类别列表发生变化时才更新
            if current_items != detection_items:
                print("类别列表已变化，更新标签下拉框")
                self.label_combo.clear()
                
                # 添加所有类别
                for i in range(self.detection_class_list.count()):
                    self.label_combo.addItem(self.detection_class_list.item(i).text())
                
                # 如果有类别，设置当前标签（仅在初始化时设置）
                if self.label_combo.count() > 0 and not hasattr(self, '_label_combo_initialized'):
                    self.annotation_canvas.set_current_label(self.label_combo.itemText(0))
                    self._label_combo_initialized = True  # 标记已初始化
            else:
                print("类别列表未变化，无需更新标签下拉框")
        except Exception as e:
            print(f"更新标签下拉框时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def on_label_changed(self, label):
        """标签改变时调用"""
        try:
            if label:
                self.annotation_canvas.set_current_label(label)
        except Exception as e:
            print(f"更改标签时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def save_annotations(self):
        """保存当前图像的标注结果"""
        try:
            # 检查是否有图像加载
            if not hasattr(self, 'image_files') or not self.image_files or self.current_index < 0 or self.current_index >= len(self.image_files):
                QMessageBox.warning(self, "错误", "没有加载图像或图像索引无效")
                return
            
            # 确定保存位置
            if self.save_to_annotations_folder_checkbox.isChecked():
                # 使用默认输出文件夹中的annotations子文件夹
                if not self.main_window or not hasattr(self.main_window, 'config'):
                    QMessageBox.warning(self, "错误", "无法获取默认输出文件夹设置")
                    return
                    
                default_output_folder = self.main_window.config.get('default_output_folder', '')
                if not default_output_folder:
                    QMessageBox.warning(self, "错误", "请先在设置中配置默认输出文件夹")
                    return
                    
                output_folder = os.path.join(default_output_folder, 'annotations')
                os.makedirs(output_folder, exist_ok=True)
            else:
                # 使用图像所在的文件夹
                image_path = self.image_files[self.current_index]
                output_folder = os.path.dirname(image_path)
                
            # 保存标注
            if self.annotation_canvas.save_annotations(output_folder, self.annotation_format):
                save_location = "标注文件夹" if self.save_to_annotations_folder_checkbox.isChecked() else "图片目录"
                self.update_detection_status(f"已保存标注结果到{save_location}: {os.path.basename(self.image_files[self.current_index])}")
                
                # 自动加载下一张图像
                self.next_image()
            else:
                self.update_detection_status("保存标注结果失败")
        except Exception as e:
            print(f"保存标注时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"保存标注时出错: {str(e)}")
            
    def undo_operation(self):
        """撤销操作"""
        if self.annotation_canvas.undo():
            self.update_detection_status("已撤销最近的操作")
        else:
            self.update_detection_status("没有可撤销的操作")
            
    def reset_view(self):
        """重置视图"""
        self.annotation_canvas.reset_view()
        self.update_detection_status("视图已重置")
        
    def set_annotation_format(self, format_type):
        """设置标注格式"""
        self.annotation_format = format_type.lower()
        self.update_detection_status(f"标注格式已设置为: {format_type}")
        
    def update_detection_status(self, message):
        """更新目标检测界面的状态信息"""
        try:
            self.detection_status_label.setText(message)
            # 同时通知主窗口更新状态栏
            if self.main_window and hasattr(self.main_window, 'update_status'):
                self.main_window.update_status(message)
        except Exception as e:
            print(f"更新状态信息时出错: {str(e)}")
            
    def set_detection_folder(self, folder):
        """设置检测文件夹路径"""
        self.detection_folder = folder
        self.detection_path_edit.setText(folder)
        self.load_image_files(folder)
        self.check_detection_ready()
        
    def set_detection_classes(self, classes):
        """设置检测类别列表"""
        self.detection_classes = classes.copy()
        self.detection_class_list.clear()
        for class_name in self.detection_classes:
            self.detection_class_list.addItem(class_name)
        self.update_label_combo()
        self.check_detection_ready()
        
    def get_detection_classes(self):
        """获取检测类别列表"""
        return self.detection_classes.copy()
        
    def get_detection_folder(self):
        """获取检测文件夹路径"""
        return self.detection_folder
        
    def generate_detection_dataset(self):
        """生成目标检测训练数据集文件结构"""
        # 获取所需路径
        source_folder = self.detection_path_edit.text()
        
        # 使用默认输出文件夹作为目标文件夹和标注文件夹
        if not self.main_window or not hasattr(self.main_window, 'config'):
            QMessageBox.warning(self, "错误", "无法获取默认输出文件夹设置")
            return
            
        default_output_folder = self.main_window.config.get('default_output_folder', '')
        if not default_output_folder:
            QMessageBox.warning(self, "错误", "请先在设置中配置默认输出文件夹")
            return
            
        target_folder = default_output_folder
        annotation_folder = os.path.join(default_output_folder, 'annotations')
        train_ratio = self.train_ratio_spin.value()
        
        # 打印路径信息以便调试
        print(f"源图像文件夹: {source_folder}")
        print(f"标注文件夹: {annotation_folder}")
        print(f"目标文件夹: {target_folder}")
        print(f"训练集比例: {train_ratio}")
        
        # 验证路径
        if not source_folder or not os.path.exists(source_folder):
            QMessageBox.warning(self, "错误", "请选择有效的图像文件夹")
            return
            
        if not os.path.exists(annotation_folder):
            os.makedirs(annotation_folder)
            
        # 确认操作
        reply = QMessageBox.question(
            self, "确认操作", 
            f"将在{target_folder}中创建训练数据结构，并将图像和标注文件复制到相应文件夹。继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        try:
            # 创建目录结构
            detection_data_dir = os.path.join(target_folder, "detection_data")
            images_dir = os.path.join(detection_data_dir, "images")
            labels_dir = os.path.join(detection_data_dir, "labels")
            train_images_dir = os.path.join(images_dir, "train")
            val_images_dir = os.path.join(images_dir, "val")
            train_labels_dir = os.path.join(labels_dir, "train")
            val_labels_dir = os.path.join(labels_dir, "val")
            
            os.makedirs(train_images_dir, exist_ok=True)
            os.makedirs(val_images_dir, exist_ok=True)
            os.makedirs(train_labels_dir, exist_ok=True)
            os.makedirs(val_labels_dir, exist_ok=True)
            
            # 获取所有图像文件
            image_files = []
            print(f"在{source_folder}中搜索图像文件...")
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                pattern = os.path.join(source_folder, ext)
                found = glob.glob(pattern)
                print(f"使用模式 {pattern} 找到 {len(found)} 个文件")
                image_files.extend(found)
                
                # 同时查找大写扩展名
                upper_pattern = os.path.join(source_folder, ext.upper())
                found_upper = glob.glob(upper_pattern)
                print(f"使用模式 {upper_pattern} 找到 {len(found_upper)} 个文件")
                image_files.extend(found_upper)
            
            print(f"总共找到 {len(image_files)} 个图像文件")
            
            if not image_files:
                QMessageBox.warning(self, "错误", f"在{source_folder}中未找到图像文件")
                return
                
            # 计算训练集和验证集大小
            import random
            random.shuffle(image_files)
            train_size = int(len(image_files) * train_ratio)
            train_files = image_files[:train_size]
            val_files = image_files[train_size:]
            
            print(f"训练集: {len(train_files)}张图像, 验证集: {len(val_files)}张图像")
            
            # 处理进度
            total_files = len(image_files)
            processed = 0
            
            # 导入需要的模块
            import shutil
            
            # 复制训练集图像和标注
            for img_path in train_files:
                img_filename = os.path.basename(img_path)
                base_name = os.path.splitext(img_filename)[0]
                
                # 复制图像
                dest_img_path = os.path.join(train_images_dir, img_filename)
                shutil.copy2(img_path, dest_img_path)
                
                # 寻找对应的标注文件(YOLO格式.txt)
                label_path = os.path.join(annotation_folder, f"{base_name}.txt")
                if os.path.exists(label_path):
                    dest_label_path = os.path.join(train_labels_dir, f"{base_name}.txt")
                    shutil.copy2(label_path, dest_label_path)
                else:
                    print(f"警告: 未找到图像 {img_filename} 对应的标注文件")
                    
                processed += 1
                self.update_detection_status(f"处理中... {processed}/{total_files}")
                QApplication.processEvents() # 保持UI响应
            
            # 复制验证集图像和标注
            for img_path in val_files:
                img_filename = os.path.basename(img_path)
                base_name = os.path.splitext(img_filename)[0]
                
                # 复制图像
                dest_img_path = os.path.join(val_images_dir, img_filename)
                shutil.copy2(img_path, dest_img_path)
                
                # 寻找对应的标注文件(YOLO格式.txt)
                label_path = os.path.join(annotation_folder, f"{base_name}.txt")
                if os.path.exists(label_path):
                    dest_label_path = os.path.join(val_labels_dir, f"{base_name}.txt")
                    shutil.copy2(label_path, dest_label_path)
                else:
                    print(f"警告: 未找到图像 {img_filename} 对应的标注文件")
                    
                processed += 1
                self.update_detection_status(f"处理中... {processed}/{total_files}")
                QApplication.processEvents() # 保持UI响应
            
            # 创建classes.txt文件
            if self.detection_classes:
                classes_path = os.path.join(detection_data_dir, "classes.txt")
                print(f"正在创建类别文件: {classes_path}")
                with open(classes_path, 'w', encoding='utf-8') as f:
                    for i, class_name in enumerate(self.detection_classes):
                        f.write(f"{i} {class_name}\n")
                        
                print(f"类别文件已创建，包含 {len(self.detection_classes)} 个类别")
            
            # 完成
            msg = (f"已成功生成训练数据结构。\n"
                  f"训练集: {len(train_files)}张图像\n"
                  f"验证集: {len(val_files)}张图像\n"
                  f"目标文件夹: {detection_data_dir}")
            
            print(msg)
            QMessageBox.information(self, "成功", msg)
            self.update_detection_status("训练数据结构生成完成")
            
            # 为训练标签页设置目标检测目录路径
            if (self.main_window and hasattr(self.main_window, 'training_tab') and 
                hasattr(self.main_window.training_tab, 'detection_widget') and 
                hasattr(self.main_window.training_tab.detection_widget, 'path_edit')):
                self.main_window.training_tab.detection_widget.path_edit.setText(detection_data_dir)
                print(f"已自动为训练标签页设置检测数据路径: {detection_data_dir}")
                # 同时更新training_tab的annotation_folder属性
                if hasattr(self.main_window.training_tab, 'annotation_folder'):
                    self.main_window.training_tab.annotation_folder = detection_data_dir
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成训练数据结构时出错: {str(e)}")
            import traceback
            traceback.print_exc() 
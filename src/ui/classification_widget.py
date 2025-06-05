from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
                           QHBoxLayout, QListWidget, QGroupBox, QGridLayout,
                           QLineEdit, QInputDialog, QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import os

class ClassificationWidget(QWidget):
    """图像分类标注组件"""
    
    # 定义信号
    annotation_started = pyqtSignal(str)
    open_validation_folder_signal = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.defect_classes = []  # 缺陷类别列表
        self.processed_folder = ""  # 处理后的图片文件夹
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        
        # 创建顶部控制面板
        control_layout = QHBoxLayout()
        
        # 图像文件夹选择组
        folder_group = QGroupBox("处理后的图片文件夹")
        folder_layout = QHBoxLayout()
        
        self.processed_path_edit = QLineEdit()
        self.processed_path_edit.setReadOnly(True)
        self.processed_path_edit.setPlaceholderText("请选择处理后的图片文件夹")
        
        folder_btn = QPushButton("浏览...")
        folder_btn.clicked.connect(self.select_processed_folder)
        
        folder_layout.addWidget(self.processed_path_edit)
        folder_layout.addWidget(folder_btn)
        
        folder_group.setLayout(folder_layout)
        control_layout.addWidget(folder_group)
        
        # 类别管理组
        class_group = QGroupBox("缺陷类别")
        class_layout = QVBoxLayout()
        
        # 添加类别列表
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(100)
        class_layout.addWidget(self.class_list)
        
        # 添加按钮组
        btn_layout = QHBoxLayout()
        
        add_class_btn = QPushButton("添加类别")
        add_class_btn.clicked.connect(self.add_defect_class)
        btn_layout.addWidget(add_class_btn)
        
        remove_class_btn = QPushButton("删除类别")
        remove_class_btn.clicked.connect(self.remove_defect_class)
        btn_layout.addWidget(remove_class_btn)
        
        # 添加未分类选项
        self.create_unclassified_checkbox = QCheckBox("创建未分类文件夹")
        self.create_unclassified_checkbox.setChecked(True)
        btn_layout.addWidget(self.create_unclassified_checkbox)
        
        class_layout.addLayout(btn_layout)
        class_group.setLayout(class_layout)
        control_layout.addWidget(class_group)
        
        main_layout.addLayout(control_layout)
        
        # 创建开始标注按钮
        self.annotation_btn = QPushButton("开始分类标注")
        self.annotation_btn.setMinimumHeight(40)
        self.annotation_btn.clicked.connect(self.start_annotation)
        self.annotation_btn.setEnabled(False)  # 默认禁用
        
        main_layout.addWidget(self.annotation_btn)
        
        # 添加打开验证集文件夹按钮
        self.val_folder_btn = QPushButton("打开验证集文件夹")
        self.val_folder_btn.setMinimumHeight(40)
        self.val_folder_btn.clicked.connect(self.open_validation_folder)
        self.val_folder_btn.setEnabled(False)  # 默认禁用
        
        main_layout.addWidget(self.val_folder_btn)
        
        # 添加使用说明
        help_text = """
        <b>图像分类标注步骤:</b>
        <ol>
            <li>选择处理后的图片文件夹</li>
            <li>添加需要的缺陷类别</li>
            <li>点击"开始分类标注"按钮</li>
            <li>系统将在训练集(train)和验证集(val)文件夹中分别创建类别文件夹</li>
            <li>在弹出的文件浏览器中，将训练集图片拖放到对应的类别文件夹中</li>
            <li>点击"打开验证集文件夹"按钮，对验证集图片进行同样的标注</li>
        </ol>
        
        <b>提示:</b>
        <ul>
            <li>可以选择是否创建"未分类"文件夹，用于存放暂时无法分类的图片</li>
            <li>分类完成后，各个缺陷类别的图片将位于对应类别文件夹中，便于后续训练</li>
            <li>务必确保训练集和验证集都完成了分类标注，以保证模型训练效果</li>
        </ul>
        """
        
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setTextFormat(Qt.RichText)
        help_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        main_layout.addWidget(help_label)
        main_layout.addStretch()
        
    def select_processed_folder(self):
        """选择处理后的图片文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择处理后的图片文件夹")
        if folder:
            self.processed_path_edit.setText(folder)
            self.processed_folder = folder
            self.check_annotation_ready()
            
    def add_defect_class(self):
        """添加缺陷类别"""
        class_name, ok = QInputDialog.getText(self, "添加缺陷类别", "请输入缺陷类别名称:")
        if ok and class_name:
            # 检查是否已存在
            if class_name in self.defect_classes:
                QMessageBox.warning(self, "警告", f"类别 '{class_name}' 已存在!")
                return
            self.defect_classes.append(class_name)
            self.class_list.addItem(class_name)
            self.check_annotation_ready()
            
    def remove_defect_class(self):
        """删除缺陷类别"""
        current_item = self.class_list.currentItem()
        if current_item:
            class_name = current_item.text()
            self.defect_classes.remove(class_name)
            self.class_list.takeItem(self.class_list.row(current_item))
            self.check_annotation_ready()
            
    def check_annotation_ready(self):
        """检查是否可以开始图片分类标注"""
        is_ready = bool(self.processed_folder and self.defect_classes)
        self.annotation_btn.setEnabled(is_ready)
        self.val_folder_btn.setEnabled(is_ready)  # 同时启用/禁用验证集按钮
        
    def start_annotation(self):
        """开始图片分类标注"""
        if not self.processed_folder:
            QMessageBox.warning(self, "警告", "请先选择处理后的图片文件夹!")
            return
            
        if not self.defect_classes:
            QMessageBox.warning(self, "警告", "请先添加至少一个缺陷类别!")
            return
            
        # 创建分类文件夹
        if not self.create_classification_folders():
            return  # 如果创建文件夹失败，终止操作
        
        # 发出标注开始信号
        self.annotation_started.emit(self.processed_folder)
        
    def create_classification_folders(self):
        """创建分类文件夹"""
        try:
            created_count = 0
            
            # 指定train和val文件夹路径
            dataset_folder = os.path.join(self.processed_folder, 'dataset')
            train_folder = os.path.join(dataset_folder, 'train')
            val_folder = os.path.join(dataset_folder, 'val')
            
            # 检查必要的文件夹是否存在
            if not os.path.exists(train_folder) or not os.path.exists(val_folder):
                QMessageBox.warning(self, "警告", "数据集文件夹结构不完整，请先完成数据预处理步骤")
                return False

            # 在train文件夹中为每个类别创建文件夹
            for class_name in self.defect_classes:
                train_class_folder = os.path.join(train_folder, class_name)
                if not os.path.exists(train_class_folder):
                    os.makedirs(train_class_folder)
                    created_count += 1
                
                # 同时在val文件夹中也创建对应的类别文件夹
                val_class_folder = os.path.join(val_folder, class_name)
                if not os.path.exists(val_class_folder):
                    os.makedirs(val_class_folder)
                    created_count += 1
                    
            # 根据复选框状态决定是否创建未分类文件夹
            if self.create_unclassified_checkbox.isChecked():
                # 在train和val文件夹中创建未分类文件夹
                train_unclassified_folder = os.path.join(train_folder, "未分类")
                if not os.path.exists(train_unclassified_folder):
                    os.makedirs(train_unclassified_folder)
                    created_count += 1
                
                val_unclassified_folder = os.path.join(val_folder, "未分类")
                if not os.path.exists(val_unclassified_folder):
                    os.makedirs(val_unclassified_folder)
                    created_count += 1
                
            # 添加成功提示
            if created_count > 0:
                QMessageBox.information(self, "成功", f"已成功创建 {created_count} 个分类文件夹")
            else:
                QMessageBox.information(self, "提示", "所有分类文件夹已存在，无需重新创建")
                
            return True
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建分类文件夹失败: {str(e)}")
            return False
            
    def open_validation_folder(self):
        """打开验证集文件夹"""
        if not self.processed_folder:
            QMessageBox.warning(self, "警告", "请先选择处理后的图片文件夹!")
            return
            
        # 发出打开验证集文件夹信号
        self.open_validation_folder_signal.emit(self.processed_folder)
        
    def set_processed_folder(self, folder):
        """设置处理后文件夹路径"""
        self.processed_folder = folder
        self.processed_path_edit.setText(folder)
        self.check_annotation_ready()
        
    def set_defect_classes(self, classes):
        """设置缺陷类别列表"""
        self.defect_classes = classes.copy()
        self.class_list.clear()
        for class_name in self.defect_classes:
            self.class_list.addItem(class_name)
        self.check_annotation_ready()
        
    def get_defect_classes(self):
        """获取缺陷类别列表"""
        return self.defect_classes.copy()
        
    def get_processed_folder(self):
        """获取处理后文件夹路径"""
        return self.processed_folder 
import os
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                           QGridLayout, QPushButton, QLabel, QLineEdit, 
                           QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox,
                           QTextEdit, QPlainTextEdit, QProgressBar, QFileDialog, QMessageBox,
                           QTabWidget, QTableWidget, QTableWidgetItem,
                           QHeaderView, QFrame, QScrollArea)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QMutex, QMutexLocker
from PyQt5.QtGui import QFont, QColor
import logging
from src.utils.logger import get_logger, log_error, PerformanceMonitor


class AutoReviewThread(QThread):
    """自动Review独立线程，监控目录并进行自动分类"""
    
    # 定义信号
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    file_processed = pyqtSignal(dict)  # 文件处理结果
    error_occurred = pyqtSignal(str)
    statistics_updated = pyqtSignal(dict)
    
    def __init__(self, predictor, config):
        super().__init__()
        self.predictor = predictor
        self.config = config
        self._stop_processing = False
        self._is_paused = False
        self.logger = get_logger(__name__, "auto_review")
        self.mutex = QMutex()
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'classified': 0,
            'unclassified': 0,
            'errors': 0,
            'skipped_lots': 0,
            'class_counts': {},
            'start_time': None,
            'last_scan_time': None
        }
        
        # 已处理文件记录，避免重复处理
        self.processed_files = set()
        
    def run(self):
        """线程主入口"""
        try:
            self.stats['start_time'] = time.time()
            self.status_updated.emit("自动Review服务已启动")
            self.logger.info("自动Review线程启动")
            
            # 验证配置
            if not self._validate_config():
                return
                
            self._auto_review_loop()
            
        except Exception as e:
            import traceback
            error_msg = f"自动Review线程出错: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            log_error(e, {"operation": "auto_review_thread"}, "auto_review")
            self.error_occurred.emit(error_msg)
    
    def stop_processing(self):
        """停止处理"""
        with QMutexLocker(self.mutex):
            self._stop_processing = True
        self.status_updated.emit("正在停止自动Review服务...")
    
    def pause_processing(self):
        """暂停处理"""
        with QMutexLocker(self.mutex):
            self._is_paused = True
        self.status_updated.emit("自动Review服务已暂停")
    
    def resume_processing(self):
        """恢复处理"""
        with QMutexLocker(self.mutex):
            self._is_paused = False
        self.status_updated.emit("自动Review服务已恢复")
    
    def _validate_config(self) -> bool:
        """验证配置参数"""
        required_keys = ['scan_folder', 'review_folder']
        for key in required_keys:
            if not self.config.get(key):
                self.error_occurred.emit(f"配置错误: {key} 不能为空")
                return False
        
        # 检查扫描文件夹是否存在
        if not os.path.exists(self.config['scan_folder']):
            self.error_occurred.emit(f"扫描文件夹不存在: {self.config['scan_folder']}")
            return False
        
        return True
    
    def _auto_review_loop(self):
        """自动Review主循环"""
        scan_interval = self.config.get('scan_interval', 10)  # 默认10秒扫描一次
        
        while not self._stop_processing:
            try:
                # 检查是否暂停
                if self._is_paused:
                    time.sleep(1)
                    continue
                
                # 扫描并处理新文件
                self._scan_and_process()
                
                # 更新统计信息
                self.statistics_updated.emit(self.stats.copy())
                
                # 等待下次扫描
                for _ in range(scan_interval):
                    if self._stop_processing:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                error_msg = f"自动Review循环出错: {str(e)}"
                self.logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                time.sleep(5)  # 出错后等待5秒再继续
    
    def _scan_and_process(self):
        """扫描并处理新文件 - 适配recipeID/setup1/lotID/waferID结构"""
        self.stats['last_scan_time'] = time.time()
        scan_folder = self.config['scan_folder']
        
        try:
            # 深度扫描：ScanResults/recipeID/setup1/lotID/waferID
            wafer_paths = []
            
            # 遍历 recipeID 层级
            for recipe_id in os.listdir(scan_folder):
                recipe_path = os.path.join(scan_folder, recipe_id)
                if not os.path.isdir(recipe_path):
                    continue
                
                # 遍历 setup1 层级
                setup_path = os.path.join(recipe_path, 'setup1')
                if not os.path.exists(setup_path) or not os.path.isdir(setup_path):
                    continue
                
                # 遍历 lotID 层级
                for lot_id in os.listdir(setup_path):
                    lot_path = os.path.join(setup_path, lot_id)
                    if not os.path.isdir(lot_path):
                        continue
                    
                    # 遍历 waferID 层级
                    for wafer_id in os.listdir(lot_path):
                        wafer_path = os.path.join(lot_path, wafer_id)
                        if os.path.isdir(wafer_path):
                            # 构建完整的路径信息
                            path_info = {
                                'recipe_id': recipe_id,
                                'lot_id': lot_id,
                                'wafer_id': wafer_id,
                                'full_path': wafer_path
                            }
                            wafer_paths.append(path_info)
            
            if not wafer_paths:
                return
            
            self.status_updated.emit(f"发现 {len(wafer_paths)} 个Wafer文件夹，开始处理...")
            
            for i, path_info in enumerate(wafer_paths):
                if self._stop_processing:
                    break
                
                # 检查是否已处理过此文件夹（内存中记录）
                if path_info['full_path'] in self.processed_files:
                    continue
                
                # 检查是否需要跳过已Review的LotID
                if self.config.get('skip_processed', True):
                    if self._is_lot_already_reviewed(path_info['recipe_id'], path_info['lot_id']):
                        self.status_updated.emit(f"跳过已Review的LotID: {path_info['recipe_id']}/{path_info['lot_id']}")
                        self.processed_files.add(path_info['full_path'])
                        self.stats['skipped_lots'] += 1
                        continue
                
                self._process_wafer_folder_new_structure(path_info)
                self.processed_files.add(path_info['full_path'])
                
                # 更新进度
                progress = int((i + 1) / len(wafer_paths) * 100)
                self.progress_updated.emit(progress)
                
        except Exception as e:
            error_msg = f"扫描文件夹出错: {str(e)}"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
    
    def _process_wafer_folder_new_structure(self, path_info: dict):
        """处理单个Wafer文件夹 - 新的路径结构"""
        try:
            recipe_id = path_info['recipe_id']
            lot_id = path_info['lot_id']
            wafer_id = path_info['wafer_id']
            wafer_path = path_info['full_path']
            
            self.status_updated.emit(f"处理 {recipe_id}/{lot_id}/{wafer_id}")
            
            # 直接在waferID文件夹中查找JPEG文件
            jpeg_files = []
            for file in os.listdir(wafer_path):
                file_path = os.path.join(wafer_path, file)
                if os.path.isfile(file_path):
                    # 只处理JPEG格式，忽略JEG格式
                    if file.lower().endswith('.jpeg'):
                        jpeg_files.append(file_path)
                    elif file.lower().endswith('.jpg'):
                        # 检查是否实际是JPEG格式而不是JEG
                        if not file.lower().endswith('.jeg'):
                            jpeg_files.append(file_path)
            
            if not jpeg_files:
                self.logger.info(f"Wafer {wafer_id} 中未找到JPEG文件")
                return
            
            self.logger.info(f"Wafer {recipe_id}/{lot_id}/{wafer_id} 找到 {len(jpeg_files)} 个JPEG文件")
            
            # 处理每个JPEG文件
            for jpeg_file in jpeg_files:
                if self._stop_processing:
                    break
                self._process_jpeg_file_new_structure(path_info, jpeg_file)
                
        except Exception as e:
            error_msg = f"处理Wafer文件夹 {wafer_id} 出错: {str(e)}"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self.stats['errors'] += 1
    
    def _process_jpeg_file_new_structure(self, path_info: dict, jpeg_file: str):
        """处理单个JPEG文件进行缺陷分类 - 新的路径结构"""
        try:
            recipe_id = path_info['recipe_id']
            lot_id = path_info['lot_id']
            wafer_id = path_info['wafer_id']
            
            # 使用预测器进行缺陷分类
            result = self.predictor.predict_image(jpeg_file)
            
            if not result or not result.get('predictions'):
                self.logger.warning(f"文件 {jpeg_file} 预测失败")
                self.stats['errors'] += 1
                return
            
            # 获取最高置信度的预测结果
            top_prediction = result['predictions'][0]
            class_name = top_prediction['class_name']
            probability = top_prediction['probability']
            
            confidence_threshold = self.config.get('confidence_threshold', 50.0)
            
            # 检查置信度是否达标
            if probability < confidence_threshold:
                self.logger.info(f"文件 {jpeg_file} 置信度过低: {probability:.2f}% < {confidence_threshold}%")
                self.stats['unclassified'] += 1
                return
            
            # 创建输出目录结构: Review/recipeID/lotID/缺陷类别
            self._create_output_structure_and_copy_new(path_info, jpeg_file, class_name)
            
            # 更新统计信息
            self.stats['total_processed'] += 1
            self.stats['classified'] += 1
            if class_name not in self.stats['class_counts']:
                self.stats['class_counts'][class_name] = 0
            self.stats['class_counts'][class_name] += 1
            
            # 发送文件处理结果信号
            file_result = {
                'recipe_id': recipe_id,
                'lot_id': lot_id,
                'wafer_id': wafer_id,
                'file_path': jpeg_file,
                'class_name': class_name,
                'probability': probability,
                'status': 'success'
            }
            self.file_processed.emit(file_result)
            
        except Exception as e:
            error_msg = f"处理JPEG文件 {jpeg_file} 出错: {str(e)}"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self.stats['errors'] += 1
    
    def _create_output_structure_and_copy_new(self, path_info: dict, jpeg_file: str, class_name: str):
        """创建输出目录结构并复制文件 - 新的路径结构"""
        try:
            review_folder = self.config['review_folder']
            recipe_id = path_info['recipe_id']
            lot_id = path_info['lot_id']
            wafer_id = path_info['wafer_id']
            
            # 创建目录结构: Review/recipeID/lotID/缺陷类别
            target_dir = os.path.join(review_folder, recipe_id, lot_id, class_name)
            os.makedirs(target_dir, exist_ok=True)
            
            # 复制文件
            file_name = os.path.basename(jpeg_file)
            target_path = os.path.join(target_dir, file_name)
            
            # 如果文件已存在，添加waferID前缀避免冲突
            if os.path.exists(target_path):
                base_name, ext = os.path.splitext(file_name)
                new_name = f"{wafer_id}_{base_name}{ext}"
                target_path = os.path.join(target_dir, new_name)
                
                # 如果还是存在，添加序号
                counter = 1
                while os.path.exists(target_path):
                    new_name = f"{wafer_id}_{base_name}_{counter}{ext}"
                    target_path = os.path.join(target_dir, new_name)
                    counter += 1
            
            # 执行文件复制
            copy_mode = self.config.get('copy_mode', 'copy')
            if copy_mode == 'move':
                shutil.move(jpeg_file, target_path)
            else:
                shutil.copy2(jpeg_file, target_path)
            
            self.logger.info(f"文件已{('移动' if copy_mode == 'move' else '复制')}到: {target_path}")
            
        except Exception as e:
            raise Exception(f"创建输出结构失败: {str(e)}")
    
    def _is_lot_already_reviewed(self, recipe_id: str, lot_id: str) -> bool:
        """检查指定的LotID是否已经被Review过"""
        try:
            review_folder = self.config['review_folder']
            lot_review_path = os.path.join(review_folder, recipe_id, lot_id)
            
            # 检查Review文件夹中是否存在对应的LotID路径
            if not os.path.exists(lot_review_path):
                return False
            
            # 检查是否包含任何分类文件夹和图片
            if not os.path.isdir(lot_review_path):
                return False
            
            # 遍历缺陷类别文件夹
            defect_folders = [f for f in os.listdir(lot_review_path) 
                            if os.path.isdir(os.path.join(lot_review_path, f))]
            
            if not defect_folders:
                return False
            
            # 检查是否包含图片文件
            for defect_folder in defect_folders:
                defect_path = os.path.join(lot_review_path, defect_folder)
                image_files = [f for f in os.listdir(defect_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    # 找到至少一个图片文件，说明已经Review过
                    self.logger.info(f"LotID {recipe_id}/{lot_id} 已存在Review结果，包含 {len(image_files)} 个图片")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"检查LotID {recipe_id}/{lot_id} Review状态时出错: {str(e)}")
            return False


class AutoReviewWidget(QWidget):
    """自动Review组件主界面"""
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.auto_review_thread = None
        self.logger = get_logger(__name__, "auto_review_widget")
        
        # 默认配置
        self.config = {
            'scan_folder': r'\\192.168.129.25\ScanResults',
            'review_folder': r'\\192.168.129.2\D:\01_AOI_ADC_Review(review)',
            'scan_interval': 10,
            'confidence_threshold': 80.0,
            'copy_mode': 'copy',
            'auto_start': False,
            'skip_processed': True
        }
        
        # 自动加载默认配置文件
        self._load_default_config()
        
        self.init_ui()
    
    def _load_default_config(self):
        """程序启动时自动加载默认配置文件"""
        try:
            # 获取项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..', '..', '..')
            project_root = os.path.normpath(project_root)
            
            # 默认配置文件路径
            default_config_path = os.path.join(project_root, 'setting', 'auto_review_config.json')
            
            if os.path.exists(default_config_path):
                self.logger.info(f"找到默认配置文件: {default_config_path}")
                
                with open(default_config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # 更新配置，保留默认值作为备份
                for key, value in loaded_config.items():
                    if key in self.config:
                        self.config[key] = value
                
                self.logger.info("默认配置文件加载成功")
            else:
                self.logger.info(f"默认配置文件不存在: {default_config_path}，使用内置默认配置")
                
        except Exception as e:
            self.logger.warning(f"加载默认配置文件时出错: {str(e)}，使用内置默认配置")
    
    def init_ui(self):
        """初始化UI界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 创建标签页
        tab_widget = QTabWidget()
        
        # 配置标签页
        config_tab = QWidget()
        self.init_config_tab(config_tab)
        tab_widget.addTab(config_tab, "配置设置")
        
        # 监控标签页
        monitor_tab = QWidget()
        self.init_monitor_tab(monitor_tab)
        tab_widget.addTab(monitor_tab, "运行监控")
        
        # 统计标签页
        stats_tab = QWidget()
        self.init_stats_tab(stats_tab)
        tab_widget.addTab(stats_tab, "统计信息")
        
        layout.addWidget(tab_widget)
    
    def init_config_tab(self, tab_widget):
        """初始化配置标签页"""
        layout = QVBoxLayout(tab_widget)
        
        # 路径配置组
        path_group = QGroupBox("路径配置")
        path_layout = QGridLayout()
        
        # 扫描文件夹
        path_layout.addWidget(QLabel("扫描文件夹:"), 0, 0)
        self.scan_folder_edit = QLineEdit(self.config['scan_folder'])
        path_layout.addWidget(self.scan_folder_edit, 0, 1)
        scan_browse_btn = QPushButton("浏览...")
        scan_browse_btn.clicked.connect(lambda: self.browse_folder(self.scan_folder_edit))
        path_layout.addWidget(scan_browse_btn, 0, 2)
        
        # Review输出文件夹
        path_layout.addWidget(QLabel("Review输出文件夹:"), 1, 0)
        self.review_folder_edit = QLineEdit(self.config['review_folder'])
        path_layout.addWidget(self.review_folder_edit, 1, 1)
        review_browse_btn = QPushButton("浏览...")
        review_browse_btn.clicked.connect(lambda: self.browse_folder(self.review_folder_edit))
        path_layout.addWidget(review_browse_btn, 1, 2)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # 参数配置组
        param_group = QGroupBox("参数配置")
        param_layout = QGridLayout()
        
        # 扫描间隔
        param_layout.addWidget(QLabel("扫描间隔(秒):"), 0, 0)
        self.scan_interval_spin = QSpinBox()
        self.scan_interval_spin.setRange(1, 3600)
        self.scan_interval_spin.setValue(self.config['scan_interval'])
        param_layout.addWidget(self.scan_interval_spin, 0, 1)
        
        # 置信度阈值
        param_layout.addWidget(QLabel("置信度阈值(%):"), 0, 2)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 100.0)
        self.confidence_spin.setSingleStep(5.0)
        self.confidence_spin.setValue(self.config['confidence_threshold'])
        param_layout.addWidget(self.confidence_spin, 0, 3)
        
        # 文件操作模式
        param_layout.addWidget(QLabel("文件操作:"), 1, 0)
        self.copy_mode_combo = QComboBox()
        self.copy_mode_combo.addItems(["复制", "移动"])
        self.copy_mode_combo.setCurrentText("复制" if self.config['copy_mode'] == 'copy' else "移动")
        param_layout.addWidget(self.copy_mode_combo, 1, 1)
        
        # 重复处理选项
        self.skip_processed_check = QCheckBox("跳过已Review的LotID")
        self.skip_processed_check.setChecked(self.config.get('skip_processed', True))
        self.skip_processed_check.setToolTip("检查Review文件夹，跳过已经处理过的LotID")
        param_layout.addWidget(self.skip_processed_check, 1, 2)
        
        # 自动启动选项
        self.auto_start_check = QCheckBox("程序启动时自动开始")
        self.auto_start_check.setChecked(self.config['auto_start'])
        param_layout.addWidget(self.auto_start_check, 1, 3)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始自动Review")
        self.start_btn.clicked.connect(self.start_auto_review)
        self.start_btn.setMinimumHeight(40)
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.pause_auto_review)
        self.pause_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_auto_review)
        self.stop_btn.setEnabled(False)
        
        self.save_config_btn = QPushButton("保存配置")
        self.save_config_btn.clicked.connect(self.save_config)
        
        self.load_config_btn = QPushButton("加载配置")
        self.load_config_btn.clicked.connect(self.load_config)
        
        self.clear_processed_btn = QPushButton("清理已处理记录")
        self.clear_processed_btn.clicked.connect(self.clear_processed_records)
        self.clear_processed_btn.setToolTip("清理内存中的已处理记录，重新扫描所有文件夹")
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.clear_processed_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.save_config_btn)
        button_layout.addWidget(self.load_config_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
    
    def init_monitor_tab(self, tab_widget):
        """初始化监控标签页"""
        layout = QVBoxLayout(tab_widget)
        
        # 状态显示组
        status_group = QGroupBox("运行状态")
        status_layout = QVBoxLayout()
        
        # 状态标签
        self.status_label = QLabel("未启动")
        self.status_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # 日志显示组
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(1000)  # 限制日志行数
        log_layout.addWidget(self.log_text)
        
        # 日志控制按钮
        log_button_layout = QHBoxLayout()
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_button_layout.addWidget(clear_log_btn)
        log_button_layout.addStretch()
        
        log_layout.addLayout(log_button_layout)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
    
    def init_stats_tab(self, tab_widget):
        """初始化统计标签页"""
        layout = QVBoxLayout(tab_widget)
        
        # 总体统计组
        general_stats_group = QGroupBox("总体统计")
        general_layout = QGridLayout()
        
        self.total_processed_label = QLabel("0")
        self.classified_label = QLabel("0")
        self.unclassified_label = QLabel("0")
        self.errors_label = QLabel("0")
        self.skipped_lots_label = QLabel("0")
        self.runtime_label = QLabel("00:00:00")
        
        general_layout.addWidget(QLabel("总处理数:"), 0, 0)
        general_layout.addWidget(self.total_processed_label, 0, 1)
        general_layout.addWidget(QLabel("已分类:"), 0, 2)
        general_layout.addWidget(self.classified_label, 0, 3)
        general_layout.addWidget(QLabel("未分类:"), 1, 0)
        general_layout.addWidget(self.unclassified_label, 1, 1)
        general_layout.addWidget(QLabel("错误数:"), 1, 2)
        general_layout.addWidget(self.errors_label, 1, 3)
        general_layout.addWidget(QLabel("跳过LotID:"), 2, 0)
        general_layout.addWidget(self.skipped_lots_label, 2, 1)
        general_layout.addWidget(QLabel("运行时间:"), 2, 2)
        general_layout.addWidget(self.runtime_label, 2, 3)
        
        general_stats_group.setLayout(general_layout)
        layout.addWidget(general_stats_group)
        
        # 分类统计表
        class_stats_group = QGroupBox("分类统计")
        class_layout = QVBoxLayout()
        
        self.class_stats_table = QTableWidget()
        self.class_stats_table.setColumnCount(2)
        self.class_stats_table.setHorizontalHeaderLabels(["缺陷类别", "数量"])
        self.class_stats_table.horizontalHeader().setStretchLastSection(True)
        class_layout.addWidget(self.class_stats_table)
        
        class_stats_group.setLayout(class_layout)
        layout.addWidget(class_stats_group)
    
    def browse_folder(self, line_edit):
        """浏览文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder:
            line_edit.setText(folder)
    
    def update_config_from_ui(self):
        """从UI更新配置"""
        self.config['scan_folder'] = self.scan_folder_edit.text()
        self.config['review_folder'] = self.review_folder_edit.text()
        self.config['scan_interval'] = self.scan_interval_spin.value()
        self.config['confidence_threshold'] = self.confidence_spin.value()
        self.config['copy_mode'] = 'copy' if self.copy_mode_combo.currentText() == '复制' else 'move'
        self.config['auto_start'] = self.auto_start_check.isChecked()
        self.config['skip_processed'] = self.skip_processed_check.isChecked()
    
    def start_auto_review(self):
        """开始自动Review"""
        # 检查模型是否已加载
        if not self.main_window or not hasattr(self.main_window, 'worker') or \
           not self.main_window.worker.predictor.model:
            QMessageBox.warning(self, "警告", "请先在预测标签页中加载模型！")
            return
        
        # 更新配置
        self.update_config_from_ui()
        
        # 验证配置
        if not self.config['review_folder']:
            QMessageBox.warning(self, "警告", "请选择Review输出文件夹！")
            return
        
        # 创建并启动线程
        self.auto_review_thread = AutoReviewThread(
            self.main_window.worker.predictor, 
            self.config
        )
        
        # 连接信号
        self.auto_review_thread.status_updated.connect(self.on_status_updated)
        self.auto_review_thread.progress_updated.connect(self.on_progress_updated)
        self.auto_review_thread.file_processed.connect(self.on_file_processed)
        self.auto_review_thread.error_occurred.connect(self.on_error_occurred)
        self.auto_review_thread.statistics_updated.connect(self.on_statistics_updated)
        
        # 启动线程
        self.auto_review_thread.start()
        
        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] 自动Review服务已启动")
    
    def pause_auto_review(self):
        """暂停自动Review"""
        if self.auto_review_thread:
            self.auto_review_thread.pause_processing()
            self.pause_btn.setText("恢复")
            self.pause_btn.clicked.disconnect()
            self.pause_btn.clicked.connect(self.resume_auto_review)
    
    def resume_auto_review(self):
        """恢复自动Review"""
        if self.auto_review_thread:
            self.auto_review_thread.resume_processing()
            self.pause_btn.setText("暂停")
            self.pause_btn.clicked.disconnect()
            self.pause_btn.clicked.connect(self.pause_auto_review)
    
    def stop_auto_review(self):
        """停止自动Review"""
        if self.auto_review_thread:
            self.auto_review_thread.stop_processing()
            self.auto_review_thread.wait(5000)  # 等待5秒
            self.auto_review_thread = None
        
        # 更新UI状态
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("暂停")
        self.pause_btn.clicked.disconnect()
        self.pause_btn.clicked.connect(self.pause_auto_review)
        
        self.status_label.setText("已停止")
        self.progress_bar.setValue(0)
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] 自动Review服务已停止")
    
    def on_status_updated(self, status):
        """状态更新"""
        self.status_label.setText(status)
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {status}")
    
    def on_progress_updated(self, progress):
        """进度更新"""
        self.progress_bar.setValue(progress)
    
    def on_file_processed(self, result):
        """文件处理完成"""
        msg = (f"处理完成: {result['recipe_id']}/{result['lot_id']}/{result['wafer_id']} -> {result['class_name']} "
               f"(置信度: {result['probability']:.2f}%)")
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    
    def on_error_occurred(self, error):
        """错误发生"""
        self.log_text.append(f"[{time.strftime('%H:%M:%S')}] ❌ 错误: {error}")
        self.status_label.setText("运行中 (有错误)")
    
    def on_statistics_updated(self, stats):
        """统计信息更新"""
        self.total_processed_label.setText(str(stats['total_processed']))
        self.classified_label.setText(str(stats['classified']))
        self.unclassified_label.setText(str(stats['unclassified']))
        self.errors_label.setText(str(stats['errors']))
        self.skipped_lots_label.setText(str(stats['skipped_lots']))
        
        # 更新运行时间
        if stats['start_time']:
            runtime = time.time() - stats['start_time']
            hours = int(runtime // 3600)
            minutes = int((runtime % 3600) // 60)
            seconds = int(runtime % 60)
            self.runtime_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # 更新分类统计表
        self.class_stats_table.setRowCount(len(stats['class_counts']))
        for i, (class_name, count) in enumerate(stats['class_counts'].items()):
            self.class_stats_table.setItem(i, 0, QTableWidgetItem(class_name))
            self.class_stats_table.setItem(i, 1, QTableWidgetItem(str(count)))
    
    def save_config(self):
        """保存配置"""
        self.update_config_from_ui()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存配置文件", "auto_review_config.json", 
            "JSON文件 (*.json);;所有文件 (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                QMessageBox.information(self, "成功", "配置文件保存成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存配置文件失败: {str(e)}")
    
    def load_config(self):
        """加载配置"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载配置文件", "", 
            "JSON文件 (*.json);;所有文件 (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                
                # 更新UI
                self.scan_folder_edit.setText(self.config.get('scan_folder', ''))
                self.review_folder_edit.setText(self.config.get('review_folder', ''))
                self.scan_interval_spin.setValue(self.config.get('scan_interval', 10))
                self.confidence_spin.setValue(self.config.get('confidence_threshold', 80.0))
                copy_mode_text = "复制" if self.config.get('copy_mode', 'copy') == 'copy' else "移动"
                self.copy_mode_combo.setCurrentText(copy_mode_text)
                self.auto_start_check.setChecked(self.config.get('auto_start', False))
                self.skip_processed_check.setChecked(self.config.get('skip_processed', True))
                
                QMessageBox.information(self, "成功", "配置文件加载成功！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载配置文件失败: {str(e)}")
    
    def clear_processed_records(self):
        """清理已处理记录"""
        if self.auto_review_thread:
            reply = QMessageBox.question(
                self, "确认清理", 
                "清理已处理记录后，下次扫描时会重新检查所有文件夹。\n"
                "如果服务正在运行，建议先停止服务。\n\n"
                "是否继续清理？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if hasattr(self.auto_review_thread, 'processed_files'):
                    self.auto_review_thread.processed_files.clear()
                    self.log_text.append(f"[{time.strftime('%H:%M:%S')}] 已清理内存中的处理记录")
                    QMessageBox.information(self, "完成", "已清理处理记录！")
        else:
            QMessageBox.information(self, "提示", "当前没有运行的Review服务。")
    
    def closeEvent(self, event):
        """关闭事件处理"""
        if self.auto_review_thread and self.auto_review_thread.isRunning():
            self.stop_auto_review()
        event.accept() 
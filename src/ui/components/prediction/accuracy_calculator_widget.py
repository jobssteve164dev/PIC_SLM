from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                             QFileDialog, QMessageBox, QProgressBar, QTextEdit,
                             QGridLayout, QHeaderView, QTabWidget, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
import os
import json
from collections import defaultdict, Counter
import re
from difflib import SequenceMatcher
from src.utils.logger import get_logger
from src.utils.config_path import get_config_file_path


class AccuracyCalculationThread(QThread):
    """准确率计算线程"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    calculation_finished = pyqtSignal(dict)
    calculation_error = pyqtSignal(str)
    
    def __init__(self, source_folder, output_folder, class_names=None):
        super().__init__()
        self.source_folder = source_folder
        self.output_folder = output_folder
        self.class_names = class_names or []
        self.logger = get_logger(__name__, "accuracy_calculation")
        
        # 尝试从配置文件加载类别信息
        if not self.class_names:
            self.class_names = self._load_class_names_from_config()
        
    def _load_class_names_from_config(self):
        """从配置文件加载类别名称"""
        try:
            config_file = get_config_file_path()
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # 尝试多种可能的类别配置源
                class_names = []
                
                # 1. 从default_classes获取
                if 'default_classes' in config and config['default_classes']:
                    class_names = config['default_classes']
                    self.logger.info(f"从default_classes加载类别: {class_names}")
                
                # 2. 从class_weights的键获取
                elif 'class_weights' in config and config['class_weights']:
                    class_names = list(config['class_weights'].keys())
                    self.logger.info(f"从class_weights加载类别: {class_names}")
                
                # 3. 从defect_classes获取
                elif 'defect_classes' in config and config['defect_classes']:
                    class_names = config['defect_classes']
                    self.logger.info(f"从defect_classes加载类别: {class_names}")
                
                return class_names
                
        except Exception as e:
            self.logger.warning(f"从配置文件加载类别失败: {str(e)}")
            
        return []
        
    def run(self):
        """执行准确率计算"""
        try:
            self.status_updated.emit("开始分析文件...")
            
            # 获取源文件夹中的图片信息
            source_images = self._get_source_images()
            if not source_images:
                self.calculation_error.emit("源文件夹中没有找到有效的图片文件")
                return
            
            self.status_updated.emit(f"找到 {len(source_images)} 张源图片")
            
            # 获取输出文件夹中的图片信息
            output_images = self._get_output_images()
            if not output_images:
                self.calculation_error.emit("输出文件夹中没有找到分类后的图片")
                return
            
            self.status_updated.emit(f"找到 {len(output_images)} 张输出图片")
            
            # 计算准确率
            results = self._calculate_accuracy(source_images, output_images)
            
            self.calculation_finished.emit(results)
            
        except Exception as e:
            import traceback
            error_msg = f"准确率计算出错: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            self.calculation_error.emit(error_msg)
    
    def _get_source_images(self):
        """获取源文件夹中的图片信息"""
        source_images = {}
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        total_files = 0
        for root, _, files in os.walk(self.source_folder):
            total_files += len([f for f in files if f.lower().endswith(supported_formats)])
        
        processed = 0
        for root, _, files in os.walk(self.source_folder):
            for file in files:
                if file.lower().endswith(supported_formats):
                    # 使用改进的类别提取算法
                    true_class = self._extract_class_from_filename_smart(file)
                    if true_class:
                        file_path = os.path.join(root, file)
                        source_images[file] = {
                            'path': file_path,
                            'true_class': true_class,
                            'relative_path': os.path.relpath(file_path, self.source_folder)
                        }
                    
                    processed += 1
                    if total_files > 0:
                        progress = int((processed / total_files) * 50)  # 前50%进度
                        self.progress_updated.emit(progress)
        
        return source_images
    
    def _get_output_images(self):
        """获取输出文件夹中的图片信息"""
        output_images = {}
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        
        total_files = 0
        for root, _, files in os.walk(self.output_folder):
            total_files += len([f for f in files if f.lower().endswith(supported_formats)])
        
        processed = 0
        for root, _, files in os.walk(self.output_folder):
            for file in files:
                if file.lower().endswith(supported_formats):
                    # 从文件夹路径中提取预测类别
                    predicted_class = self._extract_class_from_path(root)
                    if predicted_class:
                        file_path = os.path.join(root, file)
                        output_images[file] = {
                            'path': file_path,
                            'predicted_class': predicted_class,
                            'relative_path': os.path.relpath(file_path, self.output_folder)
                        }
                    
                    processed += 1
                    if total_files > 0:
                        progress = 50 + int((processed / total_files) * 50)  # 后50%进度
                        self.progress_updated.emit(progress)
        
        return output_images
    
    def _extract_class_from_filename_smart(self, filename):
        """智能类别提取算法 - 支持字符串相似度匹配"""
        # 移除文件扩展名
        name_without_ext = os.path.splitext(filename)[0]
        
        # 如果有已知的类别列表，使用智能匹配
        if self.class_names:
            return self._smart_class_matching(name_without_ext, self.class_names)
        
        # 如果没有类别列表，使用传统的模式匹配
        return self._extract_class_from_filename_traditional(name_without_ext)
    
    def _smart_class_matching(self, filename, class_names):
        """智能类别匹配算法 - 修复版本：优先精确匹配"""
        filename_upper = filename.upper()
        best_match = None
        best_score = 0.0
        match_details = []
        
        # 如果提取的类别太短（单字符），直接拒绝
        if len(filename.strip()) <= 1:
            self.logger.warning(f"单字符类别提取被拒绝: {filename}")
            return None
        
        # 如果包含明显的"噪音"关键词，直接拒绝
        noise_keywords = ['UNKNOWN', 'TEST', 'SAMPLE', 'TEMP', 'DEBUG', 'CLASS']
        if any(noise in filename_upper for noise in noise_keywords):
            self.logger.warning(f"噪音关键词类别被拒绝: {filename}")
            return None
        
        # 先去掉文件名中的数字和括号，获得核心类名
        filename_core = re.sub(r'[\d\(\)\s_]+$', '', filename_upper).rstrip('_')
        
        # 对每个类别进行全面的相似度分析
        for class_name in class_names:
            class_upper = class_name.upper()
            max_similarity = 0.0
            best_match_type = ""
            
            # 1. 核心类名精确匹配（最高优先级）
            if filename_core == class_upper:
                max_similarity = 1.0
                best_match_type = f"核心精确匹配"
            
            # 2. 完整精确匹配
            elif filename_upper == class_upper:
                max_similarity = 0.99
                best_match_type = f"完整精确匹配"
            
            # 3. 核心类名包含关系检查（高优先级）
            elif class_upper == filename_core:
                max_similarity = 0.95
                best_match_type = f"核心类名匹配"
            
            # 4. 传统包含关系检查
            elif class_upper in filename_upper:
                # 计算精确度：类名在文件名中的占比
                precision = len(class_upper) / len(filename_upper)
                # 关键修复：提高包含匹配的基础分数
                max_similarity = 0.9 + precision * 0.05  # 0.9-0.95 之间
                best_match_type = f"包含匹配(精确度:{precision:.3f})"
            
            elif filename_upper in class_upper:
                # 反向包含匹配：文件名完全包含在类名中
                precision = len(filename_upper) / len(class_upper)
                max_similarity = 0.85 + precision * 0.05  # 0.85-0.9 之间
                best_match_type = f"反向包含匹配(精确度:{precision:.3f})"
            
            else:
                # 5. 字符串相似度（最低优先级）
                full_similarity = SequenceMatcher(None, filename_upper, class_upper).ratio()
                # 关键修复：降低纯相似度匹配的分数，确保它低于包含匹配
                max_similarity = full_similarity * 0.8  # 最高只能到0.8
                best_match_type = f"相似度匹配({full_similarity:.3f})"
            
            # 记录匹配详情
            match_details.append({
                'class_name': class_name,
                'similarity': max_similarity,
                'match_type': best_match_type,
                'class_length': len(class_upper),
                'length_diff': abs(len(class_upper) - len(filename_upper))
            })
        
        # 排序策略：优先相似度，然后优先较短的类名（避免过度匹配）
        match_details.sort(key=lambda x: (-x['similarity'], x['class_length'], x['length_diff']))
        
        # 记录详细的匹配分析
        self.logger.debug(f"文件名: {filename} (核心: {filename_core}) 的匹配分析:")
        for i, detail in enumerate(match_details[:3]):  # 只显示前3个最佳匹配
            self.logger.debug(f"  {i+1}. {detail['class_name']}: {detail['similarity']:.3f} ({detail['match_type']})")
        
        if match_details:
            best_match_detail = match_details[0]
            best_score = best_match_detail['similarity']
            best_match = best_match_detail['class_name']
            
            # 使用合理的阈值
            threshold = 0.4
            
            if best_score >= threshold:
                self.logger.debug(f"最终匹配: {filename} -> {best_match} (相似度: {best_score:.3f})")
                return best_match
        
        # 如果所有方法都失败，记录并返回None
        self.logger.warning(f"无法匹配类别: {filename} (最高相似度: {best_score:.3f}, 可用类别: {class_names})")
        return None
    
    def _extract_identifiers_from_filename(self, filename):
        """从文件名中提取可能的类别标识符"""
        identifiers = []
        
        # 提取模式
        patterns = [
            r'([A-Z][A-Z_]*[A-Z])',  # 大写字母组合，如 MOUSE_BITE
            r'([A-Z]+)',             # 连续大写字母，如 SPUR
            r'([A-Za-z]+)(?=\d)',    # 字母后跟数字，如 Missing123
            r'([A-Za-z]+)(?=_)',     # 字母后跟下划线，如 Open_
            r'([A-Za-z]+)(?=-)',     # 字母后跟连字符，如 Short-
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, filename)
            for match in matches:
                if len(match) >= 2:  # 至少2个字符
                    identifiers.append(match)
        
        # 去重并按长度排序（长的优先）
        identifiers = list(set(identifiers))
        identifiers.sort(key=len, reverse=True)
        
        return identifiers
    
    def _extract_class_from_filename_traditional(self, filename):
        """传统的类别提取算法（作为备选方案）"""
        # 尝试多种模式匹配，按优先级排序
        patterns = [
            # 模式1: 数字_类别_数字格式，如 01_spur_07 -> spur
            r'^\d+_([A-Za-z_]+?)_\d+.*',
            # 模式2: 类别_数字格式，如 spur_07 -> spur
            r'^([A-Za-z_]+?)_\d+.*',
            # 模式3: 数字_类别格式，如 01_spur -> spur
            r'^\d+_([A-Za-z_]+?).*',
            # 模式4: 字母+数字组合，如 A123, B456
            r'^([A-Za-z_]+?)\d+.*',
            # 模式5: 字母+括号数字，如 A(1), B(2)
            r'^([A-Za-z_]+?)\(\d+\).*',
            # 模式6: 字母+下划线，如 A_001, B_test
            r'^([A-Za-z_]+?)_.*',
            # 模式7: 字母+连字符，如 A-001, B-test
            r'^([A-Za-z_]+?)-.*',
            # 模式8: 字母+点，如 A.001, B.test
            r'^([A-Za-z_]+?)\..*',
            # 模式9: 纯字母开头，如 Apple123, Banana456
            r'^([A-Za-z_]+?)[\d_\-\(\)\.].*',
            # 模式10: 任何字母序列开头
            r'^([A-Za-z_]+)',
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.match(pattern, filename)
            if match:
                class_name = match.group(1).upper()
                # 清理类别名称（移除尾部的下划线）
                class_name = class_name.rstrip('_')
                self.logger.debug(f"传统匹配: {filename} 使用模式 {i+1}: {pattern} -> {class_name}")
                return class_name
        
        # 如果所有模式都不匹配，返回None
        self.logger.warning(f"传统匹配失败: {filename}")
        return None
    
    def _extract_class_from_path(self, path):
        """从文件夹路径中提取类别信息"""
        # 获取相对于输出文件夹的路径
        rel_path = os.path.relpath(path, self.output_folder)
        
        # 分割路径，取最后一个文件夹名作为类别
        path_parts = rel_path.split(os.sep)
        if path_parts and path_parts[-1] != '.':
            predicted_class = path_parts[-1]
            
            # 如果有已知的类别列表，尝试匹配
            if self.class_names:
                # 尝试精确匹配
                for class_name in self.class_names:
                    if class_name.upper() == predicted_class.upper():
                        return class_name
                
                # 尝试部分匹配
                for class_name in self.class_names:
                    if class_name.upper() in predicted_class.upper() or predicted_class.upper() in class_name.upper():
                        return class_name
            
            return predicted_class
        
        return None
    
    def _validate_class_extraction(self, source_images, output_images):
        """验证类别提取的有效性"""
        # 统计源文件夹中的类别
        source_classes = set()
        failed_extractions = []
        
        for filename, info in source_images.items():
            if info['true_class']:
                source_classes.add(info['true_class'])
            else:
                failed_extractions.append(filename)
        
        # 统计输出文件夹中的类别
        output_classes = set()
        for filename, info in output_images.items():
            if info['predicted_class']:
                output_classes.add(info['predicted_class'])
        
        # 生成验证报告
        validation_report = {
            'source_classes': sorted(source_classes),
            'output_classes': sorted(output_classes),
            'failed_extractions': failed_extractions,
            'common_classes': sorted(source_classes & output_classes),
            'source_only_classes': sorted(source_classes - output_classes),
            'output_only_classes': sorted(output_classes - source_classes)
        }
        
        return validation_report
    
    def _calculate_accuracy(self, source_images, output_images):
        """计算准确率"""
        self.status_updated.emit("正在验证类别提取...")
        
        # 验证类别提取的有效性
        validation_report = self._validate_class_extraction(source_images, output_images)
        
        self.status_updated.emit("正在计算准确率...")
        
        # 统计结果
        total_images = len(source_images)
        matched_images = 0
        class_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'predicted_as': defaultdict(int)})
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        # 遍历源图片，查找对应的输出图片
        for filename, source_info in source_images.items():
            true_class = source_info['true_class']
            if not true_class:  # 跳过无法提取类别的图片
                continue
                
            class_stats[true_class]['total'] += 1
            
            if filename in output_images:
                predicted_class = output_images[filename]['predicted_class']
                if predicted_class:
                    class_stats[true_class]['predicted_as'][predicted_class] += 1
                    confusion_matrix[true_class][predicted_class] += 1
                    
                    if true_class == predicted_class:
                        matched_images += 1
                        class_stats[true_class]['correct'] += 1
                else:
                    # 预测类别提取失败
                    class_stats[true_class]['predicted_as']['提取失败'] += 1
                    confusion_matrix[true_class]['提取失败'] += 1
            else:
                # 图片未被分类（可能因为置信度太低）
                class_stats[true_class]['predicted_as']['未分类'] += 1
                confusion_matrix[true_class]['未分类'] += 1
        
        # 计算总体准确率
        valid_images = sum(stats['total'] for stats in class_stats.values())
        overall_accuracy = (matched_images / valid_images * 100) if valid_images > 0 else 0
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for class_name, stats in class_stats.items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                class_accuracies[class_name] = accuracy
        
        # 构建结果
        results = {
            'total_images': total_images,
            'valid_images': valid_images,
            'matched_images': matched_images,
            'overall_accuracy': overall_accuracy,
            'class_stats': dict(class_stats),
            'class_accuracies': class_accuracies,
            'confusion_matrix': dict(confusion_matrix),
            'unprocessed_images': total_images - len(output_images),
            'validation_report': validation_report
        }
        
        return results


class AccuracyCalculatorWidget(QWidget):
    """准确率计算器组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.logger = get_logger(__name__, "accuracy_calculator")
        self.calculation_thread = None
        self.last_results = None
        self.class_names = []  # 存储类别名称
        
        self.init_ui()
    
    def set_class_names(self, class_names):
        """设置类别名称列表"""
        self.class_names = class_names if class_names else []
        self.logger.info(f"设置类别名称: {self.class_names}")
    
    def get_class_names_from_config(self):
        """从配置文件获取类别名称"""
        try:
            config_file = get_config_file_path()
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # 尝试多种可能的类别配置源
                if 'default_classes' in config and config['default_classes']:
                    return config['default_classes']
                elif 'class_weights' in config and config['class_weights']:
                    return list(config['class_weights'].keys())
                elif 'defect_classes' in config and config['defect_classes']:
                    return config['defect_classes']
                    
        except Exception as e:
            self.logger.warning(f"从配置文件获取类别名称失败: {str(e)}")
            
        return []
    
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("批量预测准确率分析")
        title_label.setFont(QFont('微软雅黑', 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 文件夹选择组
        folder_group = QGroupBox("文件夹选择")
        folder_layout = QGridLayout()
        
        # 源文件夹
        folder_layout.addWidget(QLabel("源文件夹:"), 0, 0)
        self.source_folder_edit = QLabel("未选择")
        self.source_folder_edit.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.source_folder_edit.setMinimumHeight(25)
        self.source_folder_edit.setStyleSheet("padding: 5px;")
        folder_layout.addWidget(self.source_folder_edit, 0, 1)
        
        source_btn = QPushButton("浏览...")
        source_btn.clicked.connect(self.select_source_folder)
        folder_layout.addWidget(source_btn, 0, 2)
        
        # 输出文件夹
        folder_layout.addWidget(QLabel("输出文件夹:"), 1, 0)
        self.output_folder_edit = QLabel("未选择")
        self.output_folder_edit.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.output_folder_edit.setMinimumHeight(25)
        self.output_folder_edit.setStyleSheet("padding: 5px;")
        folder_layout.addWidget(self.output_folder_edit, 1, 1)
        
        output_btn = QPushButton("浏览...")
        output_btn.clicked.connect(self.select_output_folder)
        folder_layout.addWidget(output_btn, 1, 2)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        
        self.calculate_btn = QPushButton("开始计算准确率")
        self.calculate_btn.clicked.connect(self.start_calculation)
        self.calculate_btn.setEnabled(False)
        self.calculate_btn.setMinimumHeight(35)
        button_layout.addWidget(self.calculate_btn)
        
        self.export_btn = QPushButton("导出结果")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("请选择源文件夹和输出文件夹")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # 结果显示区域
        self.results_widget = QTabWidget()
        self.results_widget.setVisible(False)
        
        # 总体统计标签页
        self.overall_widget = QWidget()
        self.init_overall_tab()
        self.results_widget.addTab(self.overall_widget, "总体统计")
        
        # 类别详情标签页
        self.class_details_widget = QWidget()
        self.init_class_details_tab()
        self.results_widget.addTab(self.class_details_widget, "类别详情")
        
        # 混淆矩阵标签页
        self.confusion_matrix_widget = QWidget()
        self.init_confusion_matrix_tab()
        self.results_widget.addTab(self.confusion_matrix_widget, "混淆矩阵")
        
        # 验证报告标签页
        self.validation_widget = QWidget()
        self.init_validation_tab()
        self.results_widget.addTab(self.validation_widget, "验证报告")
        
        layout.addWidget(self.results_widget)
        
        # 初始化属性
        self.source_folder = ""
        self.output_folder = ""
    
    def init_overall_tab(self):
        """初始化总体统计标签页"""
        layout = QVBoxLayout(self.overall_widget)
        
        # 总体统计信息
        self.overall_stats_label = QLabel()
        self.overall_stats_label.setFont(QFont('微软雅黑', 10))
        self.overall_stats_label.setAlignment(Qt.AlignTop)
        self.overall_stats_label.setWordWrap(True)
        layout.addWidget(self.overall_stats_label)
        
        layout.addStretch()
    
    def init_class_details_tab(self):
        """初始化类别详情标签页"""
        layout = QVBoxLayout(self.class_details_widget)
        
        # 类别准确率表格
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(4)
        self.class_table.setHorizontalHeaderLabels(['类别', '总数', '正确数', '准确率'])
        self.class_table.horizontalHeader().setStretchLastSection(True)
        self.class_table.setAlternatingRowColors(True)
        layout.addWidget(self.class_table)
    
    def init_confusion_matrix_tab(self):
        """初始化混淆矩阵标签页"""
        layout = QVBoxLayout(self.confusion_matrix_widget)
        
        # 混淆矩阵表格
        self.confusion_table = QTableWidget()
        self.confusion_table.setAlternatingRowColors(True)
        layout.addWidget(self.confusion_table)
        
        # 说明文字
        help_label = QLabel("说明：行表示真实类别，列表示预测类别。对角线上的数字表示正确预测的数量。")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(help_label)
    
    def init_validation_tab(self):
        """初始化验证报告标签页"""
        layout = QVBoxLayout(self.validation_widget)
        
        # 验证报告显示
        self.validation_report_label = QLabel()
        self.validation_report_label.setFont(QFont('微软雅黑', 9))
        self.validation_report_label.setAlignment(Qt.AlignTop)
        self.validation_report_label.setWordWrap(True)
        self.validation_report_label.setStyleSheet("padding: 10px; background-color: #f5f5f5; border: 1px solid #ddd;")
        layout.addWidget(self.validation_report_label)
        
        layout.addStretch()
    
    def select_source_folder(self):
        """选择源文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择源文件夹")
        if folder:
            self.source_folder = folder
            self.source_folder_edit.setText(folder)
            self.check_ready()
    
    def select_output_folder(self):
        """选择输出文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_folder = folder
            self.output_folder_edit.setText(folder)
            self.check_ready()
    
    def check_ready(self):
        """检查是否准备就绪"""
        ready = bool(self.source_folder and os.path.exists(self.source_folder) and
                    self.output_folder and os.path.exists(self.output_folder))
        self.calculate_btn.setEnabled(ready)
        
        if ready:
            self.status_label.setText("准备就绪，可以开始计算准确率")
        else:
            self.status_label.setText("请选择源文件夹和输出文件夹")
    
    def start_calculation(self):
        """开始计算准确率"""
        if not self.source_folder or not self.output_folder:
            QMessageBox.warning(self, "警告", "请先选择源文件夹和输出文件夹")
            return
        
        # 获取类别名称
        class_names = self.class_names or self.get_class_names_from_config()
        
        # 创建计算线程
        self.calculation_thread = AccuracyCalculationThread(
            self.source_folder, self.output_folder, class_names
        )
        
        # 连接信号
        self.calculation_thread.progress_updated.connect(self.update_progress)
        self.calculation_thread.status_updated.connect(self.update_status)
        self.calculation_thread.calculation_finished.connect(self.show_results)
        self.calculation_thread.calculation_error.connect(self.show_error)
        
        # 更新UI状态
        self.calculate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_widget.setVisible(False)
        
        # 启动线程
        self.calculation_thread.start()
    
    def update_progress(self, value):
        """更新进度"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """更新状态"""
        self.status_label.setText(message)
    
    def show_results(self, results):
        """显示计算结果"""
        self.last_results = results
        
        # 更新总体统计
        self.update_overall_stats(results)
        
        # 更新类别详情
        self.update_class_details(results)
        
        # 更新混淆矩阵
        self.update_confusion_matrix(results)
        
        # 更新验证报告
        self.update_validation_report(results)
        
        # 显示结果区域
        self.results_widget.setVisible(True)
        self.export_btn.setEnabled(True)
        
        # 隐藏进度条
        self.progress_bar.setVisible(False)
        
        # 重新启用计算按钮
        self.calculate_btn.setEnabled(True)
        
        # 更新状态
        self.status_label.setText(f"计算完成！总体准确率: {results['overall_accuracy']:.2f}%")
    
    def update_overall_stats(self, results):
        """更新总体统计信息"""
        stats_text = f"""<h3>总体统计</h3>
        <p><b>总图片数:</b> {results['total_images']}</p>
        <p><b>正确预测数:</b> {results['matched_images']}</p>
        <p><b>总体准确率:</b> <span style="color: {'green' if results['overall_accuracy'] >= 80 else 'orange' if results['overall_accuracy'] >= 60 else 'red'}; font-size: 14pt; font-weight: bold;">{results['overall_accuracy']:.2f}%</span></p>
        <p><b>未处理图片数:</b> {results['unprocessed_images']}</p>
        """
        
        if results['class_accuracies']:
            stats_text += "<h3>各类别准确率</h3>"
            for class_name, accuracy in sorted(results['class_accuracies'].items()):
                color = 'green' if accuracy >= 80 else 'orange' if accuracy >= 60 else 'red'
                stats_text += f"<p><b>{class_name}:</b> <span style='color: {color};'>{accuracy:.2f}%</span></p>"
        
        self.overall_stats_label.setText(stats_text)
    
    def update_class_details(self, results):
        """更新类别详情表格"""
        class_stats = results['class_stats']
        
        self.class_table.setRowCount(len(class_stats))
        
        for row, (class_name, stats) in enumerate(sorted(class_stats.items())):
            # 类别名称
            self.class_table.setItem(row, 0, QTableWidgetItem(class_name))
            
            # 总数
            self.class_table.setItem(row, 1, QTableWidgetItem(str(stats['total'])))
            
            # 正确数
            self.class_table.setItem(row, 2, QTableWidgetItem(str(stats['correct'])))
            
            # 准确率
            accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
            accuracy_item = QTableWidgetItem(f"{accuracy:.2f}%")
            
            # 根据准确率设置颜色
            if accuracy >= 80:
                accuracy_item.setBackground(QColor(200, 255, 200))
            elif accuracy >= 60:
                accuracy_item.setBackground(QColor(255, 255, 200))
            else:
                accuracy_item.setBackground(QColor(255, 200, 200))
            
            self.class_table.setItem(row, 3, accuracy_item)
        
        # 调整列宽
        self.class_table.resizeColumnsToContents()
    
    def update_confusion_matrix(self, results):
        """更新混淆矩阵"""
        confusion_matrix = results['confusion_matrix']
        
        if not confusion_matrix:
            return
        
        # 获取所有类别（包括预测类别）
        all_classes = set()
        for true_class, predictions in confusion_matrix.items():
            all_classes.add(true_class)
            all_classes.update(predictions.keys())
        
        all_classes = sorted(all_classes)
        
        # 设置表格大小
        self.confusion_table.setRowCount(len(all_classes))
        self.confusion_table.setColumnCount(len(all_classes))
        
        # 设置标题
        self.confusion_table.setHorizontalHeaderLabels(all_classes)
        self.confusion_table.setVerticalHeaderLabels(all_classes)
        
        # 填充数据
        for i, true_class in enumerate(all_classes):
            for j, pred_class in enumerate(all_classes):
                count = confusion_matrix.get(true_class, {}).get(pred_class, 0)
                item = QTableWidgetItem(str(count))
                
                # 对角线元素（正确预测）用绿色背景
                if i == j and count > 0:
                    item.setBackground(QColor(200, 255, 200))
                elif count > 0:
                    item.setBackground(QColor(255, 240, 240))
                
                item.setTextAlignment(Qt.AlignCenter)
                self.confusion_table.setItem(i, j, item)
        
        # 调整列宽
        self.confusion_table.resizeColumnsToContents()
    
    def update_validation_report(self, results):
        """更新验证报告"""
        validation_report = results.get('validation_report', {})
        
        report_text = "<h3>📋 类别提取验证报告</h3>"
        
        # 基本统计
        report_text += f"<h4>📊 基本统计</h4>"
        report_text += f"<p><b>总图片数:</b> {results['total_images']}</p>"
        report_text += f"<p><b>有效图片数:</b> {results['valid_images']}</p>"
        report_text += f"<p><b>类别提取失败:</b> {results['total_images'] - results['valid_images']} 张</p>"
        
        # 类别分析
        source_classes = validation_report.get('source_classes', [])
        output_classes = validation_report.get('output_classes', [])
        common_classes = validation_report.get('common_classes', [])
        
        report_text += f"<h4>🏷️ 类别分析</h4>"
        report_text += f"<p><b>源文件夹发现的类别:</b> {', '.join(source_classes) if source_classes else '无'}</p>"
        report_text += f"<p><b>输出文件夹发现的类别:</b> {', '.join(output_classes) if output_classes else '无'}</p>"
        report_text += f"<p><b>共同类别:</b> {', '.join(common_classes) if common_classes else '无'}</p>"
        
        # 不匹配的类别
        source_only = validation_report.get('source_only_classes', [])
        output_only = validation_report.get('output_only_classes', [])
        
        if source_only:
            report_text += f"<p><b>⚠️ 仅在源文件夹中发现:</b> <span style='color: orange;'>{', '.join(source_only)}</span></p>"
        if output_only:
            report_text += f"<p><b>⚠️ 仅在输出文件夹中发现:</b> <span style='color: orange;'>{', '.join(output_only)}</span></p>"
        
        # 提取失败的文件
        failed_extractions = validation_report.get('failed_extractions', [])
        if failed_extractions:
            report_text += f"<h4>❌ 类别提取失败的文件</h4>"
            report_text += f"<p><b>失败数量:</b> {len(failed_extractions)}</p>"
            if len(failed_extractions) <= 10:
                report_text += f"<p><b>文件列表:</b></p>"
                for filename in failed_extractions:
                    report_text += f"<p style='margin-left: 20px; color: red;'>• {filename}</p>"
            else:
                report_text += f"<p><b>前10个失败文件:</b></p>"
                for filename in failed_extractions[:10]:
                    report_text += f"<p style='margin-left: 20px; color: red;'>• {filename}</p>"
                report_text += f"<p style='margin-left: 20px; color: gray;'>... 还有 {len(failed_extractions) - 10} 个文件</p>"
        
        # 文件名模式建议
        report_text += f"<h4>💡 文件名模式建议</h4>"
        report_text += f"<p>为了提高类别提取的准确性，建议使用以下文件名格式：</p>"
        report_text += f"<ul>"
        report_text += f"<li><b>01_spur_07.jpg</b> - 数字_类别_数字（推荐用于复杂命名）</li>"
        report_text += f"<li><b>spur_07.jpg</b> - 类别_数字</li>"
        report_text += f"<li><b>01_spur.jpg</b> - 数字_类别</li>"
        report_text += f"<li><b>A123.jpg</b> - 字母+数字</li>"
        report_text += f"<li><b>A(1).jpg</b> - 字母+括号数字</li>"
        report_text += f"<li><b>A_001.jpg</b> - 字母+下划线+数字</li>"
        report_text += f"<li><b>A-001.jpg</b> - 字母+连字符+数字</li>"
        report_text += f"</ul>"
        
        self.validation_report_label.setText(report_text)
    
    def show_error(self, error_message):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", error_message)
        self.progress_bar.setVisible(False)
        self.calculate_btn.setEnabled(True)
        self.status_label.setText("计算失败")
    
    def export_results(self):
        """导出结果"""
        if not self.last_results:
            QMessageBox.warning(self, "警告", "没有可导出的结果")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", f"accuracy_report.json", "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.last_results, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "成功", f"结果已导出到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
    
    def set_folders_from_parent(self, source_folder, output_folder):
        """从父组件设置文件夹路径"""
        if source_folder and os.path.exists(source_folder):
            self.source_folder = source_folder
            self.source_folder_edit.setText(source_folder)
        
        if output_folder and os.path.exists(output_folder):
            self.output_folder = output_folder
            self.output_folder_edit.setText(output_folder)
        
        self.check_ready() 
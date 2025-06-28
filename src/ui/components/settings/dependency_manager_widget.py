"""
依赖管理组件
提供依赖检查、安装和代理设置功能
"""

import os
import sys
import subprocess
import threading
import json
from typing import Dict, List, Optional, Tuple
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QLineEdit, QPushButton, QTextEdit, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QCheckBox, QComboBox, QMessageBox,
    QSplitter, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette
import importlib.util
import pkg_resources


class DependencyCheckThread(QThread):
    """依赖检查线程"""
    
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    dependency_checked = pyqtSignal(str, bool, str)  # 包名, 是否安装, 版本信息
    finished_check = pyqtSignal(list)  # 检查结果列表
    
    def __init__(self):
        super().__init__()
        self.requirements_file = "requirements.txt"
        self.scan_mode = "requirements"  # "requirements" 或 "code_analysis" 或 "both"
        
    def set_scan_mode(self, mode: str):
        """设置扫描模式
        Args:
            mode: "requirements" - 只扫描requirements.txt
                 "code_analysis" - 只扫描代码中的import
                 "both" - 扫描两者并合并
        """
        self.scan_mode = mode
        
    def run(self):
        """执行依赖检查"""
        try:
            self.status_updated.emit("正在读取依赖信息...")
            
            if self.scan_mode == "requirements":
                dependencies = self._parse_requirements()
            elif self.scan_mode == "code_analysis": 
                dependencies = self._analyze_code_imports()
            else:  # both
                req_deps = self._parse_requirements()
                code_deps = self._analyze_code_imports()
                dependencies = self._merge_dependencies(req_deps, code_deps)
            
            total = len(dependencies)
            results = []
            
            for i, (pkg_name, version_spec, source) in enumerate(dependencies):
                if self.isInterruptionRequested():
                    break
                    
                self.status_updated.emit(f"检查依赖: {pkg_name}")
                
                installed, version = self._check_package(pkg_name)
                results.append({
                    'name': pkg_name, 
                    'required_version': version_spec,
                    'installed': installed,
                    'current_version': version,
                    'source': source  # 'requirements', 'code', 'both'
                })
                
                self.dependency_checked.emit(pkg_name, installed, version)
                self.progress_updated.emit(int((i + 1) / total * 100))
            
            self.finished_check.emit(results)
            self.status_updated.emit("依赖检查完成")
            
        except Exception as e:
            self.status_updated.emit(f"检查失败: {str(e)}")
    
    def _parse_requirements(self) -> List[Tuple[str, str, str]]:
        """解析requirements.txt文件"""
        dependencies = []
        
        if not os.path.exists(self.requirements_file):
            return dependencies
        
        with open(self.requirements_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '==' in line:
                        pkg_name, version = line.split('==')
                        dependencies.append((pkg_name.strip(), version.strip(), 'requirements'))
                    elif ';' in line:  # 处理条件依赖
                        pkg_part = line.split(';')[0].strip()
                        if '==' in pkg_part:
                            pkg_name, version = pkg_part.split('==')
                            dependencies.append((pkg_name.strip(), version.strip(), 'requirements'))
                        else:
                            dependencies.append((pkg_part.strip(), "", 'requirements'))
                    else:
                        dependencies.append((line.strip(), "", 'requirements'))
        
        return dependencies
    
    def _analyze_code_imports(self) -> List[Tuple[str, str, str]]:
        """分析代码中的import语句，提取实际使用的依赖"""
        dependencies = set()
        
        # 需要扫描的目录
        scan_dirs = ['src']
        
        # 标准库模块（不需要安装的）
        stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'threading', 'subprocess', 
            'pathlib', 'glob', 'shutil', 'platform', 'traceback', 'logging', 
            'functools', 'enum', 'dataclasses', 'typing', 'collections', 'itertools',
            'math', 'random', 're', 'urllib', 'http', 'socket', 'ssl', 'hashlib',
            'base64', 'pickle', 'copy', 'weakref', 'gc', 'warnings', 'configparser',
            'sqlite3', 'csv', 'xml', 'html', 'email', 'mimetypes', 'tempfile',
            'io', 'codecs', 'locale', 'calendar', 'zoneinfo', 'gzip', 'zipfile',
            'tarfile', 'bz2', 'lzma', 'unittest', 'argparse', 'getopt', 'pdb'
        }
        
        # 包名映射（import名到pip包名的映射）
        package_mapping = {
            'cv2': 'opencv-python',
            'PIL': 'pillow', 
            'skimage': 'scikit-image',
            'sklearn': 'scikit-learn',
            'yaml': 'PyYAML',
            'QtWidgets': 'PyQt5',
            'QtCore': 'PyQt5',
            'QtGui': 'PyQt5',
            'torchvision': 'torchvision',
            'tensorboard': 'tensorboard',
            'efficientnet_pytorch': 'efficientnet-pytorch',
            'timm': 'timm',
            'shap': 'shap',
            'lime': 'lime',
            'captum': 'captum',
            'albumentations': 'albumentations',
            'seaborn': 'seaborn',
            'tqdm': 'tqdm',
            'requests': 'requests',
            'psutil': 'psutil',
            'colorama': 'colorama',
            'mplcursors': 'mplcursors'
        }
        
        import_count = 0
        
        for scan_dir in scan_dirs:
            if not os.path.exists(scan_dir):
                continue
                
            for root, dirs, files in os.walk(scan_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        imports = self._extract_imports_from_file(file_path)
                        
                        for imp in imports:
                            import_count += 1
                            if import_count % 50 == 0:  # 每50个import更新一次状态
                                self.status_updated.emit(f"分析代码导入... ({import_count} 个import)")
                                
                            # 过滤掉标准库模块
                            if imp not in stdlib_modules and not imp.startswith('.'):
                                # 应用包名映射
                                package_name = package_mapping.get(imp, imp)
                                dependencies.add(package_name)
        
        # 转换为列表格式
        result = [(pkg, "", "code") for pkg in sorted(dependencies)]
        return result
    
    def _extract_imports_from_file(self, file_path: str) -> set:
        """从单个Python文件中提取import的包名"""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # 解析 import xxx
                    if line.startswith('import ') and not line.startswith('import.'):
                        parts = line.split()
                        if len(parts) >= 2:
                            module = parts[1].split('.')[0]  # 只取顶级包名
                            imports.add(module)
                    
                    # 解析 from xxx import
                    elif line.startswith('from ') and ' import ' in line:
                        try:
                            from_part = line.split(' import ')[0]
                            module = from_part.replace('from ', '').strip().split('.')[0]
                            # 过滤相对导入
                            if not module.startswith('.') and module:
                                imports.add(module)
                        except:
                            continue
                            
        except Exception as e:
            # 如果文件读取失败，跳过该文件
            pass
            
        return imports
    
    def _merge_dependencies(self, req_deps: List[Tuple[str, str, str]], 
                          code_deps: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """合并requirements.txt和代码分析的依赖"""
        merged = {}
        
        # 添加requirements.txt中的依赖
        for pkg, version, source in req_deps:
            merged[pkg] = (pkg, version, source)
        
        # 添加代码中发现的依赖
        for pkg, version, source in code_deps:
            if pkg in merged:
                # 如果已存在，标记为both
                existing_pkg, existing_version, existing_source = merged[pkg]
                merged[pkg] = (existing_pkg, existing_version, 'both')
            else:
                # 新发现的依赖
                merged[pkg] = (pkg, version, 'code_only')
        
        return list(merged.values())
    
    def _check_package(self, package_name: str) -> Tuple[bool, str]:
        """检查单个包是否已安装"""
        try:
            # 尝试使用pkg_resources检查
            dist = pkg_resources.get_distribution(package_name)
            return True, dist.version
        except pkg_resources.DistributionNotFound:
            try:
                # 尝试导入模块
                spec = importlib.util.find_spec(package_name)
                if spec is not None:
                    return True, "未知版本"
                else:
                    return False, ""
            except ImportError:
                return False, ""


class DependencyInstallThread(QThread):
    """依赖安装线程"""
    
    progress_updated = pyqtSignal(str)  # 安装进度信息
    install_finished = pyqtSignal(bool, str)  # 安装完成, 成功/失败, 消息
    
    def __init__(self, packages: List[str], proxy_url: str = "", use_index: bool = False):
        super().__init__()
        self.packages = packages
        self.proxy_url = proxy_url
        self.use_index = use_index
        
    def run(self):
        """执行依赖安装"""
        try:
            for package in self.packages:
                self.progress_updated.emit(f"正在安装: {package}")
                
                # 构建pip安装命令
                cmd = [sys.executable, "-m", "pip", "install", package]
                
                # 添加代理设置
                if self.proxy_url and self.use_index:
                    cmd.extend(["-i", self.proxy_url])
                elif self.proxy_url:
                    cmd.extend(["--proxy", self.proxy_url])
                
                # 执行安装
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    encoding='utf-8'
                )
                
                if result.returncode != 0:
                    error_msg = f"安装 {package} 失败:\n{result.stderr}"
                    self.install_finished.emit(False, error_msg)
                    return
                else:
                    self.progress_updated.emit(f"成功安装: {package}")
            
            self.install_finished.emit(True, "所有依赖安装完成")
            
        except Exception as e:
            self.install_finished.emit(False, f"安装过程中出错: {str(e)}")


class DependencyManagerWidget(QWidget):
    """依赖管理组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dependencies_data = []
        self.init_ui()
        self.load_proxy_settings()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 创建代理设置组
        self._create_proxy_settings_group(layout)
        
        # 创建依赖管理组
        self._create_dependency_management_group(layout)
        
        # 创建分割器
        splitter = QSplitter(Qt.Vertical)
        
        # 创建依赖列表
        dependency_frame = self._create_dependency_list_frame()
        splitter.addWidget(dependency_frame)
        
        # 创建日志显示
        log_frame = self._create_log_frame()
        splitter.addWidget(log_frame)
        
        # 设置分割器比例
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        
    def _create_proxy_settings_group(self, parent_layout):
        """创建代理设置组"""
        proxy_group = QGroupBox("代理设置")
        proxy_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        proxy_layout = QVBoxLayout(proxy_group)
        
        # 启用代理复选框
        self.enable_proxy_checkbox = QCheckBox("启用代理")
        self.enable_proxy_checkbox.toggled.connect(self.on_proxy_enabled_changed)
        proxy_layout.addWidget(self.enable_proxy_checkbox)
        
        # 代理类型选择
        proxy_type_layout = QHBoxLayout()
        proxy_type_layout.addWidget(QLabel("代理类型:"))
        
        self.proxy_type_combo = QComboBox()
        self.proxy_type_combo.addItems(["索引代理 (-i)", "HTTP代理 (--proxy)"])
        self.proxy_type_combo.setToolTip(
            "索引代理 (-i): 指定PyPI镜像源，如清华源、阿里源等\n"
            "HTTP代理 (--proxy): 指定HTTP/HTTPS代理服务器"
        )
        proxy_type_layout.addWidget(self.proxy_type_combo)
        proxy_type_layout.addStretch()
        proxy_layout.addLayout(proxy_type_layout)
        
        # 代理地址输入
        proxy_url_layout = QHBoxLayout()
        proxy_url_layout.addWidget(QLabel("代理地址:"))
        
        self.proxy_url_input = QLineEdit()
        self.proxy_url_input.setPlaceholderText("例如: https://pypi.tuna.tsinghua.edu.cn/simple/")
        self.proxy_url_input.textChanged.connect(self.save_proxy_settings)
        proxy_url_layout.addWidget(self.proxy_url_input)
        
        proxy_layout.addLayout(proxy_url_layout)
        
        # 常用代理快捷按钮
        shortcuts_layout = QHBoxLayout()
        shortcuts_layout.addWidget(QLabel("常用镜像:"))
        
        shortcuts = [
            ("清华源", "https://pypi.tuna.tsinghua.edu.cn/simple/"),
            ("阿里源", "https://mirrors.aliyun.com/pypi/simple/"),
            ("豆瓣源", "https://pypi.douban.com/simple/"),
            ("华为源", "https://mirrors.huaweicloud.com/repository/pypi/simple/")
        ]
        
        for name, url in shortcuts:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, u=url: self.set_proxy_url(u))
            btn.setMaximumWidth(80)
            shortcuts_layout.addWidget(btn)
        
        shortcuts_layout.addStretch()
        proxy_layout.addLayout(shortcuts_layout)
        
        # 测试代理按钮
        test_proxy_btn = QPushButton("测试代理连接")
        test_proxy_btn.clicked.connect(self.test_proxy_connection)
        proxy_layout.addWidget(test_proxy_btn)
        
        parent_layout.addWidget(proxy_group)
        
    def _create_dependency_management_group(self, parent_layout):
        """创建依赖管理操作组"""
        management_group = QGroupBox("依赖管理")
        management_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        management_layout = QVBoxLayout(management_group)
        
        # 扫描模式选择
        scan_mode_layout = QHBoxLayout()
        scan_mode_layout.addWidget(QLabel("扫描模式:"))
        
        self.scan_mode_combo = QComboBox()
        self.scan_mode_combo.addItems([
            "requirements.txt", 
            "代码分析", 
            "智能扫描（推荐）"
        ])
        self.scan_mode_combo.setCurrentIndex(2)  # 默认选择智能扫描
        self.scan_mode_combo.setToolTip(
            "requirements.txt: 只检查requirements.txt中记录的依赖\n"
            "代码分析: 扫描代码中实际使用的import语句\n"
            "智能扫描: 合并两种方式，发现所有依赖"
        )
        scan_mode_layout.addWidget(self.scan_mode_combo)
        scan_mode_layout.addStretch()
        management_layout.addLayout(scan_mode_layout)
        
        # 操作按钮行
        buttons_layout = QHBoxLayout()
        
        self.check_dependencies_btn = QPushButton("检查依赖")
        self.check_dependencies_btn.clicked.connect(self.check_dependencies)
        self.check_dependencies_btn.setToolTip("根据选择的扫描模式检查依赖状态")
        buttons_layout.addWidget(self.check_dependencies_btn)
        
        self.install_missing_btn = QPushButton("安装缺失依赖")
        self.install_missing_btn.clicked.connect(self.install_missing_dependencies)
        self.install_missing_btn.setEnabled(False)
        self.install_missing_btn.setToolTip("安装所有未安装的依赖包")
        buttons_layout.addWidget(self.install_missing_btn)
        
        self.install_selected_btn = QPushButton("安装选中依赖")
        self.install_selected_btn.clicked.connect(self.install_selected_dependencies)
        self.install_selected_btn.setEnabled(False)
        self.install_selected_btn.setToolTip("安装在表格中选中的依赖包")
        buttons_layout.addWidget(self.install_selected_btn)
        
        self.refresh_btn = QPushButton("刷新列表")
        self.refresh_btn.clicked.connect(self.check_dependencies)
        buttons_layout.addWidget(self.refresh_btn)
        
        buttons_layout.addStretch()
        management_layout.addLayout(buttons_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        management_layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666;")
        management_layout.addWidget(self.status_label)
        
        parent_layout.addWidget(management_group)
        
    def _create_dependency_list_frame(self) -> QFrame:
        """创建依赖列表框架"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题
        title_label = QLabel("依赖列表")
        title_label.setFont(QFont('微软雅黑', 10, QFont.Bold))
        layout.addWidget(title_label)
        
        # 依赖表格
        self.dependencies_table = QTableWidget()
        self.dependencies_table.setColumnCount(6)
        self.dependencies_table.setHorizontalHeaderLabels([
            "选择", "包名", "要求版本", "当前版本", "状态", "来源"
        ])
        
        # 设置表格属性
        header = self.dependencies_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        
        self.dependencies_table.setColumnWidth(0, 50)
        self.dependencies_table.setAlternatingRowColors(True)
        self.dependencies_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        layout.addWidget(self.dependencies_table)
        
        return frame
        
    def _create_log_frame(self) -> QFrame:
        """创建日志显示框架"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题和清空按钮
        title_layout = QHBoxLayout()
        title_label = QLabel("安装日志")
        title_label.setFont(QFont('微软雅黑', 10, QFont.Bold))
        title_layout.addWidget(title_label)
        
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.clear_log)
        clear_log_btn.setMaximumWidth(80)
        title_layout.addWidget(clear_log_btn)
        
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont('Consolas', 9))
        layout.addWidget(self.log_text)
        
        return frame
        
    def on_proxy_enabled_changed(self, enabled: bool):
        """代理启用状态改变"""
        controls = [
            self.proxy_type_combo,
            self.proxy_url_input
        ]
        
        for control in controls:
            control.setEnabled(enabled)
            
        self.save_proxy_settings()
        
    def set_proxy_url(self, url: str):
        """设置代理URL"""
        self.proxy_url_input.setText(url)
        self.enable_proxy_checkbox.setChecked(True)
        
    def test_proxy_connection(self):
        """测试代理连接"""
        if not self.enable_proxy_checkbox.isChecked():
            QMessageBox.information(self, "提示", "请先启用代理设置")
            return
            
        proxy_url = self.proxy_url_input.text().strip()
        if not proxy_url:
            QMessageBox.warning(self, "警告", "请输入代理地址")
            return
            
        self.add_log("正在测试代理连接...")
        
        # 在新线程中测试连接
        threading.Thread(target=self._test_proxy_in_thread, args=(proxy_url,)).start()
        
    def _test_proxy_in_thread(self, proxy_url: str):
        """在线程中测试代理连接"""
        try:
            import requests
            
            if self.proxy_type_combo.currentIndex() == 0:  # 索引代理
                # 测试访问索引页面
                response = requests.get(proxy_url, timeout=10)
                if response.status_code == 200:
                    self.add_log(f"✓ 代理连接成功: {proxy_url}")
                else:
                    self.add_log(f"✗ 代理连接失败，状态码: {response.status_code}")
            else:  # HTTP代理
                proxies = {'http': proxy_url, 'https': proxy_url}
                response = requests.get('https://pypi.org', proxies=proxies, timeout=10)
                if response.status_code == 200:
                    self.add_log(f"✓ HTTP代理连接成功: {proxy_url}")
                else:
                    self.add_log(f"✗ HTTP代理连接失败，状态码: {response.status_code}")
                    
        except Exception as e:
            self.add_log(f"✗ 代理连接测试失败: {str(e)}")
            
    def check_dependencies(self):
        """检查依赖"""
        self.status_label.setText("正在检查依赖...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.check_dependencies_btn.setEnabled(False)
        
        # 清空表格
        self.dependencies_table.setRowCount(0)
        self.dependencies_data.clear()
        
        # 确定扫描模式
        mode_map = {
            0: "requirements",      # requirements.txt
            1: "code_analysis",     # 代码分析
            2: "both"              # 智能扫描
        }
        scan_mode = mode_map[self.scan_mode_combo.currentIndex()]
        
        # 启动检查线程
        self.check_thread = DependencyCheckThread()
        self.check_thread.set_scan_mode(scan_mode)
        self.check_thread.progress_updated.connect(self.progress_bar.setValue)
        self.check_thread.status_updated.connect(self.status_label.setText)
        self.check_thread.dependency_checked.connect(self.add_dependency_to_table)
        self.check_thread.finished_check.connect(self.on_check_finished)
        self.check_thread.start()
        
    def add_dependency_to_table(self, name: str, installed: bool, version: str):
        """添加依赖到表格"""
        row = self.dependencies_table.rowCount()
        self.dependencies_table.insertRow(row)
        
        # 选择复选框
        checkbox = QCheckBox()
        if not installed:
            checkbox.setChecked(True)  # 默认选中未安装的包
        self.dependencies_table.setCellWidget(row, 0, checkbox)
        
        # 包名
        self.dependencies_table.setItem(row, 1, QTableWidgetItem(name))
        
        # 要求版本（从dependencies_data中获取）
        required_version = ""
        source = ""
        for dep in self.dependencies_data:
            if dep['name'] == name:
                required_version = dep.get('required_version', '')
                source = dep.get('source', '')
                break
        self.dependencies_table.setItem(row, 2, QTableWidgetItem(required_version))
        
        # 当前版本
        current_version_item = QTableWidgetItem(version if version else "未安装")
        if not installed:
            current_version_item.setForeground(QColor('red'))
        self.dependencies_table.setItem(row, 3, current_version_item)
        
        # 状态
        status_item = QTableWidgetItem("已安装" if installed else "未安装")
        if installed:
            status_item.setForeground(QColor('green'))
        else:
            status_item.setForeground(QColor('red'))
        self.dependencies_table.setItem(row, 4, status_item)
        
        # 来源
        source_display = {
            'requirements': 'requirements.txt',
            'code': '代码导入',
            'both': '两者',
            'code_only': '仅代码'
        }
        source_item = QTableWidgetItem(source_display.get(source, source))
        if source in ['code_only']:
            source_item.setForeground(QColor('orange'))
            source_item.setToolTip("此依赖在代码中使用但未在requirements.txt中记录")
        elif source == 'code':
            source_item.setForeground(QColor('blue'))
        self.dependencies_table.setItem(row, 5, source_item)
        
    def on_check_finished(self, results: List[Dict]):
        """检查完成回调"""
        self.dependencies_data = results
        self.progress_bar.setVisible(False)
        self.check_dependencies_btn.setEnabled(True)
        
        # 统计结果
        total = len(results)
        installed = sum(1 for r in results if r['installed'])
        missing = total - installed
        
        self.status_label.setText(f"检查完成: 总共 {total} 个依赖, 已安装 {installed} 个, 缺失 {missing} 个")
        
        # 启用安装按钮
        if missing > 0:
            self.install_missing_btn.setEnabled(True)
            self.install_selected_btn.setEnabled(True)
            
    def install_missing_dependencies(self):
        """安装所有缺失的依赖"""
        missing_packages = [dep['name'] for dep in self.dependencies_data if not dep['installed']]
        if not missing_packages:
            QMessageBox.information(self, "提示", "没有缺失的依赖需要安装")
            return
            
        self._install_packages(missing_packages)
        
    def install_selected_dependencies(self):
        """安装选中的依赖"""
        selected_packages = []
        
        for row in range(self.dependencies_table.rowCount()):
            checkbox = self.dependencies_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                package_name = self.dependencies_table.item(row, 1).text()
                selected_packages.append(package_name)
                
        if not selected_packages:
            QMessageBox.information(self, "提示", "请先选择要安装的依赖")
            return
            
        self._install_packages(selected_packages)
        
    def _install_packages(self, packages: List[str]):
        """安装指定的包列表"""
        if not packages:
            return
            
        # 获取代理设置
        proxy_url = ""
        use_index = False
        
        if self.enable_proxy_checkbox.isChecked():
            proxy_url = self.proxy_url_input.text().strip()
            use_index = (self.proxy_type_combo.currentIndex() == 0)
            
        # 确认安装
        message = f"即将安装以下 {len(packages)} 个依赖包:\n\n"
        message += "\n".join(f"• {pkg}" for pkg in packages)
        if proxy_url:
            proxy_type = "索引代理" if use_index else "HTTP代理"
            message += f"\n\n使用{proxy_type}: {proxy_url}"
            
        reply = QMessageBox.question(
            self, "确认安装", message,
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
            
        # 禁用按钮
        self.install_missing_btn.setEnabled(False)
        self.install_selected_btn.setEnabled(False)
        self.check_dependencies_btn.setEnabled(False)
        
        self.add_log(f"开始安装 {len(packages)} 个依赖包...")
        
        # 启动安装线程
        self.install_thread = DependencyInstallThread(packages, proxy_url, use_index)
        self.install_thread.progress_updated.connect(self.add_log)
        self.install_thread.install_finished.connect(self.on_install_finished)
        self.install_thread.start()
        
    def on_install_finished(self, success: bool, message: str):
        """安装完成回调"""
        self.add_log(message)
        
        # 重新启用按钮
        self.install_missing_btn.setEnabled(True)
        self.install_selected_btn.setEnabled(True)
        self.check_dependencies_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "成功", "依赖安装完成！")
            # 重新检查依赖状态
            QTimer.singleShot(1000, self.check_dependencies)
        else:
            QMessageBox.critical(self, "失败", f"依赖安装失败:\n{message}")
            
    def add_log(self, message: str):
        """添加日志信息"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # 自动滚动到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)
        
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
        
    def load_proxy_settings(self):
        """加载代理设置"""
        from ....utils.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        proxy_config = config_manager.get_config_item("proxy_settings", {})
        
        self.enable_proxy_checkbox.setChecked(proxy_config.get("enabled", False))
        self.proxy_type_combo.setCurrentIndex(proxy_config.get("type", 0))
        self.proxy_url_input.setText(proxy_config.get("url", ""))
        
    def save_proxy_settings(self):
        """保存代理设置"""
        from ....utils.config_manager import ConfigManager
        
        proxy_config = {
            "enabled": self.enable_proxy_checkbox.isChecked(),
            "type": self.proxy_type_combo.currentIndex(),
            "url": self.proxy_url_input.text().strip()
        }
        
        config_manager = ConfigManager()
        config_manager.set_config_item("proxy_settings", proxy_config, save_immediately=False)
        
    def get_config(self) -> Dict:
        """获取当前配置"""
        return {
            "proxy_settings": {
                "enabled": self.enable_proxy_checkbox.isChecked(),
                "type": self.proxy_type_combo.currentIndex(),
                "url": self.proxy_url_input.text().strip()
            }
        }
        
    def apply_config(self, config: Dict):
        """应用配置"""
        proxy_config = config.get("proxy_settings", {})
        
        self.enable_proxy_checkbox.setChecked(proxy_config.get("enabled", False))
        self.proxy_type_combo.setCurrentIndex(proxy_config.get("type", 0))
        self.proxy_url_input.setText(proxy_config.get("url", "")) 
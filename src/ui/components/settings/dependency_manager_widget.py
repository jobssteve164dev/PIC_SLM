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
    QSplitter, QFrame, QDialog, QDialogButtonBox, QFormLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette
import importlib.util
import pkg_resources


class CustomMirrorDialog(QDialog):
    """自定义镜像配置对话框"""
    
    def __init__(self, parent=None, existing_config=None):
        super().__init__(parent)
        self.setWindowTitle("自定义镜像配置")
        self.setModal(True)
        self.setFixedSize(500, 200)
        
        # 初始化配置
        self.config = existing_config or {
            "name": "",
            "url": "",
            "host": ""
        }
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 表单布局
        form_layout = QFormLayout()
        
        # 镜像名称
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("例如: 公司内网源")
        self.name_input.setText(self.config.get("name", ""))
        form_layout.addRow("镜像名称:", self.name_input)
        
        # 镜像地址
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("例如: https://pypi.company.com/simple/")
        self.url_input.setText(self.config.get("url", ""))
        form_layout.addRow("镜像地址:", self.url_input)
        
        # 信任主机
        self.host_input = QLineEdit()
        self.host_input.setPlaceholderText("例如: pypi.company.com")
        self.host_input.setText(self.config.get("host", ""))
        form_layout.addRow("信任主机:", self.host_input)
        
        layout.addLayout(form_layout)
        
        # 说明文本
        help_label = QLabel("提示：信任主机用于跳过SSL证书验证，通常是镜像地址的域名部分")
        help_label.setStyleSheet("color: #666; font-size: 12px; margin: 10px 0;")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        
        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
    def get_config(self) -> Dict:
        """获取配置"""
        return {
            "name": self.name_input.text().strip(),
            "url": self.url_input.text().strip(),
            "host": self.host_input.text().strip()
        }
        
    def accept(self):
        """确认按钮处理"""
        config = self.get_config()
        
        # 验证输入
        if not config["name"]:
            QMessageBox.warning(self, "警告", "请输入镜像名称")
            return
            
        if not config["url"]:
            QMessageBox.warning(self, "警告", "请输入镜像地址")
            return
            
        # 验证URL格式
        if not (config["url"].startswith("http://") or config["url"].startswith("https://")):
            QMessageBox.warning(self, "警告", "镜像地址必须以http://或https://开头")
            return
            
        super().accept()


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
        
        # 本地模块（项目内部的模块，不需要安装）
        local_modules = self._get_local_modules()
        
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
                                
                            # 过滤掉标准库模块和本地模块
                            if (imp not in stdlib_modules and 
                                imp not in local_modules and 
                                not imp.startswith('.') and
                                self._is_valid_package_name(imp)):
                                # 应用包名映射
                                package_name = package_mapping.get(imp, imp)
                                dependencies.add(package_name)
        
        # 转换为列表格式
        result = [(pkg, "", "code") for pkg in sorted(dependencies)]
        return result
    
    def _get_local_modules(self) -> set:
        """获取项目本地模块列表"""
        local_modules = set()
        
        # 扫描项目根目录下的所有文件夹（这些都是项目内部的）
        project_dirs = set()
        try:
            for item in os.listdir('.'):
                if os.path.isdir(item) and not item.startswith('.'):
                    project_dirs.add(item)
        except:
            pass
        
        # 扫描src目录下的所有模块
        if os.path.exists('src'):
            for root, dirs, files in os.walk('src'):
                # 跳过__pycache__目录
                dirs[:] = [d for d in dirs if d != '__pycache__']
                
                for file in files:
                    if file.endswith('.py') and file != '__init__.py':
                        # 获取相对于src的模块路径
                        rel_path = os.path.relpath(os.path.join(root, file), 'src')
                        module_name = rel_path.replace(os.sep, '.').replace('.py', '')
                        # 只取顶级模块名
                        top_level = module_name.split('.')[0]
                        local_modules.add(top_level)
                
                # 添加目录作为模块（如果包含__init__.py）
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if os.path.exists(os.path.join(dir_path, '__init__.py')):
                        rel_path = os.path.relpath(dir_path, 'src')
                        module_name = rel_path.replace(os.sep, '.')
                        top_level = module_name.split('.')[0]
                        local_modules.add(top_level)
        
        # 添加项目根目录下的所有文件夹
        local_modules.update(project_dirs)
        
        # 手动添加已知的项目模块和常见项目文件夹
        known_local_modules = {
            'src', 'ui', 'utils', 'components', 'training_components', 
            'image_processing', 'model_trainer', 'predictor', 'config_loader',
            'data_processor', 'detection_trainer', 'detection_utils', 
            'image_preprocessor', 'annotation_tool', 'backup', 'logs', 'models',
            'config', 'setting', 'runs', 'train_config', 'pretrainedmodels',
            'resource', 'test', 'examples', 'docs', 'data', 'datasets',
            'checkpoints', 'weights', 'pretrained', 'cache', 'temp', 'build',
            'dist', 'venv', 'env', '.git', '.vscode', '__pycache__'
        }
        local_modules.update(known_local_modules)
        
        return local_modules
    
    def _is_valid_package_name(self, name: str) -> bool:
        """检查是否是有效的包名"""
        if not name:
            return False
        
        # 过滤无效的包名
        invalid_patterns = [
            # 文件扩展名
            name.endswith(('.py', '.pyx', '.so', '.dll', '.exe', '.bat', '.sh')),
            # 包含路径分隔符
            '/' in name or '\\' in name,
            # 数字开头
            name[0].isdigit(),
            # 包含空格或特殊字符
            any(c in name for c in ' \t\n\r!@#$%^&*()+=[]{}|;:,<>?'),
            # 太短或太长
            len(name) < 2 or len(name) > 50,
            # 常见的非包名和项目相关名称
            name.lower() in {
                'main', 'test', 'tests', 'example', 'examples', 'demo', 'app',
                'config', 'configs', 'setting', 'settings', 'data', 'dataset', 
                'datasets', 'model', 'models', 'log', 'logs', 'backup', 'cache',
                'temp', 'tmp', 'build', 'dist', 'docs', 'doc', 'readme', 'license',
                'scripts', 'script', 'tools', 'tool', 'utils', 'util', 'helpers',
                'helper', 'assets', 'static', 'resources', 'resource', 'weights',
                'checkpoints', 'checkpoint', 'pretrained', 'pretrainedmodels',
                'runs', 'experiments', 'exp', 'results', 'output', 'outputs',
                'input', 'inputs', 'src', 'source', 'lib', 'libs', 'bin'
            },
            # 版本号格式
            name.replace('.', '').replace('-', '').replace('_', '').isdigit(),
        ]
        
        return not any(invalid_patterns)
    
    def _extract_imports_from_file(self, file_path: str) -> set:
        """从单个Python文件中提取import的包名"""
        imports = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # 简单的import语句解析，避免复杂的语法解析
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # 跳过注释行
                if line.startswith('#') or not line:
                    continue
                
                # 解析 import xxx
                if line.startswith('import '):
                    # 处理多个import: import a, b, c
                    import_part = line[7:].strip()  # 去掉'import '
                    if import_part:
                        # 分割逗号分隔的导入
                        modules = [m.strip().split('.')[0] for m in import_part.split(',')]
                        for module in modules:
                            module = module.strip()
                            if module and self._is_valid_import_name(module):
                                imports.add(module)
                
                # 解析 from xxx import
                elif line.startswith('from ') and ' import ' in line:
                    try:
                        from_part = line.split(' import ')[0][5:].strip()  # 去掉'from '
                        if from_part and not from_part.startswith('.'):
                            module = from_part.split('.')[0]
                            if module and self._is_valid_import_name(module):
                                imports.add(module)
                    except:
                        continue
                        
        except Exception as e:
            # 文件读取失败时，记录但不影响整体进程
            self.status_updated.emit(f"警告: 无法解析文件 {os.path.basename(file_path)}")
            
        return imports
    
    def _is_valid_import_name(self, name: str) -> bool:
        """检查是否是有效的import名称"""
        if not name or len(name) < 1:
            return False
        
        # 基本的包名验证
        return (name.replace('_', '').replace('-', '').isalnum() and 
                not name[0].isdigit())
    
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
    
    def __init__(self, packages: List[str], proxy_url: str = "", use_index: bool = False, trusted_hosts: List[str] = None):
        super().__init__()
        self.packages = packages
        self.proxy_url = proxy_url
        self.use_index = use_index
        self.trusted_hosts = trusted_hosts or []
        
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
                
                # 添加信任主机设置
                for trusted_host in self.trusted_hosts:
                    if trusted_host.strip():
                        cmd.extend(["--trusted-host", trusted_host.strip()])
                
                # 添加详细输出和错误处理
                cmd.extend(["--verbose"])
                
                self.progress_updated.emit(f"执行命令: {' '.join(cmd)}")
                
                # 执行安装
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    encoding='utf-8'
                )
                
                if result.returncode != 0:
                    error_msg = f"安装 {package} 失败:\n"
                    error_msg += f"返回码: {result.returncode}\n"
                    error_msg += f"错误输出:\n{result.stderr}"
                    
                    # 分析常见错误类型
                    stderr_lower = result.stderr.lower()
                    if "ssl" in stderr_lower or "certificate" in stderr_lower:
                        error_msg += "\n\nSSL证书错误诊断:"
                        error_msg += "\n1. 请确认已启用'信任主机'选项"
                        error_msg += "\n2. 请检查信任主机设置是否正确"
                        error_msg += "\n3. 如果是公司内网，可能需要使用HTTP而非HTTPS"
                    elif "connection" in stderr_lower or "timeout" in stderr_lower:
                        error_msg += "\n\n连接错误诊断:"
                        error_msg += "\n1. 请检查代理地址是否正确"
                        error_msg += "\n2. 请确认网络连接正常"
                        error_msg += "\n3. 请检查防火墙设置"
                    elif "401" in stderr_lower or "unauthorized" in stderr_lower:
                        error_msg += "\n\n认证错误诊断:"
                        error_msg += "\n1. 请检查代理服务器是否需要认证"
                        error_msg += "\n2. 请确认代理地址格式正确"
                        error_msg += "\n3. 可能需要联系网络管理员"
                    
                    self.install_finished.emit(False, error_msg)
                    return
                else:
                    self.progress_updated.emit(f"成功安装: {package}")
                    if result.stdout:
                        # 显示安装成功的详细信息
                        success_info = result.stdout.strip()
                        if len(success_info) > 200:
                            success_info = success_info[:200] + "..."
                        self.progress_updated.emit(f"安装详情: {success_info}")
            
            self.install_finished.emit(True, "所有依赖安装完成")
            
        except Exception as e:
            error_msg = f"安装过程中出错: {str(e)}"
            error_msg += f"\n错误类型: {type(e).__name__}"
            self.install_finished.emit(False, error_msg)


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
        
        # 信任主机设置
        trusted_host_layout = QVBoxLayout()
        
        # 信任主机复选框
        self.enable_trusted_host_checkbox = QCheckBox("启用信任主机 (--trusted-host)")
        self.enable_trusted_host_checkbox.setToolTip(
            "启用后将跳过SSL证书验证，适用于内网镜像源或HTTP代理\n"
            "注意：这会降低安全性，请谨慎使用"
        )
        self.enable_trusted_host_checkbox.toggled.connect(self.on_trusted_host_enabled_changed)
        self.enable_trusted_host_checkbox.toggled.connect(self.save_proxy_settings)
        trusted_host_layout.addWidget(self.enable_trusted_host_checkbox)
        
        # 信任主机输入框
        trusted_hosts_input_layout = QHBoxLayout()
        trusted_hosts_input_layout.addWidget(QLabel("信任主机:"))
        
        self.trusted_hosts_input = QLineEdit()
        self.trusted_hosts_input.setPlaceholderText("例如: pypi.tuna.tsinghua.edu.cn (多个主机用逗号分隔)")
        self.trusted_hosts_input.setEnabled(False)
        self.trusted_hosts_input.textChanged.connect(self.save_proxy_settings)
        trusted_hosts_input_layout.addWidget(self.trusted_hosts_input)
        
        trusted_host_layout.addLayout(trusted_hosts_input_layout)
        proxy_layout.addLayout(trusted_host_layout)
        
        # 常用代理快捷按钮
        shortcuts_layout = QHBoxLayout()
        shortcuts_layout.addWidget(QLabel("常用镜像:"))
        
        # 预设镜像
        shortcuts = [
            ("清华源", "https://pypi.tuna.tsinghua.edu.cn/simple/", "pypi.tuna.tsinghua.edu.cn"),
            ("阿里源", "https://mirrors.aliyun.com/pypi/simple/", "mirrors.aliyun.com"),
            ("豆瓣源", "https://pypi.douban.com/simple/", "pypi.douban.com"),
            ("华为源", "https://mirrors.huaweicloud.com/repository/pypi/simple/", "mirrors.huaweicloud.com")
        ]
        
        for name, url, host in shortcuts:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, u=url, h=host: self.set_proxy_url_and_host(u, h))
            btn.setMaximumWidth(80)
            shortcuts_layout.addWidget(btn)
        
        # 自定义镜像按钮
        self.custom_mirror_btn = QPushButton("自定义")
        self.custom_mirror_btn.clicked.connect(self.show_custom_mirror_dialog)
        self.custom_mirror_btn.setMaximumWidth(80)
        shortcuts_layout.addWidget(self.custom_mirror_btn)
        
        shortcuts_layout.addStretch()
        proxy_layout.addLayout(shortcuts_layout)
        
        # 测试和诊断按钮
        test_layout = QHBoxLayout()
        
        test_proxy_btn = QPushButton("测试连接")
        test_proxy_btn.clicked.connect(self.test_proxy_connection)
        test_proxy_btn.setMaximumWidth(100)
        test_layout.addWidget(test_proxy_btn)
        
        ssl_diagnose_btn = QPushButton("SSL诊断")
        ssl_diagnose_btn.clicked.connect(self.diagnose_ssl_issues)
        ssl_diagnose_btn.setMaximumWidth(100)
        ssl_diagnose_btn.setToolTip("诊断SSL证书相关问题")
        test_layout.addWidget(ssl_diagnose_btn)
        
        apply_fix_btn = QPushButton("应用修复")
        apply_fix_btn.clicked.connect(self.apply_ssl_fix)
        apply_fix_btn.setMaximumWidth(100)
        apply_fix_btn.setToolTip("自动应用SSL证书修复设置")
        apply_fix_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        test_layout.addWidget(apply_fix_btn)
        
        test_layout.addStretch()
        proxy_layout.addLayout(test_layout)
        
        # 当前自定义镜像显示
        self.custom_mirror_label = QLabel()
        self.custom_mirror_label.setStyleSheet("color: #666; font-size: 12px; margin: 5px 0;")
        self.custom_mirror_label.setVisible(False)
        proxy_layout.addWidget(self.custom_mirror_label)
        
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
        
        # 手动安装依赖组
        manual_group = QGroupBox("手动安装依赖")
        manual_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        manual_layout = QVBoxLayout(manual_group)
        
        # 输入提示
        input_label = QLabel("请输入要安装的依赖库名（多个库用逗号或空格分隔）：")
        input_label.setStyleSheet("color: #333; margin: 5px 0;")
        manual_layout.addWidget(input_label)
        
        # 输入框和按钮的水平布局
        input_hlayout = QHBoxLayout()
        
        # 输入框
        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("例如: numpy pandas matplotlib>=3.0 scikit-learn==1.0.2")
        self.manual_input.returnPressed.connect(self._install_manual_packages)
        self.manual_input.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 3px;")
        input_hlayout.addWidget(self.manual_input)
        
        # 安装按钮
        self.install_manual_btn = QPushButton("安装")
        self.install_manual_btn.clicked.connect(self._install_manual_packages)
        self.install_manual_btn.setStyleSheet("background-color: #2196F3; color: white; padding: 5px 15px; border-radius: 3px;")
        self.install_manual_btn.setMaximumWidth(80)
        input_hlayout.addWidget(self.install_manual_btn)
        
        manual_layout.addLayout(input_hlayout)
        
        # 常用库快捷按钮
        shortcuts_label = QLabel("常用库快捷安装：")
        shortcuts_label.setStyleSheet("color: #333; margin: 10px 0 5px 0;")
        manual_layout.addWidget(shortcuts_label)
        
        shortcuts_layout = QHBoxLayout()
        common_packages = [
            ("NumPy", "numpy"),
            ("Pandas", "pandas"), 
            ("Matplotlib", "matplotlib"),
            ("OpenCV", "opencv-python"),
            ("Scikit-learn", "scikit-learn"),
            ("SHAP", "shap"),
            ("Pillow", "pillow"),
            ("Requests", "requests")
        ]
        
        for display_name, package_name in common_packages:
            btn = QPushButton(display_name)
            btn.clicked.connect(lambda checked, pkg=package_name: self._install_single_package(pkg))
            btn.setStyleSheet("padding: 3px 8px; margin: 2px; border: 1px solid #ddd; border-radius: 3px;")
            btn.setMaximumWidth(90)
            shortcuts_layout.addWidget(btn)
        
        shortcuts_layout.addStretch()
        manual_layout.addLayout(shortcuts_layout)
        
        parent_layout.addWidget(manual_group)
        
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
        self.proxy_type_combo.setEnabled(enabled)
        self.proxy_url_input.setEnabled(enabled)
        
        # 同时控制信任主机相关控件
        self.enable_trusted_host_checkbox.setEnabled(enabled)
        if enabled and self.enable_trusted_host_checkbox.isChecked():
            self.trusted_hosts_input.setEnabled(True)
        else:
            self.trusted_hosts_input.setEnabled(False)
            
        self.save_proxy_settings()
    
    def on_trusted_host_enabled_changed(self, enabled: bool):
        """信任主机启用状态改变"""
        self.trusted_hosts_input.setEnabled(enabled and self.enable_proxy_checkbox.isChecked())
        
    def set_proxy_url(self, url: str):
        """设置代理URL"""
        self.proxy_url_input.setText(url)
        self.save_proxy_settings()
        
    def set_proxy_url_and_host(self, url: str, host: str):
        """设置代理URL和信任主机"""
        self.proxy_url_input.setText(url)
        
        # 如果选择了镜像源，自动启用信任主机并设置对应的主机
        if url.startswith("https://"):
            self.enable_trusted_host_checkbox.setChecked(True)
            self.trusted_hosts_input.setText(host)
        
        self.save_proxy_settings()
        
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
            from urllib3.exceptions import InsecureRequestWarning
            import warnings
            
            # 临时禁用SSL警告
            warnings.filterwarnings('ignore', category=InsecureRequestWarning)
            
            # 获取信任主机设置
            trusted_hosts = []
            if self.enable_trusted_host_checkbox.isChecked():
                trusted_hosts_text = self.trusted_hosts_input.text().strip()
                if trusted_hosts_text:
                    trusted_hosts = [host.strip() for host in trusted_hosts_text.replace(',', ' ').split() if host.strip()]
            
            # 创建自定义session，支持SSL验证跳过
            session = requests.Session()
            
            # 如果启用了信任主机，则跳过SSL验证
            if trusted_hosts:
                session.verify = False
                self.add_log("已启用SSL证书验证跳过模式")
            
            if self.proxy_type_combo.currentIndex() == 0:  # 索引代理
                # 测试访问索引页面
                try:
                    response = session.get(proxy_url, timeout=10)
                    if response.status_code == 200:
                        self.add_log(f"✓ 代理连接成功: {proxy_url}")
                        if trusted_hosts:
                            self.add_log(f"  信任主机: {', '.join(trusted_hosts)}")
                    else:
                        self.add_log(f"✗ 代理连接失败，状态码: {response.status_code}")
                except requests.exceptions.SSLError as ssl_err:
                    self.add_log(f"✗ SSL证书验证失败: {str(ssl_err)}")
                    self.add_log("建议：请启用'信任主机'选项并设置正确的信任主机")
                except requests.exceptions.ConnectionError as conn_err:
                    self.add_log(f"✗ 连接失败: {str(conn_err)}")
                    self.add_log("建议：请检查代理地址是否正确，网络连接是否正常")
                except requests.exceptions.Timeout:
                    self.add_log("✗ 连接超时")
                    self.add_log("建议：请检查网络连接或增加超时时间")
            else:  # HTTP代理
                proxies = {'http': proxy_url, 'https': proxy_url}
                try:
                    response = session.get('https://pypi.org', proxies=proxies, timeout=10)
                    if response.status_code == 200:
                        self.add_log(f"✓ HTTP代理连接成功: {proxy_url}")
                        if trusted_hosts:
                            self.add_log(f"  信任主机: {', '.join(trusted_hosts)}")
                    else:
                        self.add_log(f"✗ HTTP代理连接失败，状态码: {response.status_code}")
                except requests.exceptions.SSLError as ssl_err:
                    self.add_log(f"✗ SSL证书验证失败: {str(ssl_err)}")
                    self.add_log("建议：请启用'信任主机'选项并设置正确的信任主机")
                except requests.exceptions.ConnectionError as conn_err:
                    self.add_log(f"✗ 连接失败: {str(conn_err)}")
                    self.add_log("建议：请检查代理地址是否正确，网络连接是否正常")
                except requests.exceptions.Timeout:
                    self.add_log("✗ 连接超时")
                    self.add_log("建议：请检查网络连接或增加超时时间")
                    
        except Exception as e:
            self.add_log(f"✗ 代理连接测试失败: {str(e)}")
            self.add_log("详细错误信息可能有助于诊断问题")
            
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
    
    def _install_manual_packages(self):
        """安装手动输入的依赖包"""
        input_text = self.manual_input.text().strip()
        if not input_text:
            QMessageBox.information(self, "提示", "请输入要安装的依赖库名")
            return
        
        # 解析输入的包名（支持逗号和空格分隔）
        packages = []
        for item in input_text.replace(',', ' ').split():
            item = item.strip()
            if item:
                packages.append(item)
        
        if not packages:
            QMessageBox.information(self, "提示", "请输入有效的依赖库名")
            return
        
        # 验证包名格式
        invalid_packages = []
        valid_packages = []
        
        for package in packages:
            # 允许版本号规范（如 package>=1.0, package==1.2.3）
            package_name = package.split('=')[0].split('>')[0].split('<')[0].split('!')[0].split('~')[0]
            if self._is_valid_manual_package_name(package_name):
                valid_packages.append(package)
            else:
                invalid_packages.append(package)
        
        if invalid_packages:
            msg = f"以下包名格式无效，将被跳过：\n{', '.join(invalid_packages)}"
            if valid_packages:
                msg += f"\n\n将安装：{', '.join(valid_packages)}"
                reply = QMessageBox.question(self, "包名验证", msg, 
                                           QMessageBox.Yes | QMessageBox.No)
                if reply != QMessageBox.Yes:
                    return
            else:
                QMessageBox.warning(self, "包名验证", msg)
                return
        
        if valid_packages:
            self._install_packages(valid_packages)
            self.manual_input.clear()  # 清空输入框
    
    def _install_single_package(self, package_name):
        """安装单个包（快捷按钮使用）"""
        self._install_packages([package_name])
    
    def _is_valid_manual_package_name(self, name: str) -> bool:
        """检查手动输入的包名是否有效（相对宽松的验证）"""
        if not name:
            return False
        
        # 基本格式检查
        if len(name) < 2 or len(name) > 100:
            return False
        
        # 不能包含路径分隔符和危险字符
        if any(c in name for c in '/\\<>"|*?'):
            return False
        
        # 不能以数字开头
        if name[0].isdigit():
            return False
        
        # 不能是明显的文件名
        if name.endswith(('.py', '.exe', '.bat', '.sh', '.dll', '.so')):
            return False
        
        return True
        
    def _install_packages(self, packages: List[str]):
        """安装指定的包列表"""
        if not packages:
            return
            
        # 获取代理设置
        proxy_url = ""
        use_index = False
        trusted_hosts = []
        
        if self.enable_proxy_checkbox.isChecked():
            proxy_url = self.proxy_url_input.text().strip()
            use_index = (self.proxy_type_combo.currentIndex() == 0)
            
            # 获取信任主机设置
            if self.enable_trusted_host_checkbox.isChecked():
                trusted_hosts_text = self.trusted_hosts_input.text().strip()
                if trusted_hosts_text:
                    # 支持逗号和空格分隔的多个主机
                    trusted_hosts = [host.strip() for host in trusted_hosts_text.replace(',', ' ').split() if host.strip()]
            
        # 确认安装
        message = f"即将安装以下 {len(packages)} 个依赖包:\n\n"
        message += "\n".join(f"• {pkg}" for pkg in packages)
        if proxy_url:
            proxy_type = "索引代理" if use_index else "HTTP代理"
            message += f"\n\n使用{proxy_type}: {proxy_url}"
            if trusted_hosts:
                message += f"\n信任主机: {', '.join(trusted_hosts)}"
            
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
        if proxy_url:
            proxy_type = "索引代理" if use_index else "HTTP代理"
            self.add_log(f"使用{proxy_type}: {proxy_url}")
            if trusted_hosts:
                self.add_log(f"信任主机: {', '.join(trusted_hosts)}")
        
        # 启动安装线程
        self.install_thread = DependencyInstallThread(packages, proxy_url, use_index, trusted_hosts)
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
        
        # 加载信任主机设置
        self.enable_trusted_host_checkbox.setChecked(proxy_config.get("trusted_host_enabled", False))
        self.trusted_hosts_input.setText(proxy_config.get("trusted_hosts", ""))
        
        # 更新控件状态
        self.on_proxy_enabled_changed(self.enable_proxy_checkbox.isChecked())
        
        # 更新自定义镜像显示
        self.update_custom_mirror_display()
        
    def save_proxy_settings(self):
        """保存代理设置"""
        from ....utils.config_manager import ConfigManager
        
        proxy_config = {
            "enabled": self.enable_proxy_checkbox.isChecked(),
            "type": self.proxy_type_combo.currentIndex(),
            "url": self.proxy_url_input.text().strip(),
            "trusted_host_enabled": self.enable_trusted_host_checkbox.isChecked(),
            "trusted_hosts": self.trusted_hosts_input.text().strip()
        }
        
        config_manager = ConfigManager()
        config_manager.set_config_item("proxy_settings", proxy_config, save_immediately=False)
        
    def get_config(self) -> Dict:
        """获取当前配置"""
        from ....utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        custom_mirror_config = config_manager.get_config_item("custom_mirror", {})
        
        return {
            "proxy_settings": {
                "enabled": self.enable_proxy_checkbox.isChecked(),
                "type": self.proxy_type_combo.currentIndex(),
                "url": self.proxy_url_input.text().strip(),
                "trusted_host_enabled": self.enable_trusted_host_checkbox.isChecked(),
                "trusted_hosts": self.trusted_hosts_input.text().strip()
            },
            "custom_mirror": custom_mirror_config
        }
        
    def apply_config(self, config: Dict):
        """应用配置"""
        proxy_config = config.get("proxy_settings", {})
        
        self.enable_proxy_checkbox.setChecked(proxy_config.get("enabled", False))
        self.proxy_type_combo.setCurrentIndex(proxy_config.get("type", 0))
        self.proxy_url_input.setText(proxy_config.get("url", ""))
        self.enable_trusted_host_checkbox.setChecked(proxy_config.get("trusted_host_enabled", False))
        self.trusted_hosts_input.setText(proxy_config.get("trusted_hosts", ""))
        
        # 更新控件状态
        self.on_proxy_enabled_changed(self.enable_proxy_checkbox.isChecked())
        
        # 应用自定义镜像配置
        custom_mirror_config = config.get("custom_mirror", {})
        if custom_mirror_config:
            from ....utils.config_manager import ConfigManager
            config_manager = ConfigManager()
            config_manager.set_config_item("custom_mirror", custom_mirror_config, save_immediately=False)
            
        # 更新自定义镜像显示
        self.update_custom_mirror_display()
        
    def show_custom_mirror_dialog(self):
        """显示自定义镜像配置对话框"""
        # 加载现有的自定义镜像配置
        from ....utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        custom_mirror_config = config_manager.get_config_item("custom_mirror", {})
        
        dialog = CustomMirrorDialog(self, custom_mirror_config)
        
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_config()
            
            # 保存自定义镜像配置
            config_manager.set_config_item("custom_mirror", config, save_immediately=True)
            
            # 应用自定义镜像设置
            self.set_proxy_url_and_host(config["url"], config["host"])
            
            # 更新自定义镜像显示
            self.update_custom_mirror_display()
            
            QMessageBox.information(self, "成功", f"自定义镜像 '{config['name']}' 配置成功！")
    
    def update_custom_mirror_display(self):
        """更新自定义镜像显示"""
        from ....utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        custom_mirror_config = config_manager.get_config_item("custom_mirror", {})
        
        if custom_mirror_config.get("name") and custom_mirror_config.get("url"):
            self.custom_mirror_label.setText(f"自定义镜像: {custom_mirror_config['name']} ({custom_mirror_config['url']})")
            self.custom_mirror_label.setVisible(True)
            # 更新按钮样式，表示已配置
            self.custom_mirror_btn.setStyleSheet("background-color: #2196F3; color: white;")
            self.custom_mirror_btn.setToolTip(f"当前自定义镜像: {custom_mirror_config['name']}\n左键应用，右键编辑")
        else:
            self.custom_mirror_label.setVisible(False)
            # 恢复默认样式，表示未配置
            self.custom_mirror_btn.setStyleSheet("background-color: #4CAF50; color: white;")
            self.custom_mirror_btn.setToolTip("点击配置自定义镜像")
    
    def handle_custom_mirror_click(self):
        """处理自定义镜像按钮点击"""
        from ....utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        custom_mirror_config = config_manager.get_config_item("custom_mirror", {})
        
        # 如果已有自定义镜像配置，直接应用；否则打开配置对话框
        if custom_mirror_config.get("name") and custom_mirror_config.get("url"):
            self.apply_custom_mirror()
        else:
            self.show_custom_mirror_dialog()
    
    def show_custom_mirror_menu(self, position):
        """显示自定义镜像右键菜单"""
        from PyQt5.QtWidgets import QMenu, QAction
        
        menu = QMenu(self)
        
        # 应用自定义镜像
        apply_action = QAction("应用自定义镜像", self)
        apply_action.triggered.connect(self.apply_custom_mirror)
        menu.addAction(apply_action)
        
        # 编辑自定义镜像
        edit_action = QAction("编辑自定义镜像", self)
        edit_action.triggered.connect(self.show_custom_mirror_dialog)
        menu.addAction(edit_action)
        
        # 检查是否有自定义镜像配置
        from ....utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        custom_mirror_config = config_manager.get_config_item("custom_mirror", {})
        
        if not (custom_mirror_config.get("name") and custom_mirror_config.get("url")):
            apply_action.setEnabled(False)
            apply_action.setText("应用自定义镜像 (未配置)")
            
        menu.exec_(self.custom_mirror_btn.mapToGlobal(position))
    
    def apply_custom_mirror(self):
        """应用自定义镜像"""
        from ....utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        custom_mirror_config = config_manager.get_config_item("custom_mirror", {})
        
        if custom_mirror_config.get("name") and custom_mirror_config.get("url"):
            self.set_proxy_url_and_host(custom_mirror_config["url"], custom_mirror_config.get("host", ""))
            self.add_log(f"已应用自定义镜像: {custom_mirror_config['name']}")
        else:
            QMessageBox.information(self, "提示", "尚未配置自定义镜像，请先点击'自定义'按钮进行配置") 
    
    def diagnose_ssl_issues(self):
        """诊断SSL证书问题"""
        self.add_log("=== SSL证书诊断开始 ===")
        
        # 获取当前设置
        proxy_url = self.proxy_url_input.text().strip()
        trusted_hosts_enabled = self.enable_trusted_host_checkbox.isChecked()
        trusted_hosts_text = self.trusted_hosts_input.text().strip()
        
        self.add_log(f"当前代理地址: {proxy_url if proxy_url else '未设置'}")
        self.add_log(f"信任主机启用: {'是' if trusted_hosts_enabled else '否'}")
        self.add_log(f"信任主机列表: {trusted_hosts_text if trusted_hosts_text else '未设置'}")
        
        if not proxy_url:
            self.add_log("✗ 未设置代理地址，无法进行SSL诊断")
            return
        
        # 在新线程中执行诊断
        threading.Thread(target=self._diagnose_ssl_in_thread, args=(proxy_url, trusted_hosts_enabled, trusted_hosts_text)).start()
    
    def _diagnose_ssl_in_thread(self, proxy_url: str, trusted_hosts_enabled: bool, trusted_hosts_text: str):
        """在线程中执行SSL诊断"""
        try:
            import requests
            import ssl
            import socket
            from urllib.parse import urlparse
            from urllib3.exceptions import InsecureRequestWarning
            import warnings
            
            # 临时禁用SSL警告
            warnings.filterwarnings('ignore', category=InsecureRequestWarning)
            
            # 解析URL
            parsed_url = urlparse(proxy_url)
            hostname = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
            
            self.add_log(f"目标主机: {hostname}:{port}")
            self.add_log(f"协议: {parsed_url.scheme}")
            
            # 1. 测试SSL证书
            if parsed_url.scheme == 'https':
                self.add_log("\n--- SSL证书检查 ---")
                try:
                    # 创建SSL上下文
                    context = ssl.create_default_context()
                    
                    # 连接到服务器并获取证书
                    with socket.create_connection((hostname, port), timeout=10) as sock:
                        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                            cert = ssock.getpeercert()
                            
                            self.add_log("✓ SSL连接成功")
                            self.add_log(f"证书主题: {cert.get('subject', 'N/A')}")
                            self.add_log(f"证书颁发者: {cert.get('issuer', 'N/A')}")
                            
                            # 检查证书有效期
                            if 'notAfter' in cert:
                                self.add_log(f"证书有效期至: {cert['notAfter']}")
                            
                except ssl.SSLError as e:
                    self.add_log(f"✗ SSL证书错误: {str(e)}")
                    self.add_log("建议: 启用信任主机选项")
                except socket.timeout:
                    self.add_log("✗ 连接超时")
                except Exception as e:
                    self.add_log(f"✗ SSL连接失败: {str(e)}")
            
            # 2. 测试HTTP连接
            self.add_log("\n--- HTTP连接测试 ---")
            session = requests.Session()
            
            if trusted_hosts_enabled and trusted_hosts_text:
                session.verify = False
                self.add_log("已启用SSL验证跳过模式")
            
            try:
                response = session.get(proxy_url, timeout=10)
                self.add_log(f"✓ HTTP连接成功，状态码: {response.status_code}")
                
                if response.status_code == 200:
                    self.add_log("✓ 代理服务器响应正常")
                else:
                    self.add_log(f"⚠ 代理服务器返回非200状态码: {response.status_code}")
                    
            except requests.exceptions.SSLError as e:
                self.add_log(f"✗ SSL证书验证失败: {str(e)}")
                self.add_log("解决方案:")
                self.add_log("1. 启用'信任主机'选项")
                self.add_log("2. 在信任主机中输入: " + hostname)
                self.add_log("3. 如果是公司内网，考虑使用HTTP而非HTTPS")
                
            except requests.exceptions.ConnectionError as e:
                self.add_log(f"✗ 连接错误: {str(e)}")
                self.add_log("可能原因:")
                self.add_log("1. 代理地址不正确")
                self.add_log("2. 网络连接问题")
                self.add_log("3. 防火墙阻止")
                
            except requests.exceptions.Timeout:
                self.add_log("✗ 连接超时")
                self.add_log("建议: 检查网络连接或增加超时时间")
            
            # 3. 测试pip命令
            self.add_log("\n--- Pip命令测试 ---")
            try:
                import subprocess
                
                # 构建测试命令
                test_cmd = [sys.executable, "-m", "pip", "search", "requests", "--verbose"]
                
                if self.proxy_type_combo.currentIndex() == 0:  # 索引代理
                    test_cmd.extend(["-i", proxy_url])
                else:  # HTTP代理
                    test_cmd.extend(["--proxy", proxy_url])
                
                # 添加信任主机
                if trusted_hosts_enabled and trusted_hosts_text:
                    trusted_hosts = [host.strip() for host in trusted_hosts_text.replace(',', ' ').split() if host.strip()]
                    for host in trusted_hosts:
                        test_cmd.extend(["--trusted-host", host])
                
                self.add_log(f"测试命令: {' '.join(test_cmd)}")
                
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self.add_log("✓ Pip命令执行成功")
                else:
                    self.add_log(f"✗ Pip命令执行失败，返回码: {result.returncode}")
                    if result.stderr:
                        error_lines = result.stderr.strip().split('\n')[:5]  # 只显示前5行
                        for line in error_lines:
                            if line.strip():
                                self.add_log(f"  错误: {line.strip()}")
                                
            except subprocess.TimeoutExpired:
                self.add_log("✗ Pip命令执行超时")
            except Exception as e:
                self.add_log(f"✗ Pip命令测试失败: {str(e)}")
            
            # 4. 提供建议
            self.add_log("\n--- 诊断建议 ---")
            if not trusted_hosts_enabled:
                self.add_log("1. 建议启用'信任主机'选项")
                self.add_log(f"2. 在信任主机中输入: {hostname}")
            
            if parsed_url.scheme == 'https' and not trusted_hosts_enabled:
                self.add_log("3. 如果SSL证书问题持续存在，考虑使用HTTP协议")
                http_url = proxy_url.replace('https://', 'http://')
                self.add_log(f"   HTTP版本: {http_url}")
            
            self.add_log("4. 如果问题仍然存在，请联系网络管理员")
            
            # 5. 快速修复建议
            self.add_log("\n--- 快速修复建议 ---")
            if parsed_url.scheme == 'https':
                self.add_log("检测到HTTPS协议，建议执行以下操作:")
                self.add_log("1. 启用'信任主机'复选框")
                self.add_log(f"2. 在信任主机输入框中输入: {hostname}")
                self.add_log("3. 点击'应用修复'按钮自动配置")
                
                # 提供自动修复选项
                self.add_log("\n点击'应用修复'按钮可自动应用上述设置")
            
            self.add_log("=== SSL证书诊断完成 ===")
            
        except Exception as e:
            self.add_log(f"✗ SSL诊断过程中出错: {str(e)}")
            self.add_log("=== SSL证书诊断失败 ===") 
    
    def apply_ssl_fix(self):
        """应用SSL修复设置"""
        proxy_url = self.proxy_url_input.text().strip()
        if not proxy_url:
            QMessageBox.warning(self, "警告", "请先设置代理地址")
            return
        
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(proxy_url)
            hostname = parsed_url.hostname
            
            if not hostname:
                QMessageBox.warning(self, "警告", "无法解析代理地址中的主机名")
                return
            
            # 自动配置信任主机
            self.enable_trusted_host_checkbox.setChecked(True)
            self.trusted_hosts_input.setText(hostname)
            
            # 保存设置
            self.save_proxy_settings()
            
            self.add_log("=== SSL修复已应用 ===")
            self.add_log(f"已启用信任主机: {hostname}")
            self.add_log("建议: 现在可以尝试测试连接或安装依赖")
            
            QMessageBox.information(self, "成功", f"已自动配置信任主机: {hostname}\n\n现在可以尝试测试连接或安装依赖。")
            
        except Exception as e:
            self.add_log(f"✗ 应用SSL修复失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"应用SSL修复失败: {str(e)}") 
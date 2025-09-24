#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片分类模型训练工具 - 轻量版主程序
功能：轻量版启动入口，处理依赖检查和错误处理
作者：AI Assistant
日期：2025-01-19
"""

import sys
import os
import json
import traceback

# 设置编码（在无控制台的 windowed 模式下避免访问 stdout/stderr 导致崩溃）
if sys.platform == 'win32':
    try:
        import codecs
        # 仅当 stdout/stderr 存在且支持 detach 时才重绑定编码
        if getattr(sys, 'stdout', None) is not None and hasattr(sys.stdout, 'detach'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        if getattr(sys, 'stderr', None) is not None and hasattr(sys.stderr, 'detach'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except Exception:
        # 在无控制台或其他环境下静默跳过
        pass

# 将src目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def safe_print(text):
    """安全打印函数，避免编码或无控制台导致的错误"""
    try:
        # 无控制台时 sys.stdout 可能为 None
        if getattr(sys, 'stdout', None) is None:
            return
        print(text)
    except UnicodeEncodeError:
        # 如果出现编码错误，转换为ASCII
        try:
            print(text.encode('ascii', errors='ignore').decode('ascii'))
        except Exception:
            # 仍有问题则静默忽略
            pass
    except Exception:
        # 非编码相关异常（如无控制台），静默忽略
        pass

def check_critical_dependencies():
    """检查关键依赖"""
    critical_deps = ['PyQt5', 'numpy', 'PIL']
    missing = []
    
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    return missing

def show_dependency_error_dialog(missing_deps):
    """显示依赖错误对话框"""
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Dependency Error - Lightweight Version")
        msg.setText("Missing Critical Dependencies")
        msg.setInformativeText(f"Missing: {', '.join(missing_deps)}")
        msg.setDetailedText(
            "This is the lightweight version that requires manual dependency installation.\n\n"
            "Please install the missing dependencies:\n"
            f"pip install {' '.join(missing_deps)}\n\n"
            "Or use the dependency management feature in the Settings tab after installation."
        )
        msg.exec_()
        return False
        
    except ImportError:
        # 如果连PyQt5都没有，使用控制台输出
        safe_print("CRITICAL ERROR: Missing essential dependencies")
        safe_print(f"Missing: {', '.join(missing_deps)}")
        safe_print("Please install: pip install " + " ".join(missing_deps))
        return False

def main():
    """轻量版主函数"""
    try:
        safe_print("Starting Image Classification Training Tool - Lightweight Version")
        
        # 检查关键依赖
        missing_deps = check_critical_dependencies()
        if missing_deps:
            safe_print(f"Missing critical dependencies: {', '.join(missing_deps)}")
            show_dependency_error_dialog(missing_deps)
            return False
        
        # 尝试启动完整程序
        try:
            # 首先设置日志系统（使用简化版本）
            try:
                from src.utils.logger import setup_logging, get_logger
                setup_logging(
                    log_dir="logs",
                    log_level=20,  # INFO级别
                    console_output=False,  # 关闭控制台输出避免编码问题
                    file_output=True,
                    structured_logging=True
                )
                logger = get_logger(__name__, "lightweight_main")
                logger.info("Lightweight version started")
            except Exception as e:
                safe_print(f"Warning: Could not setup logging: {e}")
                logger = None
            
            # 导入并启动主程序
            safe_print("Loading main application...")
            
            # 导入PyQt5
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import QThread, QObject
            
            # 检查配置文件
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
            if not os.path.exists(config_path):
                safe_print("Creating default configuration...")
                default_config = {
                    'default_source_folder': '',
                    'default_output_folder': '',
                    'default_classes': ['defect1', 'defect2', 'defect3', 'defect4', 'defect5'],
                    'default_model_file': '',
                    'default_class_info_file': '',
                    'default_model_eval_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'saved_models'),
                    'default_model_save_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'saved_models'),
                    'default_tensorboard_log_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'tensorboard')
                }
                
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                os.makedirs(default_config['default_model_eval_dir'], exist_ok=True)
                os.makedirs(default_config['default_model_save_dir'], exist_ok=True)
                os.makedirs(default_config['default_tensorboard_log_dir'], exist_ok=True)
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, ensure_ascii=False, indent=4)
                
                safe_print(f"Default configuration created: {config_path}")
            
            # 导入配置管理器
            from src.utils.config_manager import config_manager
            config_manager.set_config_path(config_path)
            
            # 创建应用程序
            app = QApplication(sys.argv)
            app.setStyle('Fusion')
            
            # 导入并创建主窗口
            try:
                from src.ui.main_window import MainWindow
                window = MainWindow()
                safe_print("Main window created successfully")
                
                # 显示窗口
                window.show()
                safe_print("Application started successfully")
                
                # 运行应用程序
                return app.exec_()
                
            except ImportError as e:
                safe_print(f"Import error: {e}")
                safe_print("Some optional dependencies may be missing.")
                safe_print("Please use the dependency management feature to install them.")
                
                # 尝试显示一个简化的依赖管理界面
                try:
                    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
                    
                    class SimpleDependencyDialog(QWidget):
                        def __init__(self):
                            super().__init__()
                            self.setWindowTitle("Dependency Management - Lightweight Version")
                            self.setGeometry(300, 300, 500, 300)
                            
                            layout = QVBoxLayout()
                            
                            info_label = QLabel(
                                "Welcome to the Lightweight Version!\n\n"
                                "Some dependencies are missing. Please:\n"
                                "1. Install missing dependencies manually, or\n"
                                "2. Use the full version for automatic dependency management\n\n"
                                f"Error: {str(e)}"
                            )
                            layout.addWidget(info_label)
                            
                            install_btn = QPushButton("Open Dependency Installation Guide")
                            install_btn.clicked.connect(self.show_install_guide)
                            layout.addWidget(install_btn)
                            
                            close_btn = QPushButton("Close")
                            close_btn.clicked.connect(self.close)
                            layout.addWidget(close_btn)
                            
                            self.setLayout(layout)
                        
                        def show_install_guide(self):
                            QMessageBox.information(
                                self, 
                                "Installation Guide",
                                "To install missing dependencies:\n\n"
                                "1. Open command prompt/terminal\n"
                                "2. Run: pip install torch torchvision matplotlib opencv-python scikit-learn\n"
                                "3. Or use: pip install -r requirements.txt\n\n"
                                "For proxy settings, use:\n"
                                "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ <package_name>"
                            )
                    
                    dialog = SimpleDependencyDialog()
                    dialog.show()
                    return app.exec_()
                    
                except Exception as e2:
                    safe_print(f"Could not create dependency dialog: {e2}")
                    return False
            
        except Exception as e:
            safe_print(f"Error starting application: {e}")
            if logger:
                logger.error(f"Application startup error: {e}")
            traceback.print_exc()
            return False
    
    except Exception as e:
        safe_print(f"Critical error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        safe_print("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        safe_print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1) 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动检查脚本
功能：检查程序启动所需的依赖，如果缺少则引导用户到依赖管理界面
作者：AI Assistant
日期：2025-01-19
"""

import sys
import os
import importlib.util
from pathlib import Path

# 将src目录添加到Python路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_minimal_dependencies():
    """检查启动必需的最小依赖"""
    required_modules = {
        'PyQt5': 'PyQt5',
        'numpy': 'numpy',
        'PIL': 'pillow',
        'yaml': 'PyYAML',
        'colorama': 'colorama',
        'requests': 'requests',
        'psutil': 'psutil'
    }
    
    missing_modules = []
    available_modules = []
    
    for module_name, package_name in required_modules.items():
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                available_modules.append(package_name)
            else:
                missing_modules.append(package_name)
        except ImportError:
            missing_modules.append(package_name)
    
    return missing_modules, available_modules

def safe_print(text):
    """安全打印，避免编码问题"""
    try:
        print(text)
    except UnicodeEncodeError:
        # 如果出现编码错误，使用ASCII字符
        print(text.encode('ascii', errors='ignore').decode('ascii'))

def show_dependency_status():
    """显示依赖状态"""
    safe_print("=" * 60)
    safe_print("Image Classification Model Training Tool - Lightweight")
    safe_print("=" * 60)
    
    # 检查必需依赖
    missing_required, available_required = check_minimal_dependencies()
    
    safe_print("\nRequired Dependencies Check:")
    if not missing_required:
        safe_print("OK - All required dependencies are installed")
        for module in available_required:
            safe_print(f"  + {module}")
    else:
        safe_print("ERROR - Missing required dependencies:")
        for module in missing_required:
            safe_print(f"  - {module}")
        safe_print("\nWARNING: Program may not start properly without these dependencies")
    
    safe_print("\n" + "=" * 60)
    
    return missing_required

def create_dependency_guide():
    """创建依赖安装指南"""
    guide_content = """
Dependency Installation Guide
=============================

Lightweight Version First Use Instructions:

1. Start the program
2. Go to "Settings" tab
3. Select "Dependency Management" sub-tab
4. Configure proxy settings if needed
5. Click "Check Dependencies" button
6. Click "Install Missing Dependencies" button
7. Wait for installation to complete

Common Mirror Sources:
- Tsinghua: https://pypi.tuna.tsinghua.edu.cn/simple/
- Alibaba: https://mirrors.aliyun.com/pypi/simple/
- Douban: https://pypi.douban.com/simple/

If you encounter problems, please check network connection or configure proxy settings.
"""
    
    guide_path = Path(__file__).parent.parent / "dependency_guide.txt"
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    return guide_path

def main():
    """主函数"""
    try:
        # 显示依赖状态
        missing_required = show_dependency_status()
        
        # 创建依赖指南
        guide_path = create_dependency_guide()
        
        # 如果缺少必需依赖，给出提示
        if missing_required:
            safe_print("\nIMPORTANT NOTICE:")
            safe_print("The program is missing required dependencies and may not run properly.")
            safe_print("Please install dependencies using one of these methods:")
            safe_print("\n1. Manual installation:")
            safe_print("   pip install " + " ".join(missing_required))
            safe_print("\n2. Or use the built-in dependency management after starting the program")
            safe_print(f"\nDetailed instructions: {guide_path}")
            
            # 询问是否继续启动
            try:
                choice = input("\nDo you still want to try to start the program? (y/N): ").strip().lower()
                if choice not in ['y', 'yes']:
                    safe_print("Startup cancelled")
                    return False
            except KeyboardInterrupt:
                safe_print("\nStartup cancelled")
                return False
        
        # 尝试启动主程序
        safe_print("\nStarting main program...")
        try:
            from src.main import main as main_app
            main_app()
            return True
        except ImportError as e:
            safe_print(f"ERROR - Startup failed: {e}")
            safe_print("Please install the required dependencies first")
            return False
        except Exception as e:
            safe_print(f"ERROR - Startup error: {e}")
            return False
    
    except Exception as e:
        safe_print(f"Error checking dependencies: {e}")
        # 即使检查失败，也尝试启动主程序
        try:
            from src.main import main as main_app
            main_app()
            return True
        except Exception as e2:
            safe_print(f"Failed to start main program: {e2}")
            return False

if __name__ == "__main__":
    success = main()
    if not success:
        input("\nPress Enter to exit...")
        sys.exit(1) 
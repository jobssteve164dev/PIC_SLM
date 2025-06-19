#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片分类模型训练应用 - 简化版EXE打包脚本
功能：快速将应用打包成exe文件
作者：AI Assistant
日期：2025-01-19
"""

import os
import sys
import subprocess
from pathlib import Path

def install_pyinstaller():
    """安装PyInstaller"""
    print("正在安装PyInstaller...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
        print("✓ PyInstaller安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ PyInstaller安装失败: {e}")
        return False

def build_exe():
    """构建exe文件"""
    project_root = Path(__file__).parent.absolute()
    main_py = project_root / 'src' / 'main.py'
    
    if not main_py.exists():
        print(f"✗ 找不到主程序文件: {main_py}")
        return False
    
    print("正在构建exe文件...")
    print("这可能需要几分钟时间，请耐心等待...")
    
    # PyInstaller命令
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',  # 打包成单个exe文件
        '--windowed',  # 窗口应用，不显示控制台
        '--name=图片分类模型训练工具',
        '--add-data=config;config',
        '--add-data=models;models', 
        '--add-data=setting;setting',
        '--add-data=readme;readme',
        '--add-data=config.json;.',
        '--hidden-import=PyQt5.QtCore',
        '--hidden-import=PyQt5.QtGui',
        '--hidden-import=PyQt5.QtWidgets',
        '--hidden-import=torch',
        '--hidden-import=torchvision',
        '--hidden-import=numpy',
        '--hidden-import=PIL',
        '--hidden-import=matplotlib',
        '--hidden-import=sklearn',
        '--hidden-import=cv2',
        '--hidden-import=pandas',
        '--hidden-import=lime',
        '--hidden-import=unittest',
        '--hidden-import=unittest.mock',
        '--clean',
        '--noconfirm',
        str(main_py)
    ]
    
    try:
        # 设置环境变量解决编码问题
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd, 
            cwd=str(project_root), 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='ignore',
            env=env
        )
        
        if result.returncode == 0:
            print("✓ exe文件构建成功！")
            
            # 检查生成的exe文件
            exe_path = project_root / 'dist' / '图片分类模型训练工具.exe'
            if exe_path.exists():
                file_size = exe_path.stat().st_size / (1024 * 1024)  # MB
                print(f"exe文件位置: {exe_path}")
                print(f"文件大小: {file_size:.1f} MB")
                
                # 创建启动脚本
                create_launcher(project_root / 'dist')
                
                return True
            else:
                print("✗ exe文件未找到")
                return False
        else:
            print("✗ exe文件构建失败")
            print("错误输出:")
            if result.stderr:
                try:
                    print(result.stderr)
                except UnicodeDecodeError:
                    print("错误信息包含特殊字符，无法显示")
            return False
            
    except Exception as e:
        print(f"✗ 构建过程中出现错误: {e}")
        return False

def create_launcher(dist_dir):
    """创建启动脚本"""
    bat_content = '''@echo off
chcp 65001 > nul
title 图片分类模型训练工具
echo 正在启动图片分类模型训练工具...
echo.

REM 设置工作目录
cd /d "%~dp0"

REM 启动应用程序
"图片分类模型训练工具.exe"

REM 如果程序异常退出，显示错误信息
if errorlevel 1 (
    echo.
    echo 程序异常退出，请检查系统环境
    pause
)
'''
    
    bat_file = dist_dir / "启动应用.bat"
    with open(bat_file, 'w', encoding='gbk') as f:
        f.write(bat_content)
    
    print(f"✓ 启动脚本已创建: {bat_file}")

def main():
    print("=" * 60)
    print("图片分类模型训练应用 - 简化版EXE打包工具")
    print("=" * 60)
    
    # 检查PyInstaller
    try:
        import PyInstaller
        print("✓ PyInstaller已安装")
    except ImportError:
        print("PyInstaller未安装，正在安装...")
        if not install_pyinstaller():
            print("安装失败，请手动安装: pip install pyinstaller")
            input("按回车键退出...")
            return
    
    # 构建exe
    if build_exe():
        print("\n🎉 构建完成！")
        print("您可以在dist文件夹中找到生成的exe文件")
        print("建议使用'启动应用.bat'来启动程序")
    else:
        print("\n构建失败，请检查错误信息")
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main() 
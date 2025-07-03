#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片分类模型训练应用 - 简化版EXE打包脚本
功能：快速将应用打包成exe文件，只包含启动必需的依赖，其他依赖让用户自行下载
作者：AI Assistant
日期：2025-01-19
更新：2025-01-19 - 适配依赖管理功能
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

def create_minimal_requirements():
    """创建最小依赖文件，只包含启动必需的依赖"""
    project_root = Path(__file__).parent.absolute()
    minimal_req_path = project_root / 'requirements_minimal.txt'
    
    # 定义启动必需的最小依赖集合
    minimal_dependencies = [
        "# 启动必需的最小依赖集合",
        "# 这些依赖是程序启动和基本界面显示所必需的",
        "",
        "# GUI框架 - 必需",
        "PyQt5>=5.15.0",
        "",
        "# 基础数据处理 - 必需",
        "numpy>=1.19.0",
        "pillow>=8.0.0",
        "",
        "# 配置和日志 - 必需", 
        "PyYAML>=5.4.0",
        "colorama>=0.4.0",
        "",
        "# 网络请求 - 依赖管理功能必需",
        "requests>=2.25.0",
        "",
        "# 系统监控 - 资源限制功能必需",
        "psutil>=5.8.0",
        "",
        "# 其他依赖将通过程序内置的依赖管理功能自动下载",
        "# 包括：torch, torchvision, matplotlib, opencv-python, sklearn等"
    ]
    
    with open(minimal_req_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(minimal_dependencies))
    
    print(f"✓ 创建最小依赖文件: {minimal_req_path}")
    return minimal_req_path

def build_exe():
    """构建exe文件"""
    project_root = Path(__file__).parent.absolute()
    main_py = project_root / 'src' / 'main.py'
    
    if not main_py.exists():
        print(f"✗ 找不到主程序文件: {main_py}")
        return False
    
    print("正在构建轻量级exe文件...")
    print("此版本只包含启动必需的依赖，其他依赖将通过程序内置功能下载")
    print("这可能需要几分钟时间，请耐心等待...")
    
    # 创建最小依赖文件
    minimal_req_path = create_minimal_requirements()
    
    # 使用轻量版主程序作为入口点
    lightweight_main = project_root / 'src' / 'main_lightweight.py'
    
    # PyInstaller命令 - 简化版只包含启动必需的依赖
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',  # 打包成单个exe文件
        '--windowed',  # 窗口应用，不显示控制台
        '--name=图片分类模型训练工具_轻量版',
        # 只添加必要的数据文件
        '--add-data=config;config',
        '--add-data=setting;setting',
        '--add-data=config.json;.',
        '--add-data=requirements_minimal.txt;.',
        '--add-data=src;src',  # 添加整个src目录
        # 只包含启动必需的隐藏导入
        '--hidden-import=PyQt5.QtCore',
        '--hidden-import=PyQt5.QtGui',
        '--hidden-import=PyQt5.QtWidgets',
        '--hidden-import=PyQt5.sip',
        '--hidden-import=numpy',
        '--hidden-import=PIL',
        '--hidden-import=PIL.Image',
        '--hidden-import=yaml',
        '--hidden-import=colorama',
        '--hidden-import=requests',
        '--hidden-import=psutil',
        '--hidden-import=json',
        '--hidden-import=subprocess',
        '--hidden-import=threading',
        '--hidden-import=importlib',
        '--hidden-import=importlib.util',
        '--hidden-import=pkg_resources',
        # 排除不必要的模块以减小体积
        '--exclude-module=torch',
        '--exclude-module=torchvision',
        '--exclude-module=matplotlib',
        '--exclude-module=cv2',
        '--exclude-module=sklearn',
        '--exclude-module=pandas',
        '--exclude-module=seaborn',
        '--exclude-module=lime',
        '--exclude-module=shap',
        '--exclude-module=tensorboard',
        '--exclude-module=albumentations',
        '--exclude-module=timm',
        '--exclude-module=efficientnet_pytorch',
        '--exclude-module=torchmetrics',
        '--exclude-module=torchsummary',
        '--exclude-module=scikit-image',
        '--exclude-module=mplcursors',
        '--exclude-module=labelImg',
        '--exclude-module=labelme',
        '--exclude-module=ultralytics',
        '--clean',
        '--noconfirm',
        str(lightweight_main)  # 使用轻量版主程序作为入口点
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
            print("✓ 轻量级exe文件构建成功！")
            
            # 检查生成的exe文件
            exe_path = project_root / 'dist' / '图片分类模型训练工具_轻量版.exe'
            if exe_path.exists():
                file_size = exe_path.stat().st_size / (1024 * 1024)  # MB
                print(f"exe文件位置: {exe_path}")
                print(f"文件大小: {file_size:.1f} MB")
                
                # 创建启动脚本和说明文档
                create_launcher_and_docs(project_root / 'dist')
                
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

def create_launcher_and_docs(dist_dir):
    """创建启动脚本和说明文档"""
    # 创建启动脚本
    bat_content = '''@echo off
chcp 65001 > nul
title 图片分类模型训练工具（轻量版）

REM 设置工作目录
cd /d "%~dp0"

REM 启动应用程序
"图片分类模型训练工具_轻量版.exe"

REM 如果程序异常退出，显示错误信息
if errorlevel 1 (
    echo.
    echo 程序异常退出，可能的原因：
    echo 1. 缺少必要的依赖库（请手动安装或使用依赖管理功能）
    echo 2. 系统环境问题（请检查Windows版本兼容性）
    echo.
    pause
)
'''
    
    bat_file = dist_dir / "启动轻量版应用.bat"
    with open(bat_file, 'w', encoding='gbk') as f:
        f.write(bat_content)
    
    # 创建使用说明文档
    readme_content = '''图片分类模型训练工具 - 轻量版使用说明
==============================================

## 版本特点

本轻量版具有以下特点：
- 文件体积小，启动速度快
- 只包含启动必需的依赖库
- 其他依赖通过程序内置功能自动下载
- 支持代理设置，适合企业内网环境

## 首次使用步骤

1. 启动程序
   - 双击"启动轻量版应用.bat"
   - 或直接双击exe文件

2. 配置依赖管理
   - 进入"设置"标签页
   - 选择"依赖管理"子标签
   - 根据网络环境配置代理设置（可选）

3. 安装依赖库
   - 点击"检查依赖"按钮
   - 等待依赖检查完成
   - 点击"安装缺失依赖"按钮
   - 等待安装完成

4. 开始使用
   - 依赖安装完成后即可正常使用所有功能

## 网络配置

### 直连网络
如果您的网络可以直接访问PyPI，无需特殊配置。

### 代理网络
如果您在企业内网环境：
1. 在"依赖管理"中启用代理设置
2. 选择合适的代理类型：
   - 索引代理：使用国内镜像源（推荐）
   - HTTP代理：使用公司代理服务器
3. 配置代理地址和信任主机

### 常用镜像源
- 清华源：https://pypi.tuna.tsinghua.edu.cn/simple/
- 阿里源：https://mirrors.aliyun.com/pypi/simple/
- 豆瓣源：https://pypi.douban.com/simple/
- 华为源：https://mirrors.huaweicloud.com/repository/pypi/simple/

## 故障排除

### 程序无法启动
1. 检查Windows版本兼容性
2. 确保已安装Visual C++运行库
3. 检查杀毒软件是否误报

### 依赖安装失败
1. 检查网络连接
2. 尝试配置代理设置
3. 手动安装核心依赖：
   ```
   pip install torch torchvision matplotlib opencv-python scikit-learn
   ```

### 功能异常
1. 确保所有依赖都已正确安装
2. 查看程序日志文件（logs目录）
3. 重新启动程序

## 技术支持

如遇到问题，请：
1. 查看程序日志文件
2. 检查依赖安装状态
3. 确认网络配置正确

## 版本信息

- 轻量版：只包含启动必需依赖
- 完整版：包含所有依赖库
- 推荐使用轻量版，按需下载依赖

最后更新：2025-01-19
'''
    
    readme_file = dist_dir / "使用说明.txt"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✓ 启动脚本已创建: {bat_file}")
    print(f"✓ 使用说明已创建: {readme_file}")

def main():
    print("=" * 70)
    print("图片分类模型训练应用 - 轻量版EXE打包工具")
    print("=" * 70)
    print("特点：")
    print("• 只包含启动必需的依赖，文件体积小")
    print("• 其他依赖通过程序内置功能自动下载")
    print("• 支持代理设置，适合企业内网环境")
    print("• 首次使用需要联网下载依赖库")
    print("=" * 70)
    
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
        print("\n🎉 轻量版构建完成！")
        print("=" * 50)
        print("重要提示：")
        print("1. 生成的exe文件体积较小，但首次运行需要下载依赖")
        print("2. 请确保目标电脑有网络连接")
        print("3. 建议首次运行时配置依赖管理设置")
        print("4. 使用'启动轻量版应用.bat'来启动程序")
        print("5. 详细说明请查看'使用说明.txt'")
        print("=" * 50)
    else:
        print("\n构建失败，请检查错误信息")
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main() 
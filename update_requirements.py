#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
依赖更新脚本
功能：自动更新requirements.txt文件，确保依赖版本与当前环境一致
作者：AI Assistant
日期：2025-01-19
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def update_requirements():
    """更新requirements.txt文件"""
    print("=" * 60)
    print("正在更新requirements.txt文件...")
    
    project_root = Path(__file__).parent.absolute()
    
    try:
        # 获取当前环境的包列表
        result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], 
                              capture_output=True, text=True, check=True)
        current_packages = {}
        for line in result.stdout.strip().split('\n'):
            if '==' in line:
                name, version = line.split('==', 1)
                current_packages[name.lower()] = f"{name}=={version}"
        
        # 定义项目实际需要的核心依赖
        core_dependencies = {
            'pyqt5': 'PyQt5',
            'numpy': 'numpy', 
            'pillow': 'Pillow',
            'matplotlib': 'matplotlib',
            'torch': 'torch',
            'torchvision': 'torchvision',
            'timm': 'timm',
            'opencv-python': 'opencv-python',
            'albumentations': 'albumentations',
            'pandas': 'pandas',
            'tensorboard': 'tensorboard',
            'scikit-learn': 'scikit-learn',
            'seaborn': 'seaborn',
            'tqdm': 'tqdm',
            'pyyaml': 'PyYAML',
            'psutil': 'psutil',
            'colorama': 'colorama',
            'efficientnet-pytorch': 'efficientnet-pytorch',
            'torchmetrics': 'torchmetrics',
            'torchsummary': 'torchsummary',
            'lime': 'lime',
            'scikit-image': 'scikit-image',
            'mplcursors': 'mplcursors',
            'requests': 'requests',
        }
        
        # 可选依赖（仅在存在时添加）
        optional_dependencies = {
            'pywin32': 'pywin32',
            'nvidia-ml-py3': 'nvidia-ml-py3',
            'imgaug': 'imgaug',
            'labelimg': 'labelImg',
            'labelme': 'labelme',
            'pytorch-lightning': 'pytorch-lightning',
            'ultralytics': 'ultralytics',
        }
        
        # 生成新的requirements.txt内容
        new_requirements = []
        new_requirements.append("# 自动生成的依赖文件")
        new_requirements.append(f"# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        new_requirements.append("")
        
        new_requirements.append("# 核心依赖")
        found_packages = []
        missing_packages = []
        
        for key, package_name in core_dependencies.items():
            if key in current_packages:
                new_requirements.append(current_packages[key])
                found_packages.append(package_name)
            else:
                missing_packages.append(package_name)
        
        # 添加可选依赖
        new_requirements.append("")
        new_requirements.append("# 可选依赖")
        optional_found = []
        
        for key, package_name in optional_dependencies.items():
            if key in current_packages:
                new_requirements.append(current_packages[key])
                optional_found.append(package_name)
        
        # 添加说明
        new_requirements.append("")
        new_requirements.append("# 平台特定依赖")
        new_requirements.append("# pywin32>=228; platform_system==\"Windows\"  # Windows系统支持")
        new_requirements.append("# nvidia-ml-py3>=7.352.0  # NVIDIA GPU监控")
        
        new_requirements.append("")
        new_requirements.append("# 注意事项：")
        new_requirements.append("# 1. 本文件由脚本自动生成，请勿手动编辑")
        new_requirements.append("# 2. 如需添加新依赖，请修改update_requirements.py脚本")
        new_requirements.append("# 3. 建议在虚拟环境中安装这些依赖")
        
        # 写入文件
        requirements_path = project_root / "requirements.txt"
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_requirements))
        
        print(f"✓ requirements.txt 已更新: {requirements_path}")
        print(f"✓ 找到核心依赖: {len(found_packages)} 个")
        print(f"✓ 找到可选依赖: {len(optional_found)} 个")
        
        if missing_packages:
            print(f"⚠ 缺失核心依赖: {', '.join(missing_packages)}")
            print("建议运行: pip install " + " ".join(missing_packages))
        
        return True
        
    except Exception as e:
        print(f"✗ 更新requirements.txt失败: {e}")
        return False

def check_environment():
    """检查当前环境"""
    print("=" * 60)
    print("检查当前Python环境...")
    
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 检查是否在虚拟环境中
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ 运行在虚拟环境中")
    else:
        print("⚠ 运行在系统Python环境中，建议使用虚拟环境")
    
    # 统计已安装包数量
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, check=True)
        package_count = len(result.stdout.strip().split('\n')) - 2  # 减去标题行
        print(f"已安装包数量: {package_count}")
    except:
        print("无法获取包列表")

def main():
    """主函数"""
    print("依赖更新脚本")
    print("=" * 60)
    
    # 检查环境
    check_environment()
    
    # 更新requirements.txt
    success = update_requirements()
    
    if success:
        print("\n✅ 依赖文件更新成功！")
        print("现在可以运行打包脚本了。")
    else:
        print("\n❌ 依赖文件更新失败！")
        print("请检查错误信息并重试。")
    
    # input("\n按回车键退出...")  # 自动化运行时注释掉

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户取消操作")
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc() 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片分类模型训练应用 - EXE打包脚本
功能：将应用打包成可独立运行的exe文件
作者：AI Assistant
日期：2025-01-19
"""

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path
import time

class ExeBuilder:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.spec_file = self.project_root / "app.spec"
        self.exe_name = "图片分类模型训练工具"
        
    def update_requirements(self):
        """更新requirements.txt文件"""
        print("=" * 60)
        print("正在更新requirements.txt文件...")
        
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
            
            # 生成新的requirements.txt内容
            new_requirements = []
            new_requirements.append("# 自动生成的依赖文件")
            new_requirements.append(f"# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            new_requirements.append("")
            new_requirements.append("# 核心GUI框架")
            
            for key, package_name in core_dependencies.items():
                if key in current_packages:
                    new_requirements.append(current_packages[key])
                else:
                    print(f"警告: 未找到包 {package_name}")
            
            # 添加可选依赖说明
            new_requirements.append("")
            new_requirements.append("# 可选依赖（根据需要安装）")
            new_requirements.append("# pywin32>=228; platform_system==\"Windows\"  # Windows系统支持")
            new_requirements.append("# nvidia-ml-py3>=7.352.0  # NVIDIA GPU监控")
            
            # 写入文件
            requirements_path = self.project_root / "requirements.txt"
            with open(requirements_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_requirements))
            
            print(f"✓ requirements.txt 已更新: {requirements_path}")
            return True
            
        except Exception as e:
            print(f"✗ 更新requirements.txt失败: {e}")
            return False
    
    def get_dynamic_hiddenimports(self):
        """动态获取隐藏导入列表"""
        # 基础第三方库
        base_imports = [
            'PyQt5.QtCore',
            'PyQt5.QtGui', 
            'PyQt5.QtWidgets',
            'PyQt5.sip',
            'torch',
            'torch.nn',
            'torch.optim',
            'torch.utils.data',
            'torchvision',
            'torchvision.transforms',
            'torchvision.models',
            'numpy',
            'PIL',
            'PIL.Image',
            'matplotlib',
            'matplotlib.pyplot',
            'matplotlib.backends.backend_qt5agg',
            'matplotlib.figure',
            'sklearn',
            'sklearn.metrics',
            'sklearn.preprocessing',
            'cv2',
            'albumentations',
            'timm',
            'timm.models',
            'tensorboard',
            'lime',
            'lime.lime_image',
            'pandas',
            'seaborn',
            'tqdm',
            'yaml',
            'psutil',
            'efficientnet_pytorch',
            'torchmetrics',
            'torchsummary',
            'colorama',
            'requests',
            'scikit-image',
            'mplcursors',
        ]
        
        # 扫描src目录获取项目内部模块
        project_imports = []
        src_dir = self.project_root / "src"
        
        def scan_directory(directory, prefix="src"):
            for item in directory.iterdir():
                if item.is_file() and item.suffix == '.py' and item.name != '__init__.py':
                    module_name = f"{prefix}.{item.stem}"
                    project_imports.append(module_name)
                elif item.is_dir() and not item.name.startswith('.') and not item.name.startswith('__pycache__'):
                    scan_directory(item, f"{prefix}.{item.name}")
        
        if src_dir.exists():
            scan_directory(src_dir)
        
        return base_imports + project_imports
    
    def check_dependencies(self):
        """检查打包所需的依赖"""
        print("=" * 60)
        print("正在检查打包依赖...")
        
        required_packages = ['pyinstaller', 'pyinstaller-hooks-contrib']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✓ {package} 已安装")
            except ImportError:
                missing_packages.append(package)
                print(f"✗ {package} 未安装")
        
        if missing_packages:
            print(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
            print("正在自动安装...")
            for package in missing_packages:
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    print(f"✓ {package} 安装成功")
                except subprocess.CalledProcessError as e:
                    print(f"✗ {package} 安装失败: {e}")
                    return False
        
        print("依赖检查完成！")
        return True
    
    def create_spec_file(self):
        """创建PyInstaller配置文件"""
        print("=" * 60)
        print("正在创建PyInstaller配置文件...")
        
        # 收集数据文件
        datas = [
            ('config', 'config'),
            ('models', 'models'),
            ('setting', 'setting'),
            ('readme', 'readme'),
            ('config.json', '.'),
            ('src', 'src'),  # 添加整个src目录
        ]
        
        # 动态获取隐藏导入
        hiddenimports = self.get_dynamic_hiddenimports()
        
        # 排除的模块（减少打包大小）
        excludes = [
            'tkinter',
            'unittest',
            'test',
            'distutils',
            'setuptools',
            'pip',
            'wheel',
            'jupyter',
            'notebook',
            'IPython',
        ]
        
        spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

block_cipher = None

# 项目根目录
project_root = Path(r"{self.project_root}")

# 数据文件
datas = {datas}

# 隐藏导入
hiddenimports = {hiddenimports}

# 排除模块
excludes = {excludes}

# 分析
a = Analysis(
    [str(project_root / 'src' / 'main.py')],
    pathex=[str(project_root), str(project_root / 'src')],  # 添加src目录到路径
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 过滤不需要的文件
def filter_files(files):
    filtered = []
    exclude_patterns = [
        '__pycache__',
        '.pyc',
        '.pyo',
        '.git',
        '.gitignore',
        'test_',
        '_test',
        'tests/',
        'test/',
        '.pytest_cache',
        'node_modules',
        '.vscode',
        '.idea',
    ]
    
    for file_tuple in files:
        file_path = file_tuple[0]
        should_exclude = any(pattern in file_path.lower() for pattern in exclude_patterns)
        if not should_exclude:
            filtered.append(file_tuple)
    
    return filtered

a.datas = filter_files(a.datas)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{self.exe_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 窗口应用，不显示控制台
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可以添加图标文件路径
    version=None,  # 可以添加版本信息文件
)
'''
        
        with open(self.spec_file, 'w', encoding='utf-8') as f:
            f.write(spec_content)
        
        print(f"✓ 配置文件已创建: {self.spec_file}")
    
    def clean_build_dirs(self):
        """清理构建目录"""
        print("=" * 60)
        print("正在清理构建目录...")
        
        dirs_to_clean = [self.build_dir, self.dist_dir]
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"✓ 已清理: {dir_path}")
        
        if self.spec_file.exists():
            os.remove(self.spec_file)
            print(f"✓ 已清理: {self.spec_file}")
    
    def build_exe(self):
        """构建exe文件"""
        print("=" * 60)
        print("正在构建exe文件...")
        print("这可能需要几分钟时间，请耐心等待...")
        
        start_time = time.time()
        
        try:
            # 运行PyInstaller
            cmd = [
                sys.executable, '-m', 'PyInstaller',
                '--clean',
                '--noconfirm',
                str(self.spec_file)
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            
            # 设置环境变量解决编码问题
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding='utf-8',
                errors='ignore',
                cwd=str(self.project_root),
                env=env
            )
            
            # 实时显示输出
            for line in process.stdout:
                print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                elapsed_time = time.time() - start_time
                print(f"✓ exe文件构建成功！耗时: {elapsed_time:.1f}秒")
                return True
            else:
                print(f"✗ exe文件构建失败！返回码: {process.returncode}")
                return False
                
        except Exception as e:
            print(f"✗ 构建过程中出现错误: {e}")
            return False
    
    def create_launcher_script(self):
        """创建启动脚本"""
        print("=" * 60)
        print("正在创建启动脚本...")
        
        exe_path = self.dist_dir / f"{self.exe_name}.exe"
        if not exe_path.exists():
            print("✗ exe文件不存在，无法创建启动脚本")
            return False
        
        # 创建批处理启动脚本
        bat_content = f'''@echo off
chcp 65001 > nul
title {self.exe_name}
echo 正在启动{self.exe_name}...
echo.

REM 设置工作目录为exe文件所在目录
cd /d "%~dp0"

REM 启动应用程序
"{self.exe_name}.exe"

REM 如果程序异常退出，暂停显示错误信息
if errorlevel 1 (
    echo.
    echo 程序异常退出，错误代码: %errorlevel%
    echo 请检查是否缺少必要的文件或权限不足
    pause
)
'''
        
        bat_file = self.dist_dir / "启动应用.bat"
        with open(bat_file, 'w', encoding='gbk') as f:
            f.write(bat_content)
        
        print(f"✓ 启动脚本已创建: {bat_file}")
        
        # 创建使用说明
        readme_content = f'''# {self.exe_name} - 使用说明

## 文件说明
- `{self.exe_name}.exe`: 主程序文件
- `启动应用.bat`: 启动脚本（推荐使用）
- 各种文件夹: 程序运行所需的资源文件

## 使用方法
1. 双击 `启动应用.bat` 启动程序（推荐）
2. 或直接双击 `{self.exe_name}.exe` 启动

## 系统要求
- Windows 10 或更高版本
- 至少 4GB 内存
- 建议有独立显卡用于深度学习训练

## 注意事项
1. 首次运行可能需要较长时间加载
2. 训练模型时建议关闭其他大型应用程序
3. 确保有足够的磁盘空间存储训练数据和模型

## 常见问题
1. 如果程序无法启动，请检查是否安装了必要的运行库
2. 如果遇到权限问题，请以管理员身份运行
3. 如果程序运行缓慢，请检查系统资源使用情况

## 技术支持
如有问题，请联系开发团队或查看项目文档。

构建日期: {time.strftime('%Y-%m-%d %H:%M:%S')}
'''
        
        readme_file = self.dist_dir / "使用说明.txt"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✓ 使用说明已创建: {readme_file}")
        return True
    
    def optimize_exe(self):
        """优化exe文件"""
        print("=" * 60)
        print("正在优化exe文件...")
        
        exe_path = self.dist_dir / f"{self.exe_name}.exe"
        if not exe_path.exists():
            print("✗ exe文件不存在，无法优化")
            return False
        
        # 获取文件大小
        file_size = exe_path.stat().st_size / (1024 * 1024)  # MB
        print(f"exe文件大小: {file_size:.1f} MB")
        
        # 如果有UPX工具，可以进一步压缩
        try:
            subprocess.run(['upx', '--version'], capture_output=True, check=True)
            print("检测到UPX工具，正在进行额外压缩...")
            subprocess.run(['upx', '--best', str(exe_path)], check=True)
            
            new_size = exe_path.stat().st_size / (1024 * 1024)
            print(f"压缩后大小: {new_size:.1f} MB (节省 {file_size - new_size:.1f} MB)")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("未检测到UPX工具，跳过额外压缩")
        
        return True
    
    def create_installer(self):
        """创建安装包（可选）"""
        print("=" * 60)
        print("正在检查是否可以创建安装包...")
        
        # 检查是否有NSIS或Inno Setup
        nsis_available = False
        inno_available = False
        
        try:
            subprocess.run(['makensis', '/VERSION'], capture_output=True, check=True)
            nsis_available = True
            print("✓ 检测到NSIS")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        try:
            subprocess.run(['iscc', '/?'], capture_output=True, check=True)
            inno_available = True
            print("✓ 检测到Inno Setup")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        if not (nsis_available or inno_available):
            print("未检测到安装包制作工具（NSIS或Inno Setup）")
            print("如需创建安装包，请安装相应工具")
            return False
        
        # 这里可以添加创建安装包的逻辑
        print("安装包制作功能待实现")
        return True
    
    def run(self):
        """运行完整的打包流程"""
        print("=" * 80)
        print(f"开始构建 {self.exe_name}")
        print("=" * 80)
        
        try:
            # 更新依赖文件
            self.update_requirements()
            
            # 检查依赖
            if not self.check_dependencies():
                print("✗ 依赖检查失败，终止构建")
                return False
            
            # 清理构建目录
            self.clean_build_dirs()
            
            # 创建配置文件
            self.create_spec_file()
            
            # 构建exe
            if not self.build_exe():
                print("✗ exe构建失败，终止流程")
                return False
            
            # 创建启动脚本和说明文档
            self.create_launcher_script()
            
            # 优化exe
            self.optimize_exe()
            
            # 尝试创建安装包
            self.create_installer()
            
            print("=" * 80)
            print("🎉 构建完成！")
            print(f"exe文件位置: {self.dist_dir / f'{self.exe_name}.exe'}")
            print(f"完整包位置: {self.dist_dir}")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"✗ 构建过程中出现未预期的错误: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # 清理临时文件
            if self.spec_file.exists():
                os.remove(self.spec_file)
                print(f"✓ 已清理临时文件: {self.spec_file}")

def main():
    """主函数"""
    print("图片分类模型训练应用 - EXE打包工具")
    print("=" * 80)
    
    builder = ExeBuilder()
    success = builder.run()
    
    if success:
        print("\n构建成功！您可以将dist文件夹中的内容复制到其他电脑上运行。")
        input("\n按回车键退出...")
    else:
        print("\n构建失败！请检查上述错误信息。")
        input("\n按回车键退出...")

if __name__ == "__main__":
    main() 
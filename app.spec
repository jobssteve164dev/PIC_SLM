# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

block_cipher = None

# 项目根目录
project_root = Path(r"f:\Qsync\00.AI_PROJECT\图片分类模型训练\C1")

# 数据文件
datas = [('config', 'config'), ('models', 'models'), ('setting', 'setting'), ('readme', 'readme'), ('config.json', '.')]

# 隐藏导入
hiddenimports = ['PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets', 'torch', 'torchvision', 'numpy', 'PIL', 'matplotlib', 'sklearn', 'cv2', 'albumentations', 'timm', 'tensorboard', 'lime', 'pandas', 'seaborn', 'tqdm', 'yaml', 'psutil', 'efficientnet_pytorch', 'torchmetrics', 'torchsummary', 'pytorch_lightning', 'imgaug', 'labelImg', 'labelme', 'colorama', 'requests', 'scikit-image', 'ultralytics', 'pywin32', 'nvidia-ml-py3']

# 排除模块
excludes = ['tkinter', 'unittest', 'test', 'distutils', 'setuptools', 'pip']

# 分析
a = Analysis(
    [str(project_root / 'src' / 'main.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
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
    name='图片分类模型训练工具',
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

#!/bin/bash

echo "正在创建虚拟环境..."
python3 -m venv venv

echo "激活虚拟环境..."
source venv/bin/activate

echo "升级pip..."
python -m pip install --upgrade pip

echo "安装依赖项..."
pip install -r requirements.txt

echo "创建必要的目录结构..."
mkdir -p models/saved_models
mkdir -p runs/tensorboard

echo "设置执行权限..."
chmod +x src/main.py

echo "安装完成！"
echo "请使用 'python src/main.py' 启动程序" 
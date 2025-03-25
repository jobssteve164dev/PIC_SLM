@echo off
echo 正在创建虚拟环境...
python -m venv venv

echo 激活虚拟环境...
call venv\Scripts\activate.bat

echo 升级pip...
python -m pip install --upgrade pip

echo 安装依赖项...
pip install -r requirements.txt

echo 创建必要的目录结构...
mkdir models 2>nul
mkdir models\saved_models 2>nul
mkdir runs 2>nul
mkdir runs\tensorboard 2>nul

echo 安装完成！
echo 请使用 'python src/main.py' 启动程序
pause 
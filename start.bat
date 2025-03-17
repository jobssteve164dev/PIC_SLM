@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: 设置标题
title 图片模型训练系统启动器

:: 设置颜色
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "MAGENTA=[95m"
set "CYAN=[96m"
set "WHITE=[97m"
set "RESET=[0m"

:: 检查是否以管理员权限运行
net session >nul 2>&1
if %errorLevel% == 0 (
    set "ADMIN=1"
) else (
    set "ADMIN=0"
)

:: 设置路径
set "SCRIPT_DIR=%~dp0"
set "DESKTOP=%USERPROFILE%\Desktop"
set "PYTHON_VERSION=3.8"
set "VENV_NAME=venv"
set "VENV_PATH=%SCRIPT_DIR%%VENV_NAME%"
set "MODELS_DIR=%SCRIPT_DIR%models"
set "OFFLINE_MODELS_DIR=%MODELS_DIR%\offline"

:: 创建菜单
:menu
cls
echo %CYAN%===========================================
echo            图片模型训练系统启动器
echo ===========================================%RESET%
echo.
echo %YELLOW%1. 检查并安装Python环境
echo 2. 检查并安装项目依赖
echo 3. 创建桌面快捷方式
echo 4. 下载离线模型
echo 5. 启动程序
echo 6. 退出%RESET%
echo.
set /p "choice=请选择操作 (1-6): "

:: 处理选择
if "%choice%"=="1" goto check_python
if "%choice%"=="2" goto check_dependencies
if "%choice%"=="3" goto create_shortcut
if "%choice%"=="4" goto download_models
if "%choice%"=="5" goto start_program
if "%choice%"=="6" goto end
goto menu

:: 检查Python环境
:check_python
cls
echo %CYAN%正在检查Python环境...%RESET%
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo %RED%未检测到Python，请先安装Python %PYTHON_VERSION% 或更高版本%RESET%
    echo 您可以从 https://www.python.org/downloads/ 下载Python
    pause
    goto menu
)

:: 检查Python版本
python -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" >nul 2>&1
if %errorLevel% neq 0 (
    echo %RED%Python版本过低，请安装Python %PYTHON_VERSION% 或更高版本%RESET%
    pause
    goto menu
)

:: 检查虚拟环境
if not exist "%VENV_PATH%" (
    echo %YELLOW%正在创建虚拟环境...%RESET%
    python -m venv "%VENV_PATH%"
    if %errorLevel% neq 0 (
        echo %RED%创建虚拟环境失败%RESET%
        pause
        goto menu
    )
)

echo %GREEN%Python环境检查完成%RESET%
pause
goto menu

:: 检查并安装依赖
:check_dependencies
cls
echo %CYAN%正在检查项目依赖...%RESET%

:: 激活虚拟环境
call "%VENV_PATH%\Scripts\activate.bat"

:: 检查pip
python -m pip --version >nul 2>&1
if %errorLevel% neq 0 (
    echo %RED%未检测到pip，正在安装...%RESET%
    python -m ensurepip --default-pip
)

:: 升级pip
python -m pip install --upgrade pip

:: 安装依赖
echo %YELLOW%正在安装项目依赖...%RESET%
pip install -r requirements.txt
if %errorLevel% neq 0 (
    echo %RED%安装依赖失败%RESET%
    pause
    goto menu
)

echo %GREEN%依赖安装完成%RESET%
pause
goto menu

:: 创建桌面快捷方式
:create_shortcut
cls
echo %CYAN%正在创建桌面快捷方式...%RESET%

:: 检查管理员权限
if "%ADMIN%"=="0" (
    echo %RED%需要管理员权限来创建快捷方式%RESET%
    echo 请右键点击此脚本，选择"以管理员身份运行"
    pause
    goto menu
)

:: 创建快捷方式
set "SHORTCUT_NAME=图片模型训练系统.lnk"
set "TARGET_PATH=%VENV_PATH%\Scripts\pythonw.exe"
set "ARGUMENTS=%SCRIPT_DIR%src\main.py"
set "WORKING_DIR=%SCRIPT_DIR%"
set "ICON_PATH=%SCRIPT_DIR%src\ui\icons\app.ico"

:: 使用PowerShell创建快捷方式
powershell -Command "$WS = New-Object -ComObject WScript.Shell; $SC = $WS.CreateShortcut('%DESKTOP%\%SHORTCUT_NAME%'); $SC.TargetPath = '%TARGET_PATH%'; $SC.Arguments = '%ARGUMENTS%'; $SC.WorkingDirectory = '%WORKING_DIR%'; if (Test-Path '%ICON_PATH%') { $SC.IconLocation = '%ICON_PATH%' }; $SC.Save()"

echo %GREEN%快捷方式创建完成%RESET%
pause
goto menu

:: 下载离线模型
:download_models
cls
echo %CYAN%正在准备下载离线模型...%RESET%

:: 创建模型目录
if not exist "%OFFLINE_MODELS_DIR%" mkdir "%OFFLINE_MODELS_DIR%"

:: 下载预训练模型
echo %YELLOW%正在下载预训练模型...%RESET%
python -c "import torch; torch.hub.download_url_to_file('https://download.pytorch.org/models/resnet50-19c8e357.pth', '%OFFLINE_MODELS_DIR%\resnet50.pth')"
python -c "import torch; torch.hub.download_url_to_file('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth', '%OFFLINE_MODELS_DIR%\mobilenet_v2.pth')"
python -c "import torch; torch.hub.download_url_to_file('https://download.pytorch.org/models/densenet121-a639ec97.pth', '%OFFLINE_MODELS_DIR%\densenet121.pth')"

echo %GREEN%模型下载完成%RESET%
pause
goto menu

:: 启动程序
:start_program
cls
echo %CYAN%正在启动程序...%RESET%

:: 激活虚拟环境
call "%VENV_PATH%\Scripts\activate.bat"

:: 启动主程序
python src/main.py
if %errorLevel% neq 0 (
    echo %RED%程序启动失败%RESET%
    pause
    goto menu
)

:: 结束
:end
cls
echo %GREEN%感谢使用图片模型训练系统！%RESET%
pause
exit /b 0 
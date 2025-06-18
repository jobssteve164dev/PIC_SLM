@echo off
chcp 65001 > nul
title 图片分类模型训练应用 - EXE打包工具

echo ========================================
echo    图片分类模型训练应用 EXE打包工具
echo ========================================
echo.
echo 此工具将帮助您将应用打包成exe文件
echo 打包后的exe文件可以在其他电脑上独立运行
echo.
echo 请选择打包方式：
echo 1. 简化版打包（推荐，快速）
echo 2. 完整版打包（功能全面）
echo 3. 退出
echo.

set /p choice=请输入选择 (1-3): 

if "%choice%"=="1" (
    echo.
    echo 开始简化版打包...
    python build_exe_simple.py
) else if "%choice%"=="2" (
    echo.
    echo 开始完整版打包...
    python build_exe.py
) else if "%choice%"=="3" (
    echo 退出打包工具
    goto :end
) else (
    echo 无效选择，请重新运行
    pause
    goto :end
)

:end
echo.
echo 感谢使用！
pause 
# 图片模型训练系统

这是一个用于图像处理、标注、训练和预测的综合系统。该系统使用PyQt5构建用户界面，TensorFlow/Keras作为深度学习框架。

## 项目结构

```
├── src/                    # 源代码目录
│   ├── ui/                # 用户界面模块
│   │   ├── base_tab.py    # 标签页基类
│   │   ├── main_window.py # 主窗口
│   │   ├── data_processing_tab.py    # 数据处理标签页
│   │   ├── annotation_tab.py         # 标注标签页
│   │   ├── training_tab.py           # 训练标签页
│   │   ├── prediction_tab.py         # 预测标签页
│   │   ├── evaluation_tab.py         # 评估标签页
│   │   ├── settings_tab.py           # 设置标签页
│   │   ├── about_tab.py              # 关于标签页
│   │   └── training_help_dialog.py   # 训练帮助对话框
│   ├── main.py            # 主程序入口
│   ├── model_trainer.py   # 模型训练模块
│   ├── data_processor.py  # 数据处理模块
│   ├── image_preprocessor.py # 图像预处理模块
│   ├── predictor.py       # 预测模块
│   ├── annotation_tool.py # 标注工具模块
│   └── config_loader.py   # 配置加载模块
├── config/                # 配置文件目录
├── requirements.txt       # 项目依赖
└── start.bat             # 启动脚本
```

## 功能模块

### 1. 数据处理模块
- 图像预处理（调整大小、格式转换等）
- 数据增强（旋转、翻转、亮度调整等）
- 数据集划分和验证

### 2. 标注模块
- 图像标注界面
- 多类别标注支持
- 标注数据导出

### 3. 训练模块
- 模型选择和配置
- 训练参数设置
- 训练过程可视化
- 模型保存和加载

### 4. 预测模块
- 单张图像预测
- 批量预测
- 预测结果可视化

### 5. 评估模块
- 模型性能评估
- 混淆矩阵分析
- 准确率、召回率等指标计算

### 6. 设置模块
- 系统配置管理
- 默认参数设置
- 用户偏好设置

## 使用方法

### 启动程序
双击 `start.bat` 或使用以下命令启动程序：
```bash
python src/main.py
```

### 基本使用流程
1. 在数据处理标签页中准备和预处理数据
2. 在标注标签页中进行图像标注
3. 在训练标签页中配置和训练模型
4. 在预测标签页中进行预测
5. 在评估标签页中评估模型性能

## 依赖项
- Python 3.6+
- PyQt5
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## 开发说明

### 添加新功能
1. 在相应的模块中添加新的功能实现
2. 在UI模块中添加对应的界面元素
3. 更新配置文件以支持新功能

### 修改现有功能
1. 修改对应的功能模块代码
2. 更新UI界面以反映功能变化
3. 确保向后兼容性

## 注意事项
- 首次使用前请确保安装所有依赖项
- 建议使用虚拟环境运行项目
- 定期备份训练数据和模型文件 
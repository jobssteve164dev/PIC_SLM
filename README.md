# 图像缺陷检测系统

## 项目简介
这是一个基于 PyQt5 和深度学习的图像缺陷检测系统，支持图像分类和目标检测两种任务类型。系统提供了完整的工作流程，包括数据预处理、标注、模型训练、评估和预测等功能。

## 主要功能
- **图像预处理**
  - 支持批量图像处理
  - 自动创建分类目录结构
  - 实时进度显示

- **数据标注**
  - 支持分类任务标注
  - 支持目标检测框标注
  - 验证集管理功能

- **模型训练**
  - 支持分类模型训练（ResNet50等）
  - 支持目标检测模型训练（YOLOv5等）
  - 支持预训练模型导入
  - 丰富的训练参数配置
  - 实时训练可视化
  - TensorBoard支持

- **模型评估**
  - 训练过程指标监控
  - 可视化评估结果
  - TensorBoard集成

- **预测功能**
  - 单张图像预测
  - 批量图像预测
  - 实时预测结果显示

## 技术特点
- 多线程处理保证UI响应
- 模块化设计，代码结构清晰
- 完整的配置管理系统
- 丰富的训练参数支持
- 实时进度和状态更新
- 异常处理机制

## 系统要求
- Python 3.6+
- PyQt5
- PyTorch
- TensorBoard
- 其他依赖见 requirements.txt

## 安装说明
1. 克隆项目到本地
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用说明
1. 运行主程序：
```bash
python src/main.py
```

2. 配置文件说明：
- 系统首次运行会自动创建默认配置文件 `config.json`
- 可以通过配置文件修改默认参数：
  - 数据源文件夹
  - 输出文件夹
  - 默认类别
  - 模型文件路径
  - 评估目录
  - TensorBoard日志目录等

## 主要参数配置
### 分类任务
- 模型选择：ResNet50等
- 批次大小：默认32
- 训练轮数：默认20
- 学习率：默认0.001
- 优化器：Adam等
- 数据增强
- 早停机制
- 混合精度训练

### 目标检测任务
- 模型选择：YOLOv5等
- 批次大小：默认16
- 训练轮数：默认50
- 学习率：默认0.0005
- IOU阈值：默认0.5
- 置信度阈值：默认0.25
- Mosaic增强
- 多尺度训练
- EMA支持

## 项目结构
```
├── src/
│   ├── main.py              # 主程序入口
│   ├── ui/                  # UI相关模块
│   ├── data_processor.py    # 数据处理模块
│   ├── model_trainer.py     # 模型训练模块
│   ├── predictor.py         # 预测模块
│   ├── image_preprocessor.py # 图像预处理模块
│   ├── annotation_tool.py   # 标注工具模块
│   └── config_loader.py     # 配置加载模块
├── models/                  # 模型保存目录
├── runs/                    # 训练日志目录
└── config.json             # 配置文件
```

## 注意事项
1. 首次运行会自动创建必要的目录结构
2. 确保有足够的磁盘空间存储训练数据和模型
3. 训练过程中可以通过TensorBoard实时监控
4. 建议使用GPU进行模型训练

## 更新日志
### v1.0.0
- 基础功能实现
- 支持分类和检测两种任务
- TensorBoard集成
- 完整的工作流程支持 
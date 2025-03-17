# 图片模型训练系统

这是一个用于图像处理、标注、训练和预测的综合系统。该系统使用PyQt5构建用户界面，TensorFlow/Keras作为深度学习框架。

## 模块化界面

为了提高代码的可维护性和可扩展性，我们将原来的单一`main_window.py`文件拆分成多个模块化的标签页类：

- `base_tab.py` - 所有标签页的基类
- `data_processing_tab.py` - 图像预处理标签页
- `annotation_tab.py` - 图像标注标签页
- `training_tab.py` - 模型训练标签页
- `prediction_tab.py` - 模型预测标签页
- `batch_prediction_tab.py` - 批量预测标签页
- `evaluation_tab.py` - 模型评估标签页
- `tensorboard_tab.py` - TensorBoard可视化标签页
- `settings_tab.py` - 设置标签页
- `about_tab.py` - 关于标签页
- `new_main_window.py` - 新的主窗口类，负责组织和管理所有标签页

## 如何使用

### 运行新版本

要运行新的模块化版本，请使用以下命令：

```bash
python src/new_main.py
```

### 运行旧版本

如果需要运行原始版本，可以使用以下命令：

```bash
python src/main.py
```

## 功能说明

### 图像预处理

- 调整图像大小
- 数据增强（水平翻转、随机旋转、亮度调整、对比度调整）

### 图像标注

- 创建分类文件夹
- 手动标注图像
- 支持多个缺陷类别

### 模型训练

- 支持多种预训练模型（MobileNetV2, ResNet50, EfficientNetB0, EfficientNetB3, VGG16）
- 可调整训练参数（批次大小、学习率、训练轮数、优化器等）
- 训练过程可视化

### 模型预测

- 单张图像预测
- 批量图像预测
- 预测结果可视化

### 模型评估

- 比较多个模型性能
- 显示准确率、损失、参数量、推理时间等指标

### TensorBoard可视化

- 启动TensorBoard服务
- 可视化训练过程

### 设置

- 默认文件夹设置
- 默认缺陷类别设置
- 自动标注设置

## 开发说明

### 添加新功能

如果需要添加新功能，可以通过以下步骤：

1. 在相应的标签页类中添加新的UI元素和功能
2. 如果需要添加新的标签页，可以创建一个继承自`BaseTab`的新类
3. 在`new_main_window.py`中添加新的标签页实例并连接相应的信号

### 修改现有功能

如果需要修改现有功能，只需修改相应的标签页类文件，而不会影响其他功能模块。

## 依赖项

- Python 3.6+
- PyQt5
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib 
# 配置文件选择器组件 (ConfigProfileSelector)

## 概述

`ConfigProfileSelector` 是一个独立的PyQt5组件，用于在设置标签页中提供配置文件切换功能。该组件允许用户从预定义的配置文件中选择并应用不同的配置。

## 功能特性

- **自动扫描配置文件**：自动扫描 `setting` 目录中的JSON配置文件
- **智能解析**：提取配置文件的元数据信息（名称、描述、版本等）
- **友好界面**：提供直观的下拉框选择器和配置信息显示
- **信号机制**：通过Qt信号机制与其他组件通信
- **错误处理**：完善的错误处理和用户提示
- **刷新功能**：支持手动刷新配置文件列表

## 组件结构

```
ConfigProfileSelector
├── 配置文件管理组框
│   ├── 配置文件选择器 (QComboBox)
│   ├── 刷新按钮
│   ├── 配置信息显示区域
│   └── 应用配置按钮
└── 分隔线
```

## 使用方法

### 1. 基本使用

```python
from src.ui.components.settings.config_profile_selector import ConfigProfileSelector

# 创建组件
config_selector = ConfigProfileSelector()

# 连接信号
config_selector.profile_changed.connect(on_profile_changed)
config_selector.profile_loaded.connect(on_profile_loaded)

# 添加到布局
layout.addWidget(config_selector)
```

### 2. 在设置标签页中集成

组件已集成到 `SettingsTab` 中：

```python
# 在 SettingsTab.__init__ 中
self.config_profile_selector = ConfigProfileSelector()

# 信号连接
self.config_profile_selector.profile_changed.connect(self.on_profile_changed)
self.config_profile_selector.profile_loaded.connect(self.on_profile_loaded)
```

## 信号说明

### profile_changed
- **触发时机**：用户在下拉框中选择配置文件时
- **参数**：`(profile_name: str, config_data: dict)`
- **用途**：预览配置文件内容，但不自动应用

### profile_loaded  
- **触发时机**：用户点击"应用配置"按钮时
- **参数**：`(config_data: dict)`
- **用途**：应用选择的配置文件

## 配置文件格式

组件支持以下JSON配置文件格式：

```json
{
    "config": {
        "default_source_folder": "路径",
        "default_output_folder": "路径",
        "default_classes": ["类别1", "类别2"],
        // ... 其他配置项
    },
    "metadata": {
        "name": "配置名称",
        "description": "配置描述",
        "version": "1.0",
        "export_time": "2024-01-01 12:00:00",
        "created_by": "创建者"
    }
}
```

## API参考

### 主要方法

#### `refresh_profile_list()`
刷新配置文件列表，重新扫描 `setting` 目录。

#### `get_current_profile() -> Tuple[Optional[str], dict]`
获取当前选择的配置文件名和配置数据。

#### `set_current_profile(filename: str) -> bool`
设置当前选择的配置文件。

#### `get_available_profiles() -> List[dict]`
获取所有可用的配置文件列表。

### 内部方法

#### `extract_profile_info(filename: str, config_data: dict) -> dict`
从配置文件中提取元数据信息。

#### `load_profile(profile_info: dict)`
加载指定的配置文件。

#### `update_profile_info_display(profile_info: dict, config_data: dict)`
更新配置文件信息显示。

## 配置文件目录

默认配置文件目录：`setting/`

支持的文件命名规则：
- 文件必须以 `.json` 结尾
- 文件名必须包含 `config` 关键字（不区分大小写）

示例有效文件名：
- `settings_config.json`
- `my_config.json`
- `Config_backup.json`

## 错误处理

组件包含完善的错误处理机制：

1. **文件读取错误**：显示具体错误信息
2. **JSON解析错误**：跳过无效文件并继续处理
3. **配置应用错误**：显示错误对话框
4. **目录不存在**：友好提示用户

## 样式定制

配置信息显示区域使用自定义样式：

```css
QLabel {
    color: #666666;
    font-size: 11px;
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 8px;
    margin: 2px 0px;
}
```

## 测试

使用提供的测试脚本进行组件测试：

```bash
python test_config_selector.py
```

## 注意事项

1. **文件权限**：确保应用对 `setting` 目录有读取权限
2. **文件格式**：配置文件必须是有效的JSON格式
3. **元数据可选**：`metadata` 字段是可选的，但建议添加以提供更好的用户体验
4. **配置结构**：主要配置数据应放在 `config` 字段中

## 维护和扩展

### 添加新功能
- 在类中添加新方法
- 更新信号定义
- 修改UI布局

### 修改配置文件格式支持
- 修改 `extract_profile_info` 方法
- 更新 `load_profile` 方法
- 调整配置应用逻辑

### 自定义样式
- 修改 `init_ui` 方法中的样式定义
- 添加新的UI元素
- 调整布局参数 
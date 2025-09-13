"""
简化版 智能训练设置组件（单页平铺）

仅保留核心、对训练决策有直接影响的设置项：
- 启用、最大/最小迭代与分析间隔
- 收敛阈值、改进阈值
- 过拟合比值阈值（val_loss/train_loss）
- 自动重启与保留最佳模型
- LLM分析开关与适配器类型
- 报告开关与格式（含参数微调报告设置）
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QDoubleSpinBox, QSpinBox,
    QComboBox, QPushButton, QMessageBox, QFileDialog, QFormLayout, QCheckBox, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from typing import Dict, Any
import os
import json


class IntelligentTrainingSettingsWidget(QWidget):
    """简化版智能训练设置组件（单页）"""

    config_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 精简后的默认配置
        self.default_config = {
            'enabled': True,
            'max_iterations': 5,
            'min_iteration_epochs': 2,
            'analysis_interval': 2,
            'monitoring_interval': 30,  # 监控循环间隔（秒）
            'convergence_threshold': 0.01,
            'improvement_threshold': 0.02,
            'auto_restart': True,
            'preserve_best_model': True,

            # 仅保留一个核心阈值：过拟合比值阈值（val_loss/train_loss）
            'overfitting_threshold': 1.30,

            # LLM分析设置（最小化）
            'llm_analysis_enabled': True,
            'adapter_type': 'openai',

            # 报告设置
            'auto_generate_reports': True,
            'report_format': 'json',
            'include_visualizations': True,
            'save_intervention_details': True,

            'parameter_tuning_reports': {
                'enabled': True,
                'save_path': 'reports/parameter_tuning',
                'format': 'markdown',
                'include_llm_analysis': True,
                'include_metrics_comparison': True,
                'include_config_changes': True
            }
        }

        self.current_config = self.default_config.copy()
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(14)

        title = QLabel("智能训练系统设置（精简版）")
        title.setFont(QFont('微软雅黑', 12, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        tips = QLabel("💡 建议新手使用默认设置；仅保留关键阈值与必要选项，降低复杂度。")
        tips.setAlignment(Qt.AlignCenter)
        tips.setStyleSheet("color:#666;padding:6px;background:#f0f8ff;border-radius:5px;font-size:10px;")
        layout.addWidget(tips)

        # 基本设置
        basic_group = QGroupBox("基本设置")
        basic_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        basic_form = QFormLayout(basic_group)

        self.enabled_checkbox = QCheckBox("启用智能训练")
        self.enabled_checkbox.setChecked(self.current_config['enabled'])
        basic_form.addRow("", self.enabled_checkbox)

        self.max_iterations_spinbox = QSpinBox()
        self.max_iterations_spinbox.setRange(1, 20)
        self.max_iterations_spinbox.setValue(self.current_config['max_iterations'])
        basic_form.addRow("最大迭代次数:", self.max_iterations_spinbox)

        self.min_iteration_epochs_spinbox = QSpinBox()
        self.min_iteration_epochs_spinbox.setRange(1, 50)
        self.min_iteration_epochs_spinbox.setValue(self.current_config['min_iteration_epochs'])
        basic_form.addRow("最小迭代轮数:", self.min_iteration_epochs_spinbox)

        self.analysis_interval_spinbox = QSpinBox()
        self.analysis_interval_spinbox.setRange(1, 50)
        self.analysis_interval_spinbox.setValue(self.current_config['analysis_interval'])
        basic_form.addRow("分析间隔:", self.analysis_interval_spinbox)

        self.monitoring_interval_spinbox = QSpinBox()
        self.monitoring_interval_spinbox.setRange(10, 300)
        self.monitoring_interval_spinbox.setValue(self.current_config['monitoring_interval'])
        self.monitoring_interval_spinbox.setToolTip("监控循环检查间隔（秒）。较小值响应更快但占用更多资源。")
        basic_form.addRow("监控间隔(秒):", self.monitoring_interval_spinbox)

        layout.addWidget(basic_group)

        # 收敛/改进阈值
        converge_group = QGroupBox("收敛与改进阈值")
        converge_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        converge_form = QFormLayout(converge_group)

        self.convergence_threshold_spinbox = QDoubleSpinBox()
        self.convergence_threshold_spinbox.setRange(0.001, 0.1)
        self.convergence_threshold_spinbox.setDecimals(3)
        self.convergence_threshold_spinbox.setSingleStep(0.001)
        self.convergence_threshold_spinbox.setValue(self.current_config['convergence_threshold'])
        converge_form.addRow("收敛阈值:", self.convergence_threshold_spinbox)

        self.improvement_threshold_spinbox = QDoubleSpinBox()
        self.improvement_threshold_spinbox.setRange(0.001, 0.1)
        self.improvement_threshold_spinbox.setDecimals(3)
        self.improvement_threshold_spinbox.setSingleStep(0.001)
        self.improvement_threshold_spinbox.setValue(self.current_config['improvement_threshold'])
        converge_form.addRow("改进阈值:", self.improvement_threshold_spinbox)

        layout.addWidget(converge_group)

        # 过拟合阈值
        overfit_group = QGroupBox("阈值设置（精简）")
        overfit_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        overfit_form = QFormLayout(overfit_group)

        self.overfitting_spinbox = QDoubleSpinBox()
        self.overfitting_spinbox.setRange(1.0, 3.0)
        self.overfitting_spinbox.setDecimals(2)
        self.overfitting_spinbox.setSingleStep(0.05)
        self.overfitting_spinbox.setValue(self.current_config['overfitting_threshold'])
        self.overfitting_spinbox.setToolTip("过拟合阈值（验证损失/训练损失 比值）。建议1.2~1.6。")
        overfit_form.addRow("过拟合比值阈值:", self.overfitting_spinbox)

        layout.addWidget(overfit_group)

        # 重启设置
        restart_group = QGroupBox("重启设置")
        restart_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        restart_form = QFormLayout(restart_group)

        self.auto_restart_checkbox = QCheckBox("自动重启训练")
        self.auto_restart_checkbox.setChecked(self.current_config['auto_restart'])
        restart_form.addRow("", self.auto_restart_checkbox)

        self.preserve_best_model_checkbox = QCheckBox("保留最佳模型")
        self.preserve_best_model_checkbox.setChecked(self.current_config['preserve_best_model'])
        restart_form.addRow("", self.preserve_best_model_checkbox)

        layout.addWidget(restart_group)

        # LLM设置
        llm_group = QGroupBox("LLM分析设置")
        llm_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        llm_form = QFormLayout(llm_group)

        self.llm_analysis_checkbox = QCheckBox("启用LLM分析")
        self.llm_analysis_checkbox.setChecked(self.current_config['llm_analysis_enabled'])
        llm_form.addRow("", self.llm_analysis_checkbox)

        self.adapter_type_combo = QComboBox()
        self.adapter_type_combo.addItems(['openai', 'deepseek', 'ollama', 'custom', 'mock'])
        self.adapter_type_combo.setCurrentText(self.current_config['adapter_type'])
        llm_form.addRow("适配器类型:", self.adapter_type_combo)

        layout.addWidget(llm_group)

        # 报告设置
        report_group = QGroupBox("报告设置")
        report_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        report_form = QFormLayout(report_group)

        self.auto_generate_reports_checkbox = QCheckBox("自动生成报告")
        self.auto_generate_reports_checkbox.setChecked(self.current_config['auto_generate_reports'])
        report_form.addRow("", self.auto_generate_reports_checkbox)

        self.report_format_combo = QComboBox()
        self.report_format_combo.addItems(['json', 'html', 'markdown', 'pdf'])
        self.report_format_combo.setCurrentText(self.current_config['report_format'])
        report_form.addRow("报告格式:", self.report_format_combo)

        self.include_visualizations_checkbox = QCheckBox("包含可视化")
        self.include_visualizations_checkbox.setChecked(self.current_config['include_visualizations'])
        report_form.addRow("", self.include_visualizations_checkbox)

        self.save_intervention_details_checkbox = QCheckBox("保存干预详情")
        self.save_intervention_details_checkbox.setChecked(self.current_config['save_intervention_details'])
        report_form.addRow("", self.save_intervention_details_checkbox)

        layout.addWidget(report_group)

        # 参数微调报告
        tuning_group = QGroupBox("参数微调报告设置")
        tuning_group.setFont(QFont('微软雅黑', 10, QFont.Bold))
        tuning_form = QFormLayout(tuning_group)

        tuning_cfg = self.current_config['parameter_tuning_reports']
        self.enable_tuning_reports_checkbox = QCheckBox("启用参数微调报告")
        self.enable_tuning_reports_checkbox.setChecked(tuning_cfg.get('enabled', True))
        tuning_form.addRow("", self.enable_tuning_reports_checkbox)

        path_layout = QHBoxLayout()
        self.tuning_report_path_edit = QLineEdit(tuning_cfg.get('save_path', 'reports/parameter_tuning'))
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_report_path)
        path_layout.addWidget(self.tuning_report_path_edit)
        path_layout.addWidget(browse_btn)
        tuning_form.addRow("保存路径:", path_layout)

        self.tuning_report_format_combo = QComboBox()
        self.tuning_report_format_combo.addItems(['markdown', 'json', 'html'])
        self.tuning_report_format_combo.setCurrentText(tuning_cfg.get('format', 'markdown'))
        tuning_form.addRow("报告格式:", self.tuning_report_format_combo)

        self.include_llm_analysis_checkbox = QCheckBox("包含LLM分析")
        self.include_llm_analysis_checkbox.setChecked(tuning_cfg.get('include_llm_analysis', True))
        tuning_form.addRow("", self.include_llm_analysis_checkbox)

        self.include_metrics_comparison_checkbox = QCheckBox("包含指标对比")
        self.include_metrics_comparison_checkbox.setChecked(tuning_cfg.get('include_metrics_comparison', True))
        tuning_form.addRow("", self.include_metrics_comparison_checkbox)

        self.include_config_changes_checkbox = QCheckBox("包含配置变更")
        self.include_config_changes_checkbox.setChecked(tuning_cfg.get('include_config_changes', True))
        tuning_form.addRow("", self.include_config_changes_checkbox)

        layout.addWidget(tuning_group)

        # 底部操作按钮
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("保存配置")
        self.save_btn.setIcon(QIcon(":/icons/save.png"))
        btn_layout.addWidget(self.save_btn)

        self.reset_btn = QPushButton("重置默认")
        self.reset_btn.setIcon(QIcon(":/icons/reset.png"))
        btn_layout.addWidget(self.reset_btn)

        self.validate_btn = QPushButton("验证配置")
        self.validate_btn.setIcon(QIcon(":/icons/check.png"))
        btn_layout.addWidget(self.validate_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()

    def _connect_signals(self):
        for w in [
            self.enabled_checkbox, self.max_iterations_spinbox, self.min_iteration_epochs_spinbox,
            self.analysis_interval_spinbox, self.monitoring_interval_spinbox, self.convergence_threshold_spinbox,
            self.improvement_threshold_spinbox, self.auto_restart_checkbox,
            self.preserve_best_model_checkbox, self.overfitting_spinbox,
            self.llm_analysis_checkbox, self.adapter_type_combo,
            self.auto_generate_reports_checkbox, self.report_format_combo,
            self.include_visualizations_checkbox, self.save_intervention_details_checkbox,
            self.enable_tuning_reports_checkbox, self.tuning_report_path_edit,
            self.tuning_report_format_combo, self.include_llm_analysis_checkbox,
            self.include_metrics_comparison_checkbox, self.include_config_changes_checkbox
        ]:
            if hasattr(w, 'toggled'):
                w.toggled.connect(self._on_config_changed)
            elif hasattr(w, 'valueChanged'):
                w.valueChanged.connect(self._on_config_changed)
            elif hasattr(w, 'currentTextChanged'):
                w.currentTextChanged.connect(self._on_config_changed)
            elif hasattr(w, 'textChanged'):
                w.textChanged.connect(self._on_config_changed)

        self.save_btn.clicked.connect(self._save_config)
        self.reset_btn.clicked.connect(self._reset_to_default)
        self.validate_btn.clicked.connect(self._validate_config)

    def _on_config_changed(self):
        self._update_current_config()
        self.config_changed.emit(self.current_config)

    def _update_current_config(self):
        self.current_config = {
            'enabled': self.enabled_checkbox.isChecked(),
            'max_iterations': self.max_iterations_spinbox.value(),
            'min_iteration_epochs': self.min_iteration_epochs_spinbox.value(),
            'analysis_interval': self.analysis_interval_spinbox.value(),
            'monitoring_interval': self.monitoring_interval_spinbox.value(),
            'convergence_threshold': self.convergence_threshold_spinbox.value(),
            'improvement_threshold': self.improvement_threshold_spinbox.value(),
            'auto_restart': self.auto_restart_checkbox.isChecked(),
            'preserve_best_model': self.preserve_best_model_checkbox.isChecked(),

            'overfitting_threshold': self.overfitting_spinbox.value(),

            'llm_analysis_enabled': self.llm_analysis_checkbox.isChecked(),
            'adapter_type': self.adapter_type_combo.currentText(),

            'auto_generate_reports': self.auto_generate_reports_checkbox.isChecked(),
            'report_format': self.report_format_combo.currentText(),
            'include_visualizations': self.include_visualizations_checkbox.isChecked(),
            'save_intervention_details': self.save_intervention_details_checkbox.isChecked(),

            'parameter_tuning_reports': {
                'enabled': self.enable_tuning_reports_checkbox.isChecked(),
                'save_path': self.tuning_report_path_edit.text(),
                'format': self.tuning_report_format_combo.currentText(),
                'include_llm_analysis': self.include_llm_analysis_checkbox.isChecked(),
                'include_metrics_comparison': self.include_metrics_comparison_checkbox.isChecked(),
                'include_config_changes': self.include_config_changes_checkbox.isChecked()
            }
        }

    def get_current_config(self) -> Dict[str, Any]:
        self._update_current_config()
        return self.current_config.copy()

    def set_config(self, config: Dict[str, Any]):
        cfg = {**self.default_config, **(config or {})}

        self.enabled_checkbox.setChecked(cfg['enabled'])
        self.max_iterations_spinbox.setValue(cfg['max_iterations'])
        self.min_iteration_epochs_spinbox.setValue(cfg['min_iteration_epochs'])
        self.analysis_interval_spinbox.setValue(cfg['analysis_interval'])
        self.monitoring_interval_spinbox.setValue(cfg.get('monitoring_interval', 30))
        self.convergence_threshold_spinbox.setValue(cfg['convergence_threshold'])
        self.improvement_threshold_spinbox.setValue(cfg['improvement_threshold'])
        self.auto_restart_checkbox.setChecked(cfg['auto_restart'])
        self.preserve_best_model_checkbox.setChecked(cfg['preserve_best_model'])

        self.overfitting_spinbox.setValue(cfg['overfitting_threshold'])

        self.llm_analysis_checkbox.setChecked(cfg['llm_analysis_enabled'])
        self.adapter_type_combo.setCurrentText(cfg['adapter_type'])

        self.auto_generate_reports_checkbox.setChecked(cfg['auto_generate_reports'])
        self.report_format_combo.setCurrentText(cfg['report_format'])
        self.include_visualizations_checkbox.setChecked(cfg['include_visualizations'])
        self.save_intervention_details_checkbox.setChecked(cfg['save_intervention_details'])

        tuning_cfg = cfg.get('parameter_tuning_reports', {})
        self.enable_tuning_reports_checkbox.setChecked(tuning_cfg.get('enabled', True))
        self.tuning_report_path_edit.setText(tuning_cfg.get('save_path', 'reports/parameter_tuning'))
        self.tuning_report_format_combo.setCurrentText(tuning_cfg.get('format', 'markdown'))
        self.include_llm_analysis_checkbox.setChecked(tuning_cfg.get('include_llm_analysis', True))
        self.include_metrics_comparison_checkbox.setChecked(tuning_cfg.get('include_metrics_comparison', True))
        self.include_config_changes_checkbox.setChecked(tuning_cfg.get('include_config_changes', True))

        self.current_config = cfg.copy()

    def _browse_report_path(self):
        try:
            current_path = self.tuning_report_path_edit.text() or 'reports/parameter_tuning'
            if not os.path.exists(current_path):
                os.makedirs(current_path, exist_ok=True)
            folder = QFileDialog.getExistingDirectory(self, "选择参数微调报告保存路径", current_path)
            if folder:
                self.tuning_report_path_edit.setText(folder)
                self._on_config_changed()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"选择路径时发生错误: {str(e)}")

    def _save_config(self):
        try:
            self._update_current_config()
            config_file = "setting/intelligent_training_config.json"
            os.makedirs(os.path.dirname(config_file), exist_ok=True)

            full_config = {
                'enabled': self.current_config['enabled'],
                'max_iterations': self.current_config['max_iterations'],
                'min_iteration_epochs': self.current_config['min_iteration_epochs'],
                'analysis_interval': self.current_config['analysis_interval'],
                'monitoring_interval': self.current_config['monitoring_interval'],
                'convergence_threshold': self.current_config['convergence_threshold'],
                'improvement_threshold': self.current_config['improvement_threshold'],
                'auto_restart': self.current_config['auto_restart'],
                'preserve_best_model': self.current_config['preserve_best_model'],
                'overfitting_threshold': self.current_config['overfitting_threshold'],
                'llm_config': {
                    'adapter_type': self.current_config['adapter_type']
                },
                'auto_generate_reports': self.current_config['auto_generate_reports'],
                'report_format': self.current_config['report_format'],
                'include_visualizations': self.current_config['include_visualizations'],
                'save_intervention_details': self.current_config['save_intervention_details'],
                'parameter_tuning_reports': self.current_config['parameter_tuning_reports']
            }

            # 保存到智能训练专用配置文件
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(full_config, f, ensure_ascii=False, indent=2)

            # 同时更新主配置文件中的智能训练配置
            self._update_main_config_file()

            self.config_changed.emit(self.current_config)
            QMessageBox.information(self, "成功", "智能训练配置已保存")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存配置失败: {str(e)}")
    
    def _update_main_config_file(self):
        """更新主配置文件中的智能训练配置"""
        try:
            main_config_file = "config.json"
            
            # 读取主配置文件
            if os.path.exists(main_config_file):
                with open(main_config_file, 'r', encoding='utf-8') as f:
                    main_config = json.load(f)
            else:
                main_config = {}
            
            # 更新智能训练配置部分
            if 'intelligent_training' not in main_config:
                main_config['intelligent_training'] = {}
            
            # 只更新编排器相关的核心配置
            orchestrator_keys = [
                'enabled', 'max_iterations', 'min_iteration_epochs', 'analysis_interval', 'monitoring_interval',
                'convergence_threshold', 'improvement_threshold', 'auto_restart', 'preserve_best_model'
            ]
            
            for key in orchestrator_keys:
                if key in self.current_config:
                    main_config['intelligent_training'][key] = self.current_config[key]
            
            # 保存更新后的主配置文件
            with open(main_config_file, 'w', encoding='utf-8') as f:
                json.dump(main_config, f, ensure_ascii=False, indent=4)
            
            print(f"[INFO] 已同步更新主配置文件中的智能训练配置")
            
        except Exception as e:
            print(f"[WARNING] 更新主配置文件失败: {str(e)}")
            # 不抛出异常，因为智能训练专用配置文件已经保存成功

    def _reset_to_default(self):
        self.set_config(self.default_config)
        self.config_changed.emit(self.current_config)
        QMessageBox.information(self, "成功", "已重置为默认配置")

    def _validate_config(self):
        try:
            self._update_current_config()
            cfg = self.current_config
            warns = []

            if cfg['max_iterations'] < 2:
                warns.append("最大迭代次数建议至少为2")
            if cfg['min_iteration_epochs'] < 2:
                warns.append("最小迭代轮数建议至少为2")
            if cfg['monitoring_interval'] < 10 or cfg['monitoring_interval'] > 300:
                warns.append("监控间隔建议在[10, 300]秒之间")
            if cfg['convergence_threshold'] < 0.001 or cfg['convergence_threshold'] > 0.1:
                warns.append("收敛阈值建议在[0.001, 0.1]")
            if cfg['improvement_threshold'] < 0.001 or cfg['improvement_threshold'] > 0.1:
                warns.append("改进阈值建议在[0.001, 0.1]")
            if cfg['overfitting_threshold'] < 1.0 or cfg['overfitting_threshold'] > 3.0:
                warns.append("过拟合阈值建议在[1.0, 3.0]")

            if warns:
                QMessageBox.warning(self, "配置建议", "\n".join(warns))
            else:
                QMessageBox.information(self, "配置验证", "✅ 配置验证通过")
        except Exception as e:
            QMessageBox.critical(self, "验证错误", f"配置验证失败: {str(e)}")




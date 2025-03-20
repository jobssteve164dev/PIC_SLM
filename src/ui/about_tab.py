from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QTextBrowser
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
import os
from .base_tab import BaseTab

class AboutTab(BaseTab):
    """关于标签页，显示应用信息"""
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent, main_window)
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self.scroll_content)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 添加标题
        title_label = QLabel("关于应用")
        title_label.setFont(QFont('微软雅黑', 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 添加应用信息
        info_group = QGroupBox("应用信息")
        info_layout = QVBoxLayout()
        
        app_name = QLabel("图片模型训练系统")
        app_name.setFont(QFont('微软雅黑', 12, QFont.Bold))
        app_name.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(app_name)
        
        version = QLabel("版本: 1.0.0")
        version.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(version)
        
        copyright_label = QLabel("版权所有 © 2023-2024  遵循 AGPL-3.0 许可协议")
        copyright_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(copyright_label)
        
        info_group.setLayout(info_layout)
        main_layout.addWidget(info_group)
        
        # 添加功能说明
        features_group = QGroupBox("功能说明")
        features_layout = QVBoxLayout()
        
        features_text = QTextBrowser()
        features_text.setOpenExternalLinks(True)
        features_text.setHtml("""
        <h3>主要功能</h3>
        <ul>
            <li><b>图像预处理</b>: 调整图像大小，数据增强</li>
            <li><b>图像标注</b>: 手动标注图像，创建分类文件夹</li>
            <li><b>模型训练</b>: 训练图像分类模型，支持多种预训练模型</li>
            <li><b>模型预测</b>: 使用训练好的模型进行预测</li>
            <li><b>批量预测</b>: 批量处理图像并生成预测结果</li>
            <li><b>模型评估</b>: 比较不同模型的性能</li>
            <li><b>TensorBoard</b>: 可视化训练过程</li>
        </ul>
        """)
        features_layout.addWidget(features_text)
        
        features_group.setLayout(features_layout)
        main_layout.addWidget(features_group)
        
        # 添加技术说明
        tech_group = QGroupBox("技术说明")
        tech_layout = QVBoxLayout()
        
        tech_text = QTextBrowser()
        tech_text.setOpenExternalLinks(True)
        tech_text.setHtml("""
        <h3>技术栈</h3>
        <ul>
            <li><b>UI框架</b>: PyQt5</li>
            <li><b>深度学习框架</b>: TensorFlow/Keras</li>
            <li><b>图像处理</b>: OpenCV</li>
            <li><b>数据处理</b>: NumPy, Pandas</li>
            <li><b>可视化</b>: Matplotlib, TensorBoard</li>
        </ul>
        """)
        tech_layout.addWidget(tech_text)
        
        tech_group.setLayout(tech_layout)
        main_layout.addWidget(tech_group)
        
        # 添加许可证信息
        license_group = QGroupBox("许可证信息")
        license_layout = QVBoxLayout()
        
        license_text = QTextBrowser()
        license_text.setOpenExternalLinks(True)
        license_text.setHtml("""
        <h3>AGPL许可协议</h3>
        <p>本软件根据<b>GNU Affero通用公共许可证第3版(AGPL-3.0)</b>发布。</p>
        
        <p>根据AGPL-3.0协议:</p>
        <ul>
            <li>您可以自由使用、修改和分发本软件</li>
            <li>如果您修改本软件并提供网络服务，您必须公开您的修改版本源代码</li>
            <li>任何基于本软件的衍生作品必须以相同的许可证发布</li>
            <li>您必须保留版权和许可声明</li>
        </ul>
        
        <p>完整的许可证文本可在以下网址查看: <a href="https://www.gnu.org/licenses/agpl-3.0.html">https://www.gnu.org/licenses/agpl-3.0.html</a></p>
        """)
        license_layout.addWidget(license_text)
        
        license_group.setLayout(license_layout)
        main_layout.addWidget(license_group)
        
        # 添加弹性空间
        main_layout.addStretch()
        
        # 设置滚动区域
        self.layout.addWidget(self.scroll_content) 
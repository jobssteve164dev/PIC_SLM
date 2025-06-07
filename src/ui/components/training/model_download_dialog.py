from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QTextBrowser, QMessageBox)
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices
import os
import platform
import subprocess


class ModelDownloadDialog(QDialog):
    """模型下载失败时显示的对话框，提供下载链接和文件夹打开按钮"""
    
    def __init__(self, model_name, model_link, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.model_link = model_link
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("预训练模型下载失败")
        self.setMinimumWidth(600)
        self.setMinimumHeight(450)
        
        layout = QVBoxLayout(self)
        
        # 添加说明标签
        title_label = QLabel(f"预训练模型 <b>{self.model_name}</b> 下载失败")
        title_label.setStyleSheet("font-size: 16px; color: #C62828;")
        layout.addWidget(title_label)
        
        # 添加详细说明
        info_text = QTextBrowser()
        info_text.setOpenExternalLinks(True)
        info_text.setStyleSheet("background-color: #F5F5F5; border: 1px solid #E0E0E0;")
        info_text.setHtml(f"""
        <h3>预训练模型下载失败</h3>
        <p>PyTorch无法自动下载预训练模型，可能是由于以下原因：</p>
        <ul>
            <li>网络连接问题（如防火墙限制、代理设置等）</li>
            <li>服务器暂时不可用</li>
            <li>下载超时</li>
        </ul>
        
        <h3>预训练模型的优势</h3>
        <p>使用预训练模型可以显著提高训练效果和速度：</p>
        <ul>
            <li>缩短训练时间，通常减少50%-90%的训练轮数</li>
            <li>提高模型精度，特别是在训练数据量较少的情况下</li>
            <li>更好的泛化能力和特征提取能力</li>
        </ul>
        
        <h3>解决方法</h3>
        <p>您可以：</p>
        <ol>
            <li>手动下载模型文件: <a href="{self.model_link}"><b>{self.model_name}</b> 模型下载链接</a></li>
            <li>将下载的文件放入PyTorch的模型缓存目录</li>
        </ol>
        
        <h4>模型缓存目录通常位于:</h4>
        <ul>
            <li>Windows: <code>%USERPROFILE%\\.cache\\torch\\hub\\checkpoints</code></li>
            <li>Linux: <code>~/.cache/torch/hub/checkpoints</code></li>
            <li>macOS: <code>~/Library/Caches/torch/hub/checkpoints</code> 或 <code>~/.cache/torch/hub/checkpoints</code></li>
        </ul>
        
        <p>放置模型文件后，<b>不需要重命名文件</b>，直接将下载的文件放入目录即可。</p>
        <p>下载完成后，重新开始训练即可使用预训练模型。</p>
        
        <p><i>注意：如果不使用预训练模型，当前训练会继续进行，但可能需要更长时间才能达到相同精度。</i></p>
        """)
        layout.addWidget(info_text)
        
        # 添加按钮
        button_layout = QHBoxLayout()
        
        # 打开下载链接按钮
        open_link_btn = QPushButton("打开下载链接")
        open_link_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        open_link_btn.setMinimumWidth(120)
        open_link_btn.clicked.connect(self.open_download_link)
        
        # 打开缓存目录按钮
        open_cache_btn = QPushButton("打开模型缓存目录")
        open_cache_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        open_cache_btn.setMinimumWidth(150)
        open_cache_btn.clicked.connect(self.open_cache_directory)
        
        # 关闭按钮
        close_btn = QPushButton("继续训练(不使用预训练)")
        close_btn.setStyleSheet("padding: 8px;")
        close_btn.clicked.connect(self.accept)
        
        button_layout.addWidget(open_link_btn)
        button_layout.addWidget(open_cache_btn)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def open_download_link(self):
        """打开模型下载链接"""
        QDesktopServices.openUrl(QUrl(self.model_link))
    
    def open_cache_directory(self):
        """打开PyTorch模型缓存目录"""
        try:
            # 根据不同操作系统找到缓存目录
            cache_dir = self.get_torch_cache_dir()
            
            # 确保目录存在
            os.makedirs(cache_dir, exist_ok=True)
            
            # 打开目录
            if platform.system() == "Windows":
                os.startfile(cache_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", cache_dir])
            else:  # Linux
                subprocess.run(["xdg-open", cache_dir])
            
        except Exception as e:
            QMessageBox.warning(self, "无法打开目录", f"无法打开PyTorch缓存目录: {str(e)}")
    
    def get_torch_cache_dir(self):
        """获取PyTorch缓存目录路径"""
        # 获取用户主目录
        home_dir = os.path.expanduser("~")
        
        # 根据操作系统确定缓存目录
        if platform.system() == "Windows":
            cache_dir = os.path.join(os.environ.get("USERPROFILE", home_dir), ".cache", "torch", "hub", "checkpoints")
        elif platform.system() == "Darwin":  # macOS
            # 先尝试macOS特有的缓存目录，如果不存在则使用通用的.cache目录
            macos_cache = os.path.join(home_dir, "Library", "Caches", "torch", "hub", "checkpoints")
            if os.path.exists(os.path.dirname(macos_cache)):
                cache_dir = macos_cache
            else:
                cache_dir = os.path.join(home_dir, ".cache", "torch", "hub", "checkpoints")
        else:  # Linux
            cache_dir = os.path.join(home_dir, ".cache", "torch", "hub", "checkpoints")
        
        return cache_dir 
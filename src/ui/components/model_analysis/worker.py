import torch
import torch.nn as nn
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PIL import Image
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm


class ModelAnalysisWorker(QThread):
    """模型分析工作线程"""
    
    analysis_finished = pyqtSignal(str, object)  # 分析类型, 结果
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.image = None
        self.image_tensor = None
        self.class_names = []
        self.analysis_type = ""
        self.target_class = 0
        self.analysis_params = {}
        self.is_stopped = False
        
    def set_analysis_task(self, analysis_type, model, image, image_tensor, class_names, target_class, params=None):
        """设置分析任务"""
        self.analysis_type = analysis_type
        self.model = model
        self.image = image
        self.image_tensor = image_tensor
        self.class_names = class_names
        self.target_class = target_class
        self.analysis_params = params or {}
        self.is_stopped = False
        
    def stop_analysis(self):
        """停止分析任务"""
        self.is_stopped = True
        self.status_updated.emit("正在停止分析...")
        
    def run(self):
        """执行分析任务"""
        try:
            if self.analysis_type == "特征可视化":
                result = self._feature_visualization()
            elif self.analysis_type == "GradCAM":
                result = self._gradcam_analysis()
            elif self.analysis_type == "LIME解释":
                result = self._lime_analysis()
            elif self.analysis_type == "敏感性分析":
                result = self._sensitivity_analysis()
            elif self.analysis_type == "Integrated Gradients":
                result = self._integrated_gradients_analysis()
            elif self.analysis_type == "SHAP解释":
                result = self._shap_analysis()
            elif self.analysis_type == "SmoothGrad":
                result = self._smoothgrad_analysis()
            else:
                raise ValueError(f"未知的分析类型: {self.analysis_type}")
                
            if not self.is_stopped:
                self.analysis_finished.emit(self.analysis_type, result)
            
        except Exception as e:
            self.error_occurred.emit(f"{self.analysis_type}分析失败: {str(e)}")
            
    def _feature_visualization(self):
        """特征可视化"""
        self.status_updated.emit("正在提取特征...")
        features = {}
        
        def hook_fn(module, input, output):
            features[module] = output.detach()
            
        # 注册钩子
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                hooks.append(module.register_forward_hook(hook_fn))
                
        # 前向传播
        with torch.no_grad():
            _ = self.model(self.image_tensor.unsqueeze(0))
            
        # 移除钩子
        for hook in hooks:
            hook.remove()
            
        self.progress_updated.emit(100)
        return features
        
    def _gradcam_analysis(self):
        """GradCAM分析"""
        self.status_updated.emit("正在生成GradCAM...")
        
        gradients = {}
        activations = {}
        
        def forward_hook(module, input, output):
            activations['value'] = output
            
        def backward_hook(module, grad_input, grad_output):
            gradients['value'] = grad_output[0]
            
        # 找到最后一个卷积层
        last_conv_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module
                
        if last_conv_layer is None:
            raise ValueError("未找到卷积层")
            
        # 注册钩子
        forward_handle = last_conv_layer.register_forward_hook(forward_hook)
        backward_handle = last_conv_layer.register_backward_hook(backward_hook)
        
        try:
            # 前向传播
            self.model.eval()
            input_tensor = self.image_tensor.unsqueeze(0)
            input_tensor.requires_grad_()
            
            output = self.model(input_tensor)
            
            # 反向传播
            self.model.zero_grad()
            output[0, self.target_class].backward(retain_graph=True)
            
            # 计算GradCAM
            grads = gradients['value'].detach()
            acts = activations['value'].detach()
            
            weights = torch.mean(grads, dim=(2, 3))
            gradcam = torch.zeros(acts.shape[2:], device=acts.device)
            
            for i in range(acts.shape[1]):
                gradcam += weights[0, i] * acts[0, i]
                
            gradcam = torch.relu(gradcam)
            # 避免除零错误
            max_val = torch.max(gradcam)
            if max_val > 0:
                gradcam = gradcam / max_val
            
            self.progress_updated.emit(100)
            return gradcam.detach().cpu().numpy()
            
        finally:
            forward_handle.remove()
            backward_handle.remove()
            
    def _lime_analysis(self):
        """LIME分析"""
        self.status_updated.emit("正在进行LIME分析...")
        
        # 获取参数
        num_superpixels = self.analysis_params.get('num_superpixels', 100)
        num_samples = self.analysis_params.get('num_samples', 1000)
        
        def predict_fn(images):
            """预测函数"""
            batch_predictions = []
            for img in images:
                # 检查是否已停止
                if self.is_stopped:
                    return np.array([])
                    
                # 转换为tensor并应用与训练时相同的预处理
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                
                # 应用标准化 (假设使用ImageNet预训练模型)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std
                
                img_tensor = img_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    pred = self.model(img_tensor)
                    pred = torch.softmax(pred, dim=1)
                    
                batch_predictions.append(pred.cpu().numpy()[0])
                
            return np.array(batch_predictions)
            
        # 创建LIME解释器
        explainer = lime_image.LimeImageExplainer()
        
        # 转换图像
        image_array = np.array(self.image)
        
        # 生成解释 - 确保包含目标类别
        explanation = explainer.explain_instance(
            image_array,
            predict_fn,
            top_labels=5,  # 解释前5个类别，确保包含目标类别
            hide_color=0,  # 隐藏不重要区域的颜色
            num_samples=num_samples,
            segmentation_fn=SegmentationAlgorithm('quickshift', kernel_size=1,
                                                max_dist=100, ratio=0.1)
        )
        
        self.progress_updated.emit(100)
        return explanation
        
    def _sensitivity_analysis(self):
        """敏感性分析"""
        self.status_updated.emit("正在进行敏感性分析...")
        
        # 获取参数
        perturbation_range = self.analysis_params.get('perturbation_range', 0.1)
        num_steps = self.analysis_params.get('num_steps', 20)
        
        # 生成扰动
        epsilons = np.linspace(0, perturbation_range, num_steps)
        predictions = []
        
        original_tensor = self.image_tensor.unsqueeze(0)
        
        with torch.no_grad():
            original_pred = self.model(original_tensor)
            original_confidence = torch.softmax(original_pred, dim=1)[0, self.target_class].item()
            
        for i, epsilon in enumerate(epsilons):
            # 检查是否已停止
            if self.is_stopped:
                return {
                    'epsilons': epsilons[:i],
                    'predictions': predictions,
                    'original_confidence': original_confidence
                }
                
            # 添加随机噪声
            noise = torch.randn_like(original_tensor) * epsilon
            perturbed_tensor = original_tensor + noise
            
            with torch.no_grad():
                pred = self.model(perturbed_tensor)
                confidence = torch.softmax(pred, dim=1)[0, self.target_class].item()
                predictions.append(confidence)
                
            self.progress_updated.emit(int((i + 1) / num_steps * 100))
            
        return {
            'epsilons': epsilons,
            'predictions': predictions,
            'original_confidence': original_confidence
        }
    
    def _integrated_gradients_analysis(self):
        """Integrated Gradients分析"""
        self.status_updated.emit("正在计算Integrated Gradients...")
        
        # 获取参数
        num_steps = self.analysis_params.get('ig_steps', 50)
        baseline_type = self.analysis_params.get('baseline_type', 'zero')
        
        # 创建基线图像
        if baseline_type == 'zero':
            baseline = torch.zeros_like(self.image_tensor)
        elif baseline_type == 'gaussian':
            baseline = torch.randn_like(self.image_tensor) * 0.1
        elif baseline_type == 'mean':
            baseline = torch.ones_like(self.image_tensor) * 0.5
        else:
            baseline = torch.zeros_like(self.image_tensor)
        
        # 计算积分梯度
        integrated_gradients = torch.zeros_like(self.image_tensor)
        
        for i in range(num_steps):
            # 检查是否已停止
            if self.is_stopped:
                return integrated_gradients.detach().cpu().numpy()
            
            # 线性插值
            alpha = i / (num_steps - 1) if num_steps > 1 else 0
            interpolated = baseline + alpha * (self.image_tensor - baseline)
            interpolated = interpolated.unsqueeze(0)
            interpolated.requires_grad_()
            
            # 前向传播
            output = self.model(interpolated)
            
            # 反向传播
            self.model.zero_grad()
            output[0, self.target_class].backward()
            
            # 累加梯度
            if interpolated.grad is not None:
                integrated_gradients += interpolated.grad.squeeze(0)
            
            self.progress_updated.emit(int((i + 1) / num_steps * 100))
        
        # 计算最终的积分梯度
        integrated_gradients = integrated_gradients * (self.image_tensor - baseline) / num_steps
        
        return integrated_gradients.detach().cpu().numpy()
    
    def _shap_analysis(self):
        """SHAP分析"""
        try:
            import shap
        except ImportError:
            raise ImportError("请安装shap库: pip install shap")
        
        self.status_updated.emit("正在进行SHAP分析...")
        
        # 获取参数
        num_samples = self.analysis_params.get('shap_samples', 100)
        
        # 使用简化的梯度SHAP方法，更稳定
        try:
            # 准备输入
            input_tensor = self.image_tensor.unsqueeze(0)
            
            # 创建梯度SHAP解释器
            explainer = shap.GradientExplainer(self.model, input_tensor)
            
            # 计算SHAP值
            shap_values = explainer.shap_values(input_tensor)
            
            self.progress_updated.emit(100)
            
            # 处理返回的SHAP值
            if isinstance(shap_values, list):
                # 多类分类情况，选择目标类别
                if self.target_class < len(shap_values):
                    selected_shap_values = shap_values[self.target_class][0]
                else:
                    selected_shap_values = shap_values[0][0]
                
                return {
                    'shap_values': selected_shap_values,
                    'method': 'gradient',
                    'target_class': self.target_class
                }
            else:
                # 单一SHAP值
                return {
                    'shap_values': shap_values[0],
                    'method': 'gradient',
                    'target_class': self.target_class
                }
                
        except Exception as e:
            # 如果SHAP失败，使用简单的梯度方法作为替代
            self.status_updated.emit("SHAP失败，使用简单梯度方法...")
            
            input_tensor = self.image_tensor.unsqueeze(0)
            input_tensor.requires_grad_()
            
            # 前向传播
            output = self.model(input_tensor)
            
            # 反向传播
            self.model.zero_grad()
            output[0, self.target_class].backward()
            
            # 获取梯度
            gradients = input_tensor.grad.squeeze(0).detach().cpu().numpy()
            
            self.progress_updated.emit(100)
            
            return {
                'shap_values': gradients,
                'method': 'simple_gradient',
                'target_class': self.target_class
            }
    
    def _smoothgrad_analysis(self):
        """SmoothGrad分析"""
        self.status_updated.emit("正在计算SmoothGrad...")
        
        # 获取参数
        num_samples = self.analysis_params.get('smoothgrad_samples', 50)
        noise_level = self.analysis_params.get('noise_level', 0.15)
        
        # 累积梯度
        smooth_gradients = torch.zeros_like(self.image_tensor)
        
        for i in range(num_samples):
            # 检查是否已停止
            if self.is_stopped:
                return smooth_gradients.detach().cpu().numpy()
            
            # 添加高斯噪声
            noise = torch.randn_like(self.image_tensor) * noise_level
            noisy_input = self.image_tensor + noise
            noisy_input = noisy_input.unsqueeze(0)
            noisy_input.requires_grad_()
            
            # 前向传播
            output = self.model(noisy_input)
            
            # 反向传播
            self.model.zero_grad()
            output[0, self.target_class].backward()
            
            # 累加梯度
            if noisy_input.grad is not None:
                smooth_gradients += noisy_input.grad.squeeze(0)
            
            self.progress_updated.emit(int((i + 1) / num_samples * 100))
        
        # 平均梯度
        smooth_gradients = smooth_gradients / num_samples
        
        return smooth_gradients.detach().cpu().numpy() 
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import shutil
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from typing import Dict, Tuple, List, Optional
import time
from src.utils.logger import get_logger, log_error, performance_monitor, PerformanceMonitor

class BatchPredictionThread(QThread):
    """批量预测独立线程"""
    
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    prediction_finished = pyqtSignal(dict)
    prediction_error = pyqtSignal(str)
    
    def __init__(self, predictor, params):
        super().__init__()
        self.predictor = predictor
        self.params = params
        self._stop_processing = False
        self.logger = get_logger(__name__, "batch_prediction")
        
    def run(self):
        """线程运行入口"""
        try:
            with PerformanceMonitor("batch_prediction_full", "batch_prediction"):
                self._batch_predict()
        except Exception as e:
            import traceback
            error_msg = f"批量预测线程出错: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            log_error(e, {"operation": "batch_prediction_thread"}, "batch_prediction")
            self.prediction_error.emit(error_msg)
    
    def stop_processing(self):
        """停止处理"""
        self._stop_processing = True
        
    def _batch_predict(self):
        """执行批量预测"""
        batch_start_time = time.time()
        print("=" * 60)
        print("开始批量预测（独立线程）")
        print(f"批量预测开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            if self.predictor.model is None:
                print("❌ 错误: 模型未加载")
                self.prediction_error.emit('请先加载模型')
                return
            else:
                print(f"✅ 模型已加载: {type(self.predictor.model).__name__}")
                print(f"✅ 设备: {self.predictor.device}")
                print(f"✅ 模型状态: {'训练模式' if self.predictor.model.training else '评估模式'}")
            
            if self.predictor.class_names is None or len(self.predictor.class_names) == 0:
                print("❌ 错误: 类别信息未加载")
                self.prediction_error.emit('类别信息未加载，请先加载模型')
                return
            else:
                print(f"✅ 类别信息已加载: {len(self.predictor.class_names)} 个类别")
                print(f"   类别列表: {self.predictor.class_names}")
                
            source_folder = self.params.get('source_folder')
            target_folder = self.params.get('target_folder')
            
            print(f"📁 源文件夹: {source_folder}")
            print(f"📁 目标文件夹: {target_folder}")
            
            # 验证必要参数
            if not source_folder:
                self.prediction_error.emit('源文件夹路径不能为空')
                return
            
            if not target_folder:
                self.prediction_error.emit('目标文件夹路径不能为空')
                return
            
            if not os.path.exists(source_folder):
                self.prediction_error.emit(f'源文件夹不存在: {source_folder}')
                return
            
            if not os.path.isdir(source_folder):
                self.prediction_error.emit(f'源路径不是文件夹: {source_folder}')
                return
            
            confidence_threshold = self.params.get('confidence_threshold', 50.0)  # 默认50%
            copy_mode = self.params.get('copy_mode', 'copy')
            create_subfolders = self.params.get('create_subfolders', True)
            
            print(f"⚙️ 置信度阈值: {confidence_threshold}%")
            print(f"⚙️ 文件操作模式: {copy_mode}")
            print(f"⚙️ 创建子文件夹: {create_subfolders}")
            
            # 获取所有图片文件
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
            image_files = [f for f in os.listdir(source_folder) 
                          if os.path.isfile(os.path.join(source_folder, f)) and 
                          os.path.splitext(f.lower())[1] in valid_extensions]
            
            if not image_files:
                print("❌ 未找到图片文件")
                self.status_updated.emit('未找到图片文件')
                return
            
            print(f"📷 找到 {len(image_files)} 张图片")
            print(f"   图片格式统计: {dict((ext, sum(1 for f in image_files if f.lower().endswith(ext))) for ext in valid_extensions if any(f.lower().endswith(ext) for f in image_files))}")
                
            # 创建目标文件夹
            os.makedirs(target_folder, exist_ok=True)
            
            # 如果需要创建子文件夹，为每个类别创建一个子文件夹
            if create_subfolders:
                for class_name in self.predictor.class_names:
                    os.makedirs(os.path.join(target_folder, class_name), exist_ok=True)
                print(f"📁 已创建 {len(self.predictor.class_names)} 个类别子文件夹")
            
            # 统计结果
            results = {
                'total': len(image_files),
                'processed': 0,
                'classified': 0,
                'unclassified': 0,
                'class_counts': {class_name: 0 for class_name in self.predictor.class_names}
            }
            
            prediction_times = []
            
            # 批量处理图片
            print("\n🔄 开始处理图片...")
            for i, image_file in enumerate(image_files):
                if self._stop_processing:
                    print("⏹️ 批量处理已停止")
                    self.status_updated.emit('批量处理已停止')
                    break
                    
                image_path = os.path.join(source_folder, image_file)
                
                # 记录单张图片预测时间
                single_start = time.time()
                result = self.predictor.predict_image(image_path)
                single_time = time.time() - single_start
                prediction_times.append(single_time)
                
                if result:
                    # 获取最高置信度的预测
                    top_prediction = result['predictions'][0]
                    class_name = top_prediction['class_name']
                    probability = top_prediction['probability']
                    
                    # 更新进度
                    progress = int((i + 1) / len(image_files) * 100)
                    self.progress_updated.emit(progress)
                    self.status_updated.emit(f'处理图片 {i+1}/{len(image_files)}: {image_file}')
                    
                    results['processed'] += 1
                    
                    # 如果置信度高于阈值，则分类图片
                    print(f"🔍 置信度比较: {image_file}")
                    print(f"   预测置信度: {probability:.2f}%")
                    print(f"   设定阈值: {confidence_threshold}%")
                    print(f"   比较结果: {probability:.2f}% {'≥' if probability >= confidence_threshold else '<'} {confidence_threshold}%")
                    
                    if probability >= confidence_threshold:
                        print(f"   ✅ 置信度达标，将分类到: {class_name}")
                        # 确定目标路径
                        if create_subfolders:
                            target_path = os.path.join(target_folder, class_name, image_file)
                        else:
                            # 如果不创建子文件夹，则在文件名前添加类别名称
                            base_name, ext = os.path.splitext(image_file)
                            target_path = os.path.join(target_folder, f"{class_name}_{base_name}{ext}")
                        
                        # 复制或移动文件
                        try:
                            if copy_mode == 'copy':
                                shutil.copy2(image_path, target_path)
                            else:  # move
                                shutil.move(image_path, target_path)
                                
                            results['classified'] += 1
                            results['class_counts'][class_name] += 1
                            print(f"   ✅ 文件已{('复制' if copy_mode == 'copy' else '移动')}到: {target_path}")
                        except Exception as e:
                            print(f"❌ 处理文件 {image_file} 时出错: {str(e)}")
                            self.status_updated.emit(f'处理文件 {image_file} 时出错: {str(e)}')
                    else:
                        results['unclassified'] += 1
                        print(f"   ❌ 置信度不达标，未分类")
                        print(f"⚠️ 图片 {image_file} 置信度过低 ({probability:.2f}% < {confidence_threshold}%)，未分类")
                else:
                    print(f"❌ 图片 {image_file} 预测失败")
            
            # 计算统计信息
            batch_total_time = time.time() - batch_start_time
            avg_prediction_time = sum(prediction_times) / len(prediction_times) if prediction_times else 0
            
            print("\n" + "=" * 60)
            print("批量预测完成统计:")
            print(f"⏱️ 总耗时: {batch_total_time:.2f}秒")
            print(f"⏱️ 平均每张图片预测时间: {avg_prediction_time:.4f}秒")
            print(f"🏃 预测速度: {len(image_files)/batch_total_time:.2f} 张/秒")
            print(f"📊 总图片数: {results['total']}")
            print(f"📊 已处理: {results['processed']}")
            print(f"📊 已分类: {results['classified']}")
            print(f"📊 未分类: {results['unclassified']}")
            print("📊 各类别统计:")
            for class_name, count in results['class_counts'].items():
                if count > 0:
                    print(f"   {class_name}: {count} 张")
            print("=" * 60)
            
            # 发送完成信号
            self.prediction_finished.emit(results)
            self.status_updated.emit('批量处理完成')
            
        except Exception as e:
            print(f"❌ 批量预测过程中出错: {str(e)}")
            self.prediction_error.emit(f'批量预测过程中出错: {str(e)}')

class Predictor(QObject):
    # 定义信号
    prediction_finished = pyqtSignal(dict)
    prediction_error = pyqtSignal(str)
    batch_prediction_progress = pyqtSignal(int)
    batch_prediction_status = pyqtSignal(str)
    batch_prediction_finished = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.class_names = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._stop_batch_processing = False
        self.batch_prediction_thread = None  # 添加线程引用
        self.logger = get_logger(__name__, "predictor")
        
        self.logger.info(f"预测器初始化完成，使用设备: {self.device}")

    @performance_monitor("load_model", "predictor")
    def load_model(self, model_path: str, class_info_path: str) -> None:
        """
        加载模型和类别信息
        """
        try:
            self.logger.info(f"开始加载模型: {model_path}")
            self.logger.info(f"类别信息文件: {class_info_path}")
            
            # 加载类别信息
            with open(class_info_path, 'r', encoding='utf-8') as f:
                class_info = json.load(f)
            self.class_names = class_info['class_names']
            self.logger.info(f"类别信息加载成功，类别数量: {len(self.class_names)}")
            self.logger.debug(f"类别列表: {self.class_names}")

            # 1. 尝试直接创建ResNet模型并加载权重
            try:
                num_classes = len(self.class_names)
                # 尝试多种常见的模型架构
                models_to_try = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
                
                for model_arch in models_to_try:
                    print(f"尝试加载模型架构: {model_arch}")
                    # 导入对应的模型构建函数
                    if model_arch == 'resnet18':
                        from torchvision.models import resnet18
                        self.model = resnet18(pretrained=False)
                    elif model_arch == 'resnet34':
                        from torchvision.models import resnet34
                        self.model = resnet34(pretrained=False)
                    elif model_arch == 'resnet50':
                        from torchvision.models import resnet50
                        self.model = resnet50(pretrained=False)
                    elif model_arch == 'resnet101':
                        from torchvision.models import resnet101
                        self.model = resnet101(pretrained=False)
                    
                    # 修改最后一层以匹配类别数
                    in_features = self.model.fc.in_features
                    self.model.fc = nn.Linear(in_features, num_classes)
                    
                    # 加载权重
                    state_dict = torch.load(model_path, map_location=self.device)
                    
                    # 尝试多种可能的权重格式
                    if isinstance(state_dict, dict):
                        if 'model_state_dict' in state_dict:
                            state_dict = state_dict['model_state_dict']
                        elif 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                    
                    # 尝试加载权重
                    try:
                        self.model.load_state_dict(state_dict, strict=False)
                        print(f"成功加载模型: {model_arch}")
                        break  # 如果加载成功，跳出循环
                    except Exception as e:
                        print(f"加载模型 {model_arch} 失败: {str(e)}")
                        continue  # 尝试下一个模型架构
                
                # 检查模型是否加载成功
                if self.model is None:
                    raise Exception("所有模型架构尝试均失败")
                
                # 将模型移动到正确的设备并设置为评估模式
                self.model.to(self.device)
                self.model.eval()
                print(f"模型加载成功，类别数量: {len(self.class_names)}")
                
            except Exception as arch_error:
                print(f"尝试加载模型架构失败: {str(arch_error)}")
                raise arch_error

        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            error_msg = f'加载模型时出错: {str(e)}\n{traceback_str}'
            print(error_msg)
            self.prediction_error.emit(error_msg)

    def predict(self, image_path: str, top_k: int = 3) -> None:
        """
        预测单张图片类别
        Args:
            image_path: 图片路径
            top_k: 返回前k个预测结果
        """
        try:
            if self.model is None:
                self.prediction_error.emit("模型未加载，请先加载模型")
                return
                
            # 加载和预处理图片
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as img_err:
                self.prediction_error.emit(f"无法加载图像: {str(img_err)}")
                return
                
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 预测
            try:
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    # 检查输出格式
                    if isinstance(outputs, tuple):
                        # 有些模型返回多个输出，我们取第一个
                        outputs = outputs[0]
                    
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    top_k = min(top_k, len(self.class_names), probabilities.size(1))
                    top_prob, top_class = torch.topk(probabilities, top_k)
            except Exception as pred_err:
                self.prediction_error.emit(f"预测过程出错: {str(pred_err)}")
                return

            # 获取预测结果
            predictions = []
            for i in range(top_k):
                try:
                    class_idx = top_class[0][i].item()
                    if 0 <= class_idx < len(self.class_names):
                        prob = top_prob[0][i].item()
                        predictions.append({
                            'class_name': self.class_names[class_idx],
                            'probability': prob * 100  # 转换为百分比
                        })
                except Exception as idx_err:
                    print(f"处理预测结果 {i} 时出错: {str(idx_err)}")
                    continue

            if not predictions:
                self.prediction_error.emit("无法获取有效的预测结果")
                return
                
            # 发送预测结果
            result = {
                'predictions': predictions,
                'image_path': image_path
            }
            self.prediction_finished.emit(result)

        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            self.prediction_error.emit(f'预测过程中出错: {str(e)}\n{traceback_str}')

    def predict_image(self, image_path: str, top_k: int = 3) -> Optional[Dict]:
        """
        预测单张图片类别并返回结果（不发送信号）
        Args:
            image_path: 图片路径
            top_k: 返回前k个预测结果
        """
        start_time = time.time()
        
        try:
            if self.model is None:
                print("模型未加载，请先加载模型")
                return None
                
            # 加载和预处理图片
            try:
                load_start = time.time()
                image = Image.open(image_path).convert('RGB')
                load_time = time.time() - load_start
                print(f"图片加载时间: {load_time:.4f}秒")
            except Exception as img_err:
                print(f"无法加载图像: {str(img_err)}")
                return None
                
            preprocess_start = time.time()
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            preprocess_time = time.time() - preprocess_start
            print(f"图片预处理时间: {preprocess_time:.4f}秒")

            # 预测
            try:
                predict_start = time.time()
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    # 检查输出格式
                    if isinstance(outputs, tuple):
                        # 有些模型返回多个输出，我们取第一个
                        outputs = outputs[0]
                    
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    top_k = min(top_k, len(self.class_names), probabilities.size(1))
                    top_prob, top_class = torch.topk(probabilities, top_k)
                predict_time = time.time() - predict_start
                print(f"模型推理时间: {predict_time:.4f}秒")
                print(f"模型输出shape: {outputs.shape}")
                print(f"概率分布前3个值: {probabilities[0][:3].tolist()}")
            except Exception as pred_err:
                print(f"预测过程出错: {str(pred_err)}")
                return None

            # 获取预测结果
            predictions = []
            for i in range(top_k):
                try:
                    class_idx = top_class[0][i].item()
                    if 0 <= class_idx < len(self.class_names):
                        prob = top_prob[0][i].item()
                        predictions.append({
                            'class_name': self.class_names[class_idx],
                            'probability': prob * 100  # 转换为百分比
                        })
                        print(f"预测结果 {i+1}: {self.class_names[class_idx]} - {prob*100:.2f}%")
                except Exception as idx_err:
                    print(f"处理预测结果 {i} 时出错: {str(idx_err)}")
                    continue

            if not predictions:
                print("无法获取有效的预测结果")
                return None
                
            total_time = time.time() - start_time
            print(f"单张图片预测总时间: {total_time:.4f}秒")
            print(f"图片: {os.path.basename(image_path)}, 最高置信度: {predictions[0]['class_name']} ({predictions[0]['probability']:.2f}%)")
                
            return {
                'predictions': predictions,
                'image_path': image_path
            }

        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f'预测图片 {image_path} 时出错: {str(e)}\n{traceback_str}')
            return None

    def batch_predict(self, params: Dict) -> None:
        """
        启动批量预测线程
        
        参数:
            params: 包含批量预测参数的字典
                - source_folder: 源图片文件夹
                - target_folder: 目标文件夹（分类后的图片将保存在这里）
                - confidence_threshold: 置信度阈值，只有高于此阈值的预测才会被接受
                - copy_mode: 'copy'（复制）或 'move'（移动）
                - create_subfolders: 是否为每个类别创建子文件夹
        """
        # 如果已有线程在运行，先停止它
        if self.batch_prediction_thread and self.batch_prediction_thread.isRunning():
            print("⚠️ 已有批量预测线程在运行，先停止它")
            self.stop_batch_processing()
            self.batch_prediction_thread.wait()  # 等待线程结束
        
        # 创建新的批量预测线程
        self.batch_prediction_thread = BatchPredictionThread(self, params)
            
        # 连接线程信号到Predictor的信号
        self.batch_prediction_thread.progress_updated.connect(self.batch_prediction_progress.emit)
        self.batch_prediction_thread.status_updated.connect(self.batch_prediction_status.emit)
        self.batch_prediction_thread.prediction_finished.connect(self.batch_prediction_finished.emit)
        self.batch_prediction_thread.prediction_error.connect(self.prediction_error.emit)
            
        # 启动线程
        print("🚀 启动批量预测独立线程")
        self.batch_prediction_thread.start()

    def stop_batch_processing(self) -> None:
        """停止批量处理"""
        self._stop_batch_processing = True
        
        # 如果有运行中的批量预测线程，停止它
        if self.batch_prediction_thread and self.batch_prediction_thread.isRunning():
            print("🛑 正在停止批量预测线程...")
            self.batch_prediction_thread.stop_processing()
            # 不在这里wait()，让调用者决定是否等待
    
    def is_batch_prediction_running(self) -> bool:
        """检查批量预测线程是否正在运行"""
        return self.batch_prediction_thread is not None and self.batch_prediction_thread.isRunning()
    
    def wait_for_batch_prediction_to_finish(self, timeout_ms: int = 5000) -> bool:
        """等待批量预测线程完成
        
        Args:
            timeout_ms: 超时时间（毫秒）
            
        Returns:
            bool: True表示线程正常结束，False表示超时
        """
        if self.batch_prediction_thread and self.batch_prediction_thread.isRunning():
            return self.batch_prediction_thread.wait(timeout_ms)
        return True
    
    def cleanup_batch_prediction_thread(self):
        """清理批量预测线程"""
        if self.batch_prediction_thread:
            if self.batch_prediction_thread.isRunning():
                self.batch_prediction_thread.stop_processing()
                self.batch_prediction_thread.wait(3000)  # 等待3秒
            self.batch_prediction_thread.deleteLater()
            self.batch_prediction_thread = None

    def _create_model(self, model_name: str, num_classes: int) -> nn.Module:
        """创建模型"""
        if model_name == 'ResNet50':
            from torchvision.models import resnet50
            model = resnet50(pretrained=False)  # 不使用预训练权重
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f'不支持的模型: {model_name}')
        
        return model 

    def load_model_with_info(self, model_info: Dict) -> None:
        """
        根据模型信息加载模型和类别信息
        
        参数:
            model_info: 包含模型信息的字典
                - model_path: 模型文件路径
                - class_info_path: 类别信息文件路径
                - model_type: 模型类型（分类模型或检测模型）
                - model_arch: 模型架构（ResNet18, ResNet34等）
        """
        try:
            model_path = model_info.get('model_path')
            class_info_path = model_info.get('class_info_path')
            model_type = model_info.get('model_type')
            model_arch = model_info.get('model_arch')
            
            print(f"加载模型信息: {model_type} - {model_arch}")
            print(f"模型路径: {model_path}")
            print(f"类别信息路径: {class_info_path}")
            
            # 加载类别信息
            with open(class_info_path, 'r', encoding='utf-8') as f:
                class_info = json.load(f)
            self.class_names = class_info['class_names']
            print(f"类别信息加载成功，类别: {self.class_names}")
            
            # 根据模型类型和架构加载不同的模型
            if model_type == "分类模型":
                self._load_classification_model(model_path, model_arch)
            elif model_type == "检测模型":
                self._load_detection_model(model_path, model_arch)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
                
            print(f"模型加载成功: {model_type} - {model_arch}")
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            error_msg = f'加载模型时出错: {str(e)}\n{traceback_str}'
            print(error_msg)
            self.prediction_error.emit(error_msg)
            
    def _load_classification_model(self, model_path: str, model_arch: str) -> None:
        """加载分类模型"""
        try:
            num_classes = len(self.class_names)
            
            # 根据架构创建对应的模型
            if model_arch in ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]:
                # ResNet系列模型
                if model_arch == "ResNet18":
                    from torchvision.models import resnet18
                    self.model = resnet18(pretrained=False)
                    in_features = self.model.fc.in_features
                elif model_arch == "ResNet34":
                    from torchvision.models import resnet34
                    self.model = resnet34(pretrained=False)
                    in_features = self.model.fc.in_features
                elif model_arch == "ResNet50":
                    from torchvision.models import resnet50
                    self.model = resnet50(pretrained=False)
                    in_features = self.model.fc.in_features
                elif model_arch == "ResNet101":
                    from torchvision.models import resnet101
                    self.model = resnet101(pretrained=False)
                    in_features = self.model.fc.in_features
                elif model_arch == "ResNet152":
                    from torchvision.models import resnet152
                    self.model = resnet152(pretrained=False)
                    in_features = self.model.fc.in_features
                
                # 修改分类头
                self.model.fc = nn.Linear(in_features, num_classes)
            
            elif model_arch in ["MobileNetV2", "MobileNetV3", "MobileNetV3Small", "MobileNetV3Large"]:
                # MobileNet系列
                if model_arch == "MobileNetV2":
                    from torchvision.models import mobilenet_v2
                    self.model = mobilenet_v2(pretrained=False)
                    in_features = self.model.classifier[1].in_features
                    self.model.classifier[1] = nn.Linear(in_features, num_classes)
                elif model_arch == "MobileNetV3" or model_arch == "MobileNetV3Large":
                    from torchvision.models import mobilenet_v3_large
                    self.model = mobilenet_v3_large(pretrained=False)
                    in_features = self.model.classifier[3].in_features
                    self.model.classifier[3] = nn.Linear(in_features, num_classes)
                elif model_arch == "MobileNetV3Small":
                    from torchvision.models import mobilenet_v3_small
                    self.model = mobilenet_v3_small(pretrained=False)
                    in_features = self.model.classifier[3].in_features
                    self.model.classifier[3] = nn.Linear(in_features, num_classes)
            
            elif model_arch.startswith("EfficientNet"):
                # EfficientNet系列
                if model_arch == "EfficientNetB0":
                    from torchvision.models import efficientnet_b0
                    self.model = efficientnet_b0(pretrained=False)
                elif model_arch == "EfficientNetB1":
                    from torchvision.models import efficientnet_b1
                    self.model = efficientnet_b1(pretrained=False)
                elif model_arch == "EfficientNetB2":
                    from torchvision.models import efficientnet_b2
                    self.model = efficientnet_b2(pretrained=False)
                elif model_arch == "EfficientNetB3":
                    from torchvision.models import efficientnet_b3
                    self.model = efficientnet_b3(pretrained=False)
                elif model_arch == "EfficientNetB4":
                    from torchvision.models import efficientnet_b4
                    self.model = efficientnet_b4(pretrained=False)
                
                # 修改分类头
                in_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(in_features, num_classes)
            
            elif model_arch in ["VGG16", "VGG19"]:
                # VGG系列
                if model_arch == "VGG16":
                    from torchvision.models import vgg16
                    self.model = vgg16(pretrained=False)
                elif model_arch == "VGG19":
                    from torchvision.models import vgg19
                    self.model = vgg19(pretrained=False)
                
                # 修改分类头
                in_features = self.model.classifier[6].in_features
                self.model.classifier[6] = nn.Linear(in_features, num_classes)
            
            elif model_arch.startswith("DenseNet"):
                # DenseNet系列
                if model_arch == "DenseNet121":
                    from torchvision.models import densenet121
                    self.model = densenet121(pretrained=False)
                elif model_arch == "DenseNet169":
                    from torchvision.models import densenet169
                    self.model = densenet169(pretrained=False)
                elif model_arch == "DenseNet201":
                    from torchvision.models import densenet201
                    self.model = densenet201(pretrained=False)
                    
                # 修改分类头
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Linear(in_features, num_classes)
            
            elif model_arch in ["InceptionV3", "Xception"]:
                # 其他复杂模型
                print(f"注意: {model_arch}模型需要特殊的预处理，可能需要调整transform")
                if model_arch == "InceptionV3":
                    from torchvision.models import inception_v3
                    self.model = inception_v3(pretrained=False)
                    in_features = self.model.fc.in_features
                    self.model.fc = nn.Linear(in_features, num_classes)
                elif model_arch == "Xception":
                    # Xception需要外部库支持
                    try:
                        from pretrainedmodels import xception
                        self.model = xception(num_classes=1000, pretrained=None)
                        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, num_classes)
                    except ImportError:
                        raise ImportError("加载Xception模型需要安装pretrainedmodels库: pip install pretrainedmodels")
            
            else:
                raise ValueError(f"不支持的分类模型架构: {model_arch}")
                
            print(f"成功创建模型架构: {model_arch}")
                
            # 加载模型权重
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 处理不同格式的模型文件
            if isinstance(state_dict, dict):
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
            
            # 加载权重（使用非严格模式，允许部分权重不匹配）
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"成功加载模型权重: {model_path}")
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            error_msg = f"加载分类模型失败: {str(e)}\n{traceback_str}"
            print(error_msg)
            raise Exception(error_msg)
            
    def _load_detection_model(self, model_path: str, model_arch: str) -> None:
        """加载检测模型"""
        try:
            # YOLO系列模型
            if model_arch.startswith("YOLO"):
                if model_arch == "YOLOv5":
                    # 尝试导入YOLOv5
                    try:
                        import yolov5
                        self.model = yolov5.load(model_path)
                        print(f"使用yolov5库加载模型: {model_path}")
                    except ImportError:
                        raise ImportError("需要安装YOLOv5库: pip install yolov5")
                elif model_arch == "YOLOv8":
                    # 尝试导入Ultralytics YOLOv8
                    try:
                        from ultralytics import YOLO
                        self.model = YOLO(model_path)
                        print(f"使用ultralytics库加载模型: {model_path}")
                    except ImportError:
                        raise ImportError("需要安装Ultralytics库: pip install ultralytics")
                elif model_arch in ["YOLOv7", "YOLOv6", "YOLOv4", "YOLOv3"]:
                    # 这些YOLO版本需要特殊处理，可能需要导入专门的代码库
                    try:
                        if model_arch == "YOLOv7":
                            # YOLOv7通常需要特定的代码库和权重格式
                            from models.experimental import attempt_load
                            self.model = attempt_load(model_path, map_location=self.device)
                            print(f"使用YOLOv7专用加载器加载模型: {model_path}")
                        elif model_arch == "YOLOv6":
                            # YOLOv6可能需要安装特定版本
                            from yolov6.core.inferer import Inferer
                            self.model = Inferer(model_path, device=self.device)
                            print(f"使用YOLOv6库加载模型: {model_path}")
                        elif model_arch in ["YOLOv4", "YOLOv3"]:
                            # YOLOv3/v4可能使用Darknet格式
                            import cv2
                            self.model = cv2.dnn.readNetFromDarknet(model_path, model_path.replace(".weights", ".cfg"))
                            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                            print(f"使用OpenCV DNN加载{model_arch}模型: {model_path}")
                    except ImportError as e:
                        raise ImportError(f"加载{model_arch}需要安装特定库: {str(e)}")
                    except Exception as e:
                        raise Exception(f"加载{model_arch}模型失败: {str(e)}")
            
            # SSD系列模型
            elif model_arch.startswith("SSD"):
                try:
                    import torch
                    # 不同SSD变体的处理
                    if model_arch == "SSD":
                        from torchvision.models.detection import ssd300_vgg16
                        self.model = ssd300_vgg16(pretrained=False, num_classes=len(self.class_names) + 1)
                    elif model_arch == "SSD300":
                        from torchvision.models.detection import ssd300_vgg16
                        self.model = ssd300_vgg16(pretrained=False, num_classes=len(self.class_names) + 1)
                    elif model_arch == "SSD512":
                        # 需要自定义实现或第三方库
                        raise NotImplementedError("SSD512需要特定的实现，目前尚未支持")
                    
                    # 加载模型权重
                    state_dict = torch.load(model_path, map_location=self.device)
                    
                    # 处理不同格式的模型文件
                    if isinstance(state_dict, dict):
                        if 'model_state_dict' in state_dict:
                            state_dict = state_dict['model_state_dict']
                        elif 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                    
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"成功加载SSD模型: {model_path}")
                    
                except ImportError:
                    raise ImportError("加载SSD模型需要安装PyTorch和torchvision")
                
            # Faster R-CNN和Mask R-CNN系列
            elif model_arch in ["Faster R-CNN", "Mask R-CNN"]:
                try:
                    import torch
                    if model_arch == "Faster R-CNN":
                        from torchvision.models.detection import fasterrcnn_resnet50_fpn
                        self.model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=len(self.class_names) + 1)
                    elif model_arch == "Mask R-CNN":
                        from torchvision.models.detection import maskrcnn_resnet50_fpn
                        self.model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=len(self.class_names) + 1)
                        
                    # 加载模型权重
                    state_dict = torch.load(model_path, map_location=self.device)
                    
                    # 处理不同格式的模型文件
                    if isinstance(state_dict, dict):
                        if 'model_state_dict' in state_dict:
                            state_dict = state_dict['model_state_dict']
                        elif 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                    
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"成功加载{model_arch}模型: {model_path}")
                    
                except ImportError:
                    raise ImportError(f"加载{model_arch}模型需要安装PyTorch和torchvision")
            
            # RetinaNet模型
            elif model_arch == "RetinaNet":
                try:
                    import torch
                    from torchvision.models.detection import retinanet_resnet50_fpn
                    self.model = retinanet_resnet50_fpn(pretrained=False, num_classes=len(self.class_names) + 1)
                    
                    # 加载模型权重
                    state_dict = torch.load(model_path, map_location=self.device)
                    
                    # 处理不同格式的模型文件
                    if isinstance(state_dict, dict):
                        if 'model_state_dict' in state_dict:
                            state_dict = state_dict['model_state_dict']
                        elif 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                    
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"成功加载RetinaNet模型: {model_path}")
                    
                except ImportError:
                    raise ImportError("加载RetinaNet模型需要安装PyTorch和torchvision")
            
            # DETR模型
            elif model_arch == "DETR":
                try:
                    import torch
                    # 尝试使用DETR
                    try:
                        from transformers import DetrForObjectDetection
                        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
                        # 修改分类头以适应自定义类别数量
                        self.model.config.num_labels = len(self.class_names)
                        self.model.class_labels_classifier = nn.Linear(
                            in_features=self.model.class_labels_classifier.in_features,
                            out_features=len(self.class_names)
                        )
                    except ImportError:
                        # 备选：使用torchvision的DETR实现
                        from torchvision.models.detection import detr_resnet50
                        self.model = detr_resnet50(pretrained=False, num_classes=len(self.class_names) + 1)
                    
                    # 加载模型权重
                    state_dict = torch.load(model_path, map_location=self.device)
                    
                    # 处理不同格式的模型文件
                    if isinstance(state_dict, dict):
                        if 'model_state_dict' in state_dict:
                            state_dict = state_dict['model_state_dict']
                        elif 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                    
                    self.model.load_state_dict(state_dict, strict=False)
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"成功加载DETR模型: {model_path}")
                    
                except ImportError:
                    raise ImportError("加载DETR模型需要安装transformers库或最新版本的torchvision")
            
            else:
                raise ValueError(f"不支持的检测模型架构: {model_arch}")
                
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            error_msg = f"加载检测模型失败: {str(e)}\n{traceback_str}"
            print(error_msg)
            raise Exception(error_msg) 
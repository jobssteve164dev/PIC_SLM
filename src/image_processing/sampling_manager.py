"""
采样管理器 - 处理图像数据的欠采样和过采样
用于自动平衡训练类别样本数量差距大的情况
"""

import os
import random
import shutil
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Callable, Optional
from sklearn.model_selection import train_test_split
from .image_transformer import ImageTransformer
from .augmentation_manager import AugmentationManager
from .advanced_sampling import AdvancedSamplingManager
from .advanced_oversampling import AdvancedOversamplingManager


class SamplingManager:
    """采样管理器，处理类别不平衡的采样策略"""
    
    def __init__(self, image_transformer: ImageTransformer, 
                 augmentation_manager: AugmentationManager):
        self.image_transformer = image_transformer
        self.augmentation_manager = augmentation_manager
        self.advanced_sampling = AdvancedSamplingManager()
        self.advanced_oversampling = AdvancedOversamplingManager(image_transformer, augmentation_manager)
        
    def balance_dataset_with_sampling(self, params: Dict, 
                                    get_image_files_func: Callable[[str], List[str]],
                                    progress_callback: Optional[Callable[[int], None]] = None,
                                    status_callback: Optional[Callable[[str], None]] = None,
                                    stop_check: Optional[Callable[[], bool]] = None) -> Dict:
        """
        使用采样方法平衡数据集
        
        Args:
            params: 预处理参数
            get_image_files_func: 获取图片文件的函数
            progress_callback: 进度回调函数
            status_callback: 状态回调函数  
            stop_check: 停止检查函数
            
        Returns:
            Dict: 包含采样结果统计信息的字典
        """
        if status_callback:
            status_callback("开始分析类别分布...")
            
        # 分析类别分布
        class_stats = self._analyze_class_distribution(
            params['source_folder'], 
            params['class_names'], 
            get_image_files_func,
            status_callback
        )
        
        if not class_stats:
            if status_callback:
                status_callback("错误：未找到有效的类别数据")
            return {}
            
        # 确定采样策略
        sampling_strategy = params.get('sampling_strategy', 'auto')
        target_samples = self._determine_target_samples(class_stats, sampling_strategy, params)
        
        if status_callback:
            status_callback(f"目标样本数: {target_samples}, 采样策略: {sampling_strategy}")
            
        # 执行采样
        sampling_results = self._execute_sampling(
            params, class_stats, target_samples, 
            get_image_files_func, progress_callback, 
            status_callback, stop_check
        )
        
        return sampling_results
        
    def _analyze_class_distribution(self, source_folder: str, class_names: List[str],
                                  get_image_files_func: Callable[[str], List[str]],
                                  status_callback: Optional[Callable[[str], None]] = None) -> Dict:
        """分析各类别的样本分布"""
        class_stats = {}
        
        for class_name in class_names:
            class_folder = os.path.join(source_folder, class_name)
            if os.path.exists(class_folder) and os.path.isdir(class_folder):
                image_files = get_image_files_func(class_folder)
                class_stats[class_name] = {
                    'count': len(image_files),
                    'files': image_files,
                    'folder': class_folder
                }
                
                if status_callback:
                    status_callback(f"类别 {class_name}: {len(image_files)} 个样本")
            else:
                if status_callback:
                    status_callback(f"警告：类别文件夹不存在: {class_folder}")
                    
        return class_stats
        
    def _determine_target_samples(self, class_stats: Dict, sampling_strategy: str, 
                                params: Dict) -> int:
        """确定目标样本数量"""
        counts = [stats['count'] for stats in class_stats.values()]
        
        if sampling_strategy == 'auto':
            # 自动策略：根据不平衡程度选择
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio <= 2:
                # 轻微不平衡，使用中位数
                target = int(np.median(counts))
            elif imbalance_ratio <= 5:
                # 中度不平衡，使用中位数和最大值的平均
                target = int((np.median(counts) + max_count) / 2)
            else:
                # 严重不平衡，使用最大值的80%
                target = int(max_count * 0.8)
                
        elif sampling_strategy == 'oversample':
            # 过采样：以最多的类别为目标
            target = max(counts)
            
        elif sampling_strategy == 'undersample':
            # 欠采样：以最少的类别为目标
            target = min(counts)
            
        elif sampling_strategy == 'median':
            # 中位数采样
            target = int(np.median(counts))
            
        elif sampling_strategy == 'custom':
            # 自定义目标样本数
            target = params.get('target_samples_per_class', int(np.mean(counts)))
            
        else:
            # 默认使用平均数
            target = int(np.mean(counts))
            
        return max(target, 10)  # 确保至少有10个样本
        
    def _execute_sampling(self, params: Dict, class_stats: Dict, target_samples: int,
                         get_image_files_func: Callable[[str], List[str]],
                         progress_callback: Optional[Callable[[int], None]] = None,
                         status_callback: Optional[Callable[[str], None]] = None,
                         stop_check: Optional[Callable[[], bool]] = None) -> Dict:
        """执行采样操作"""
        
        # 创建输出目录
        output_folder = params['target_folder']
        dataset_folder = params['dataset_folder']
        train_folder = os.path.join(dataset_folder, 'train')
        val_folder = os.path.join(dataset_folder, 'val')
        
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        
        sampling_results = {
            'original_distribution': {},
            'target_distribution': {},
            'final_distribution': {},
            'sampling_methods': {}
        }
        
        total_classes = len(class_stats)
        processed_classes = 0
        
        for class_name, stats in class_stats.items():
            if stop_check and stop_check():
                if status_callback:
                    status_callback('采样已停止')
                break
                
            if status_callback:
                status_callback(f"处理类别: {class_name}")
                
            original_count = stats['count']
            sampling_results['original_distribution'][class_name] = original_count
            sampling_results['target_distribution'][class_name] = target_samples
            
            # 创建类别文件夹
            class_output_folder = os.path.join(output_folder, class_name)
            class_train_folder = os.path.join(train_folder, class_name)
            class_val_folder = os.path.join(val_folder, class_name)
            
            os.makedirs(class_output_folder, exist_ok=True)
            os.makedirs(class_train_folder, exist_ok=True)
            os.makedirs(class_val_folder, exist_ok=True)
            
            # 执行采样
            if original_count < target_samples:
                # 过采样
                sampled_files = self._oversample_class(
                    class_name, stats, target_samples, params, 
                    class_output_folder, status_callback
                )
                sampling_results['sampling_methods'][class_name] = 'oversample'
                
            elif original_count > target_samples:
                # 欠采样
                sampled_files = self._undersample_class(
                    class_name, stats, target_samples, params,
                    class_output_folder, status_callback
                )
                sampling_results['sampling_methods'][class_name] = 'undersample'
                
            else:
                # 无需采样，直接处理
                sampled_files = self._process_class_without_sampling(
                    class_name, stats, params, class_output_folder, status_callback
                )
                sampling_results['sampling_methods'][class_name] = 'none'
                
            # 划分训练集和验证集
            train_files, val_files = train_test_split(
                sampled_files, train_size=params['train_ratio'], random_state=42
            )
            
            # 复制到训练集和验证集文件夹
            self._copy_files_to_dataset(train_files, class_train_folder)
            self._copy_files_to_dataset(val_files, class_val_folder)
            
            sampling_results['final_distribution'][class_name] = len(sampled_files)
            
            # 更新进度
            processed_classes += 1
            if progress_callback:
                progress = int((processed_classes / total_classes) * 100)
                progress_callback(progress)
                
        return sampling_results
        
    def _oversample_class(self, class_name: str, stats: Dict, target_samples: int,
                         params: Dict, output_folder: str, 
                         status_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """过采样处理"""
        original_files = stats['files']
        original_count = len(original_files)
        needed_samples = target_samples - original_count
        
        if status_callback:
            status_callback(f"过采样 {class_name}: {original_count} -> {target_samples} (+{needed_samples})")
            
        sampled_files = []
        
        # 首先处理所有原始文件
        for i, img_file in enumerate(original_files):
            source_path = os.path.join(stats['folder'], img_file)
            file_name = os.path.splitext(img_file)[0]
            target_path = os.path.join(output_folder, f"{file_name}.{params['format']}")
            
            # 预处理原始图片
            self.image_transformer.process_single_image(
                source_path, target_path, params['width'], params['height'],
                params['brightness_value'], params['contrast_value']
            )
            sampled_files.append(target_path)
            
        # 生成增强样本来达到目标数量
        oversample_method = params.get('oversample_method', 'augmentation')
        
        if oversample_method == 'augmentation':
            # 使用数据增强生成新样本
            additional_files = self._generate_augmented_samples(
                original_files, stats['folder'], output_folder, 
                needed_samples, params, class_name, status_callback
            )
            sampled_files.extend(additional_files)
            
        elif oversample_method == 'duplication':
            # 使用重复采样
            additional_files = self._generate_duplicated_samples(
                original_files, stats['folder'], output_folder,
                needed_samples, params, class_name, status_callback
            )
            sampled_files.extend(additional_files)
            
        elif oversample_method == 'mixup':
            # Mixup过采样
            additional_files = self.advanced_oversampling.mixup_oversampling(
                original_files, stats['folder'], output_folder,
                needed_samples, params, class_name, status_callback
            )
            sampled_files.extend(additional_files)
            
        elif oversample_method == 'cutmix':
            # CutMix过采样
            additional_files = self.advanced_oversampling.cutmix_oversampling(
                original_files, stats['folder'], output_folder,
                needed_samples, params, class_name, status_callback
            )
            sampled_files.extend(additional_files)
            
        elif oversample_method == 'interpolation':
            # 特征插值过采样
            additional_files = self.advanced_oversampling.feature_interpolation_oversampling(
                original_files, stats['folder'], output_folder,
                needed_samples, params, class_name, status_callback
            )
            sampled_files.extend(additional_files)
            
        elif oversample_method == 'adaptive':
            # 自适应过采样
            additional_files = self.advanced_oversampling.adaptive_oversampling(
                original_files, stats['folder'], output_folder,
                needed_samples, params, class_name, status_callback
            )
            sampled_files.extend(additional_files)
            
        elif oversample_method == 'smart':
            # 智能增强过采样
            additional_files = self.advanced_oversampling.smart_augmentation_oversampling(
                original_files, stats['folder'], output_folder,
                needed_samples, params, class_name, status_callback
            )
            sampled_files.extend(additional_files)
            
        return sampled_files
        
    def _undersample_class(self, class_name: str, stats: Dict, target_samples: int,
                          params: Dict, output_folder: str,
                          status_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """欠采样处理"""
        original_files = stats['files']
        original_count = len(original_files)
        
        if status_callback:
            status_callback(f"欠采样 {class_name}: {original_count} -> {target_samples}")
            
        # 选择采样方法
        undersample_method = params.get('undersample_method', 'random')
        
        if undersample_method == 'random':
            # 随机采样
            selected_files = random.sample(original_files, target_samples)
        elif undersample_method == 'cluster':
            # 真正的聚类采样
            selected_files = self.advanced_sampling.cluster_based_sampling(
                original_files, stats['folder'], target_samples, status_callback
            )
        elif undersample_method == 'diversity':
            # 多样性采样
            selected_files = self.advanced_sampling.diversity_based_sampling(
                original_files, stats['folder'], target_samples, status_callback
            )
        elif undersample_method == 'quality':
            # 质量采样
            selected_files = self.advanced_sampling.quality_based_sampling(
                original_files, stats['folder'], target_samples, status_callback
            )
        else:
            # 默认随机采样
            selected_files = random.sample(original_files, target_samples)
            
        # 处理选中的文件
        sampled_files = []
        for img_file in selected_files:
            source_path = os.path.join(stats['folder'], img_file)
            file_name = os.path.splitext(img_file)[0]
            target_path = os.path.join(output_folder, f"{file_name}.{params['format']}")
            
            # 预处理图片
            self.image_transformer.process_single_image(
                source_path, target_path, params['width'], params['height'],
                params['brightness_value'], params['contrast_value']
            )
            sampled_files.append(target_path)
            
        return sampled_files
        
    def _process_class_without_sampling(self, class_name: str, stats: Dict, params: Dict,
                                      output_folder: str, 
                                      status_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """无需采样，直接处理"""
        if status_callback:
            status_callback(f"直接处理 {class_name}: {stats['count']} 个样本")
            
        sampled_files = []
        for img_file in stats['files']:
            source_path = os.path.join(stats['folder'], img_file)
            file_name = os.path.splitext(img_file)[0]
            target_path = os.path.join(output_folder, f"{file_name}.{params['format']}")
            
            # 预处理图片
            self.image_transformer.process_single_image(
                source_path, target_path, params['width'], params['height'],
                params['brightness_value'], params['contrast_value']
            )
            sampled_files.append(target_path)
            
        return sampled_files
        
    def _generate_augmented_samples(self, original_files: List[str], source_folder: str,
                                  output_folder: str, needed_samples: int, params: Dict,
                                  class_name: str, status_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """生成增强样本"""
        additional_files = []
        
        # 检查是否使用高级随机增强
        use_enhanced_augmentation = params.get('use_enhanced_augmentation', True)
        
        # 计算每个原始文件需要生成多少个增强样本
        files_per_original = needed_samples // len(original_files)
        remaining_samples = needed_samples % len(original_files)
        
        for i, img_file in enumerate(original_files):
            source_path = os.path.join(source_folder, img_file)
            file_name = os.path.splitext(img_file)[0]
            
            # 为这个文件生成增强样本
            samples_for_this_file = files_per_original + (1 if i < remaining_samples else 0)
            
            for j in range(samples_for_this_file):
                aug_file_name = f"{file_name}_aug_{j+1}"
                aug_target_path = os.path.join(output_folder, f"{aug_file_name}.{params['format']}")
                
                # 先预处理原始图片到临时位置
                temp_path = os.path.join(output_folder, f"temp_{file_name}.{params['format']}")
                self.image_transformer.process_single_image(
                    source_path, temp_path, params['width'], params['height'],
                    params['brightness_value'], params['contrast_value']
                )
                
                # 选择增强方法
                if use_enhanced_augmentation:
                    # 使用增强版随机增强（更多连续随机参数）
                    try:
                        from PIL import Image
                        
                        # 读取图片
                        image = Image.open(temp_path)
                        
                        # 获取增强版随机增强
                        augmentation = self.augmentation_manager.get_enhanced_random_augmentation(params)
                        
                        # 应用增强
                        augmented = augmentation(image=image)['image']
                        
                        # 保存增强后的图片
                        augmented.save(aug_target_path)
                        
                        if status_callback and j % 10 == 0:
                            status_callback(f"高级增强进度 {class_name}: {j+1}/{samples_for_this_file}")
                            
                    except Exception as e:
                        # 如果高级增强失败，回退到标准增强
                        if status_callback:
                            status_callback(f"高级增强失败，使用标准增强: {str(e)}")
                        self.augmentation_manager.apply_single_augmentation_to_image(
                            temp_path, output_folder, aug_file_name, params['format'], params
                        )
                else:
                    # 使用标准随机增强
                    self.augmentation_manager.apply_single_augmentation_to_image(
                        temp_path, output_folder, aug_file_name, params['format'], params
                    )
                
                # 删除临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                additional_files.append(aug_target_path)
                
        return additional_files
        
    def _generate_duplicated_samples(self, original_files: List[str], source_folder: str,
                                   output_folder: str, needed_samples: int, params: Dict,
                                   class_name: str, status_callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """生成重复样本"""
        additional_files = []
        
        # 重复选择原始文件直到达到需要的数量
        for i in range(needed_samples):
            # 循环选择原始文件
            selected_file = original_files[i % len(original_files)]
            source_path = os.path.join(source_folder, selected_file)
            file_name = os.path.splitext(selected_file)[0]
            
            # 创建副本
            dup_file_name = f"{file_name}_dup_{i+1}"
            dup_target_path = os.path.join(output_folder, f"{dup_file_name}.{params['format']}")
            
            # 预处理图片
            self.image_transformer.process_single_image(
                source_path, dup_target_path, params['width'], params['height'],
                params['brightness_value'], params['contrast_value']
            )
            
            additional_files.append(dup_target_path)
            
        return additional_files
        
    def _cluster_based_sampling(self, files: List[str], target_samples: int) -> List[str]:
        """基于聚类的采样（简化版本）"""
        # 这里使用均匀分布采样来模拟聚类采样
        # 在实际应用中，可以基于图像特征进行聚类
        if len(files) <= target_samples:
            return files
            
        # 均匀选择样本
        indices = np.linspace(0, len(files) - 1, target_samples, dtype=int)
        return [files[i] for i in indices]
        
    def _copy_files_to_dataset(self, files: List[str], destination_folder: str):
        """复制文件到数据集文件夹"""
        for file_path in files:
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(destination_folder, file_name)
                shutil.copy2(file_path, dest_path)
                
    def get_sampling_strategy_description(self, strategy: str) -> str:
        """获取采样策略的描述"""
        descriptions = {
            'auto': '自动选择最适合的采样策略',
            'oversample': '过采样 - 增加少数类样本到最多类的数量',
            'undersample': '欠采样 - 减少多数类样本到最少类的数量',
            'median': '中位数采样 - 将所有类别调整到中位数样本量',
            'custom': '自定义 - 使用用户指定的目标样本数'
        }
        return descriptions.get(strategy, '未知策略')
        
    def calculate_imbalance_ratio(self, class_stats: Dict) -> float:
        """计算类别不平衡比率"""
        counts = [stats['count'] for stats in class_stats.values()]
        if not counts or min(counts) == 0:
            return float('inf')
        return max(counts) / min(counts) 
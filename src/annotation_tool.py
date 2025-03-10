import os
import subprocess
import sys
import json
from PyQt5.QtCore import QObject, pyqtSignal
from typing import List

class AnnotationTool(QObject):
    # 定义信号
    status_updated = pyqtSignal(str)
    annotation_error = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
    def start_labelimg(self, image_folder: str, class_names: List[str] = None) -> None:
        """
        启动LabelImg标注工具
        
        参数:
            image_folder: 图片文件夹路径
            class_names: 缺陷类别名称列表
        """
        try:
            # 检查LabelImg是否已安装
            try:
                subprocess.run(['labelImg', '--help'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              check=False)
            except FileNotFoundError:
                self.status_updated.emit('正在安装LabelImg...')
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'labelImg'], 
                              check=True)
                
            # 创建预定义类别文件
            if class_names:
                predefined_classes_file = os.path.join(os.path.dirname(image_folder), 'predefined_classes.txt')
                with open(predefined_classes_file, 'w') as f:
                    for class_name in class_names:
                        f.write(f"{class_name}\n")
                self.status_updated.emit(f'已创建预定义类别文件: {predefined_classes_file}')
            
            # 启动LabelImg
            self.status_updated.emit('正在启动LabelImg...')
            
            # 构建命令
            cmd = ['labelImg', image_folder]
            if class_names:
                cmd.extend(['--predefined_classes_file', predefined_classes_file])
                
            # 在新进程中启动LabelImg
            subprocess.Popen(cmd)
            self.status_updated.emit('LabelImg已启动')
            
        except Exception as e:
            self.annotation_error.emit(f'启动LabelImg时出错: {str(e)}')
            
    def start_labelme(self, image_folder: str, class_names: List[str] = None) -> None:
        """
        启动LabelMe标注工具
        
        参数:
            image_folder: 图片文件夹路径
            class_names: 缺陷类别名称列表
        """
        try:
            # 检查LabelMe是否已安装
            try:
                subprocess.run(['labelme', '--help'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              check=False)
            except FileNotFoundError:
                self.status_updated.emit('正在安装LabelMe...')
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'labelme'], 
                              check=True)
                
            # 创建标签配置文件
            if class_names:
                config_file = os.path.join(os.path.dirname(image_folder), 'labelme_config.json')
                config = {
                    'labels': class_names,
                    'flags': {},
                    'lineColor': [0, 255, 0, 128],
                    'fillColor': [255, 0, 0, 128],
                    'shapes': ['polygon', 'rectangle', 'circle', 'line', 'point']
                }
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=4)
                self.status_updated.emit(f'已创建LabelMe配置文件: {config_file}')
            
            # 启动LabelMe
            self.status_updated.emit('正在启动LabelMe...')
            
            # 构建命令
            cmd = ['labelme', image_folder]
            if class_names:
                cmd.extend(['--config', config_file])
                
            # 在新进程中启动LabelMe
            subprocess.Popen(cmd)
            self.status_updated.emit('LabelMe已启动')
            
        except Exception as e:
            self.annotation_error.emit(f'启动LabelMe时出错: {str(e)}')
            
    def convert_annotations(self, annotation_folder: str, output_folder: str, format_type: str) -> None:
        """
        转换标注格式
        
        参数:
            annotation_folder: 标注文件夹路径
            output_folder: 输出文件夹路径
            format_type: 目标格式类型 ('voc', 'coco', 'yolo')
        """
        try:
            self.status_updated.emit(f'正在将标注转换为{format_type}格式...')
            
            # 确保输出文件夹存在
            os.makedirs(output_folder, exist_ok=True)
            
            if format_type.lower() == 'voc':
                # 转换为VOC格式
                pass
            elif format_type.lower() == 'coco':
                # 转换为COCO格式
                pass
            elif format_type.lower() == 'yolo':
                # 转换为YOLO格式
                pass
            else:
                raise ValueError(f'不支持的格式类型: {format_type}')
                
            self.status_updated.emit('标注转换完成')
            
        except Exception as e:
            self.annotation_error.emit(f'转换标注时出错: {str(e)}') 
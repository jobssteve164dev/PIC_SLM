"""
集成了真正资源限制的训练组件
确保训练过程遵守资源限制，并提供优雅的中断机制
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, Callable
from ..utils.resource_limiter import (
    get_resource_limiter, ResourceLimitException, resource_limited_operation
)


class ResourceLimitedTrainer:
    """集成了资源限制的训练器"""
    
    def __init__(self, original_trainer):
        """
        初始化资源限制训练器
        
        Args:
            original_trainer: 原始训练器实例
        """
        self.original_trainer = original_trainer
        self.resource_limiter = get_resource_limiter()
        self.interrupted = False
        self.last_checkpoint_epoch = 0
        
    @resource_limited_operation("模型训练")
    def train_epoch(self, epoch: int, train_loader, model, optimizer, criterion, device):
        """训练一个epoch，集成资源检查"""
        if self.resource_limiter and self.resource_limiter.is_stop_requested():
            print(f"🛑 训练在第 {epoch} 个epoch被资源限制器中断")
            self.interrupted = True
            return None
            
        try:
            # 在每个epoch开始前检查资源
            if self.resource_limiter:
                self.resource_limiter.check_resource_before_operation(f"训练epoch {epoch}")
                
            # 执行原始的训练逻辑
            if hasattr(self.original_trainer, 'train_epoch'):
                return self.original_trainer.train_epoch(epoch, train_loader, model, optimizer, criterion, device)
            else:
                # 默认训练逻辑
                return self._default_train_epoch(epoch, train_loader, model, optimizer, criterion, device)
                
        except ResourceLimitException as e:
            print(f"❌ 训练因资源限制中断: {e}")
            self.interrupted = True
            self._emergency_save_checkpoint(epoch, model, optimizer)
            raise
        except Exception as e:
            print(f"❌ 训练出错: {e}")
            raise
    
    def _default_train_epoch(self, epoch: int, train_loader, model, optimizer, criterion, device):
        """默认的训练epoch实现"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 在每个batch前检查是否被中断
            if self.resource_limiter and self.resource_limiter.is_stop_requested():
                print(f"🛑 训练在第 {epoch} epoch的第 {batch_idx} batch被中断")
                self.interrupted = True
                break
                
            # 每隔一定间隔检查资源
            if batch_idx % 50 == 0 and self.resource_limiter:
                try:
                    self.resource_limiter.check_resource_before_operation(f"batch {batch_idx}")
                except ResourceLimitException as e:
                    print(f"⚠️ Batch {batch_idx} 资源检查失败: {e}")
                    # 不直接中断，而是尝试继续
                    if self.resource_limiter:
                        self.resource_limiter.emergency_cleanup()
            
            try:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # 定期注册临时文件（如果有的话）
                if batch_idx % 100 == 0 and self.resource_limiter:
                    self._register_temp_files()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"💾 CUDA内存不足，执行清理...")
                    torch.cuda.empty_cache()
                    if self.resource_limiter:
                        self.resource_limiter.emergency_cleanup()
                    continue
                else:
                    raise
                    
        if self.interrupted:
            return None
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    @resource_limited_operation("模型验证")
    def validate_model(self, val_loader, model, criterion, device):
        """验证模型，集成资源检查"""
        if self.resource_limiter and self.resource_limiter.is_stop_requested():
            print("🛑 验证被资源限制器中断")
            return None
            
        try:
            if hasattr(self.original_trainer, 'validate_model'):
                return self.original_trainer.validate_model(val_loader, model, criterion, device)
            else:
                return self._default_validate_model(val_loader, model, criterion, device)
                
        except ResourceLimitException as e:
            print(f"❌ 验证因资源限制中断: {e}")
            return None
            
    def _default_validate_model(self, val_loader, model, criterion, device):
        """默认的模型验证实现"""
        model.eval()
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                # 检查中断
                if self.resource_limiter and self.resource_limiter.is_stop_requested():
                    break
                    
                try:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
        
        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)
        
        return {
            'val_loss': val_loss,
            'val_accuracy': accuracy,
            'correct': correct,
            'total': len(val_loader.dataset)
        }
    
    def _register_temp_files(self):
        """注册训练过程中产生的临时文件"""
        if not self.resource_limiter:
            return
            
        try:
            # 注册常见的临时文件路径
            temp_patterns = [
                "/tmp/torch_*",
                "/tmp/python_*", 
                "*.tmp",
                "checkpoint_temp.pth",
                "model_temp.pth"
            ]
            
            import glob
            for pattern in temp_patterns:
                for file_path in glob.glob(pattern):
                    if os.path.isfile(file_path):
                        self.resource_limiter.register_temp_file(file_path)
                        
        except Exception as e:
            print(f"⚠️ 注册临时文件失败: {e}")
    
    def _emergency_save_checkpoint(self, epoch: int, model, optimizer):
        """紧急保存检查点"""
        try:
            checkpoint_path = f"emergency_checkpoint_epoch_{epoch}.pth"
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'interrupted': True,
                'timestamp': time.time()
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 紧急检查点已保存: {checkpoint_path}")
            
            # 注册检查点文件为临时文件
            if self.resource_limiter:
                self.resource_limiter.register_temp_file(checkpoint_path)
                
        except Exception as e:
            print(f"❌ 保存紧急检查点失败: {e}")
    
    def train_with_resource_limits(self, epochs: int, train_loader, val_loader, 
                                 model, optimizer, criterion, device, 
                                 save_callback: Optional[Callable] = None):
        """
        执行完整的训练流程，集成资源限制
        
        Args:
            epochs: 训练轮数
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            model: 模型
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
            save_callback: 保存回调函数
        """
        print(f"🚀 开始资源限制训练，共 {epochs} 个epoch")
        
        if self.resource_limiter:
            print(f"📊 资源限制状态: {self.resource_limiter.get_resource_status()}")
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            try:
                # 训练
                train_result = self.train_epoch(epoch, train_loader, model, optimizer, criterion, device)
                
                if self.interrupted or train_result is None:
                    print(f"⏹️ 训练在第 {epoch} 个epoch被中断")
                    break
                
                # 验证
                val_result = self.validate_model(val_loader, model, criterion, device)
                
                if self.interrupted or val_result is None:
                    print(f"⏹️ 验证在第 {epoch} 个epoch被中断")
                    break
                
                # 打印结果
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch}/{epochs} - "
                      f"Loss: {train_result['loss']:.4f}, "
                      f"Acc: {train_result['accuracy']:.2f}%, "
                      f"Val Loss: {val_result['val_loss']:.4f}, "
                      f"Val Acc: {val_result['val_accuracy']:.2f}%, "
                      f"Time: {epoch_time:.2f}s")
                
                # 保存检查点
                if save_callback:
                    save_callback(epoch, model, optimizer, train_result, val_result)
                
                self.last_checkpoint_epoch = epoch
                
                # 每5个epoch检查一次资源状态
                if epoch % 5 == 0 and self.resource_limiter:
                    status = self.resource_limiter.get_resource_status()
                    print(f"📊 资源状态: 内存 {status.get('memory_percent', 0):.1f}%, "
                          f"CPU {status.get('cpu_percent', 0):.1f}%")
                
            except ResourceLimitException as e:
                print(f"❌ 第 {epoch} epoch训练因资源限制中断: {e}")
                break
            except KeyboardInterrupt:
                print(f"⏹️ 用户手动中断训练 (Epoch {epoch})")
                self._emergency_save_checkpoint(epoch, model, optimizer)
                break
            except Exception as e:
                print(f"❌ 第 {epoch} epoch训练出错: {e}")
                self._emergency_save_checkpoint(epoch, model, optimizer)
                raise
        
        # 训练结束后的清理
        if self.resource_limiter:
            self.resource_limiter.emergency_cleanup()
            
        if self.interrupted:
            print(f"⚠️ 训练因资源限制在第 {self.last_checkpoint_epoch} epoch后中断")
        else:
            print(f"✅ 训练成功完成 {epochs} 个epoch")
    
    def create_resource_limited_thread(self, target, *args, **kwargs):
        """创建受资源限制的线程"""
        if self.resource_limiter:
            return self.resource_limiter.create_limited_thread(target, args, kwargs)
        else:
            import threading
            return threading.Thread(target=target, args=args, kwargs=kwargs)


def wrap_trainer_with_resource_limits(trainer_class):
    """装饰器：为训练器类添加资源限制支持"""
    class ResourceLimitedTrainerWrapper(trainer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.resource_limited_trainer = ResourceLimitedTrainer(self)
            
        def train_with_limits(self, *args, **kwargs):
            return self.resource_limited_trainer.train_with_resource_limits(*args, **kwargs)
            
    return ResourceLimitedTrainerWrapper


# 便捷函数
def enable_resource_limited_training(trainer_instance):
    """为现有训练器实例启用资源限制"""
    return ResourceLimitedTrainer(trainer_instance) 
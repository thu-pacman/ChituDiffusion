import torch
import csv
from pathlib import Path
from typing import Dict, Union, Optional

class MagLogger:
    """Static magnitude logger for tracking tensor norms by steps and layers"""
    
    # 存储格式: {name: [{step, layer, value}, ...]}
    _logs: Dict[str, list] = {}
    
    @staticmethod
    def log_magnitude(tensor: torch.Tensor, 
                     step: int,
                     layer: str,
                     name: str = "default",
                     norm_type: Union[str, float] = "fro") -> float:
        """记录张量的范数
        
        Args:
            tensor: 需要计算范数的张量
            step: 当前的去噪步骤
            layer: 层的名称或标识
            name: 该组记录的标识名称
            norm_type: 范数类型，可以是 "fro"、1、2、float('inf') 等
            
        Returns:
            计算得到的范数值
        """
        # 确保张量在CPU上且为float类型
        if tensor.is_cuda:
            tensor = tensor.detach().cpu()
        
        # 计算范数
        # if norm_type == "fro":
        #     magnitude = torch.norm(tensor, p="fro").item()
        # else:
        #     magnitude = torch.norm(tensor, p=norm_type).item()
        magnitude = tensor.item()
        # print(f"T{step}L{layer} | Mag = {magnitude}", flush=True)
            
        # 初始化该名称的记录列表（如果不存在）
        if name not in MagLogger._logs:
            MagLogger._logs[name] = []
            
        # 存储记录
        MagLogger._logs[name].append({
            'step': step,
            'layer': layer,
            'value': magnitude
        })
        
        return magnitude
    
    @staticmethod
    def save_to_csv(save_dir: str, name: Optional[str] = None, mode: str = 'a'):
        """将记录保存到CSV文件
        
        Args:
            save_dir: 保存目录的路径
            name: 要保存的记录名称，如果为None则保存所有记录
            mode: 文件打开模式，'a' 为追加，'w' 为覆写
        """
        # 确保保存目录存在
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 确定要保存的记录
        names_to_save = [name] if name is not None else MagLogger._logs.keys()
        
        # 遍历需要保存的记录
        for log_name in names_to_save:
            if log_name not in MagLogger._logs or not MagLogger._logs[log_name]:
                print(f"No logs found for name: {log_name}")
                continue
                
            # 构建CSV文件路径
            filepath = save_dir / f"{log_name}.csv"
            
            fieldnames = ['step', 'layer', 'value']
            file_exists = filepath.exists()
            
            with open(filepath, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # 如果是新文件或覆写模式，写入表头
                if mode == 'w' or not file_exists:
                    writer.writeheader()
                
                # 写入数据
                writer.writerows(MagLogger._logs[log_name])
            
            print(f"Saved logs to {filepath}")
    
    @staticmethod
    def clear(name: Optional[str] = None):
        """清除指定名称或所有的记录
        
        Args:
            name: 要清除的记录名称，如果为None则清除所有记录
        """
        if name is None:
            MagLogger._logs.clear()
        elif name in MagLogger._logs:
            MagLogger._logs[name].clear()
    
    @staticmethod
    def get_logs(name: str = "default") -> list:
        """获取指定名称的记录
        
        Args:
            name: 记录名称
            
        Returns:
            记录列表，如果不存在则返回空列表
        """
        return MagLogger._logs.get(name, [])
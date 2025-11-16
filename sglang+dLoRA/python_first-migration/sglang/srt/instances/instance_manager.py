"""
Simple Engine Manager for multi-instance LoRA serving
支持两种架构：
1. sglang: LoRA 平均分配到各实例
2. dlora: 动态调度（后续实现）
"""

import asyncio
import random
from typing import Dict, List, Optional
from enum import Enum


class InferenceArchitecture(Enum):
    SGLANG = "sglang"  # 平均分配
    DLORA = "dlora"    # 动态调度


class InstanceManager:
    """管理多个 LLM Engine 实例的请求分发"""
    
    def __init__(
        self,
        worker_urls: List[str],
        num_loras: int,
        inference_architecture: InferenceArchitecture = InferenceArchitecture.SGLANG,
    ):
        self.worker_urls = worker_urls
        self.num_workers = len(worker_urls)
        self.num_loras = num_loras
        self.inference_architecture = inference_architecture
        
        # LoRA 到 worker 的映射
        self.lora_to_worker: Dict[str, str] = {}
        self._init_lora_mapping()
    
    def _init_lora_mapping(self):
        """初始化 LoRA 到 worker 的映射"""
        if self.inference_architecture == InferenceArchitecture.SGLANG:
            # SGLang 模式：平均分配
            loras_per_worker = self.num_loras // self.num_workers
            remainder = self.num_loras % self.num_workers
            
            lora_idx = 0
            for worker_idx, worker_url in enumerate(self.worker_urls):
                # 计算这个 worker 应该负责的 LoRA 数量
                count = loras_per_worker + (1 if worker_idx < remainder else 0)
                for _ in range(count):
                    lora_name = f"lora{lora_idx}"
                    self.lora_to_worker[lora_name] = worker_url
                    lora_idx += 1
            
            print(f"[EngineManager] SGLang mode - LoRA mapping:")
            for lora, worker in self.lora_to_worker.items():
                print(f"  {lora} -> {worker}")
        
        elif self.inference_architecture == InferenceArchitecture.DLORA:
            # dLoRA 模式：暂时随机分配，后续实现动态调度
            print(f"[EngineManager] dLoRA mode - using random assignment (dynamic scheduling TBD)")
    
    def select_worker(self, lora_name: str) -> str:
        """根据 LoRA 名称选择对应的 worker URL"""
        if self.inference_architecture == InferenceArchitecture.SGLANG:
            # SGLang 模式：查表
            return self.lora_to_worker.get(lora_name, self.worker_urls[0])
        
        elif self.inference_architecture == InferenceArchitecture.DLORA:
            # dLoRA 模式：随机选择（临时实现）
            return random.choice(self.worker_urls)
        
        return self.worker_urls[0]
    
    def get_stats(self) -> Dict:
        """获取当前状态统计"""
        return {
            "inference_architecture": self.inference_architecture.value,
            "num_workers": self.num_workers,
            "num_loras": self.num_loras,
            "lora_mapping": self.lora_to_worker if self.inference_architecture == InferenceArchitecture.SGLANG else {},
        }
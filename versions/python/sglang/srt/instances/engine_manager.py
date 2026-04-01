"""
Unified Engine Manager for SGLang Multi-Instance Load Balancing.
Simple least-loaded instance selection without LoRA-aware routing.
"""

import asyncio
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class InstanceMetrics:
    """Track active request count for each instance."""
    instance_id: int
    active_requests: int = 0
    total_completed: int = 0
    
    def add_request(self):
        """Increment active request count."""
        self.active_requests += 1
    
    def complete_request(self):
        """Decrement active request count and increment completed."""
        if self.active_requests > 0:
            self.active_requests -= 1
        self.total_completed += 1


class UnifiedEngineManager:
    """
    Unified Engine Manager for SGLang.
    
    Simple load balancing: always route to the instance with fewest active requests.
    """

    def __init__(
        self,
        num_instances: int,
        num_loras: int,
        instance_urls: List[str],
        **kwargs  # Ignore extra arguments for compatibility
    ):
        """
        Initialize the UnifiedEngineManager.
        
        Args:
            num_instances: Number of SGLang instances
            num_loras: Number of LoRA adapters (not used, for compatibility)
            instance_urls: List of instance base URLs
        """
        self.num_instances = num_instances
        self.num_loras = num_loras
        self.instance_urls = [url.rstrip('/') for url in instance_urls]
        
        # Instance metrics tracking
        self.instance_metrics: List[InstanceMetrics] = [
            InstanceMetrics(instance_id=i) for i in range(num_instances)
        ]
        
        # Request tracking
        self.request_to_instance: Dict[str, int] = {}
        self.request_to_lora: Dict[str, int] = {}
        
        # Synchronization
        self.select_lock = asyncio.Lock()
        
        print(f"[UnifiedEngineManager] Initialized:")
        print(f"  - Instances: {num_instances}")
        print(f"  - Instance URLs: {self.instance_urls}")

    async def select_instance(
        self, 
        request_id: str, 
        lora_id: int
    ) -> Tuple[int, str]:
        """
        Select the instance with the least active requests.
        
        Args:
            request_id: Unique request identifier
            lora_id: LoRA adapter ID (ignored, for compatibility)
            
        Returns:
            Tuple of (instance_id, instance_url)
        """
        async with self.select_lock:
            # Find instance with minimum active requests
            min_load = float('inf')
            min_instance_id = 0
            
            for metrics in self.instance_metrics:
                if metrics.active_requests < min_load:
                    min_load = metrics.active_requests
                    min_instance_id = metrics.instance_id
            
            # Track request assignment
            self.request_to_instance[request_id] = min_instance_id
            self.request_to_lora[request_id] = lora_id
            
            # Increment active request count
            self.instance_metrics[min_instance_id].add_request()
            
            instance_url = self.instance_urls[min_instance_id]
            
            print(f"[UnifiedEngineManager] Request {request_id} (LoRA {lora_id}) -> "
                  f"Instance {min_instance_id} (active: {min_load + 1})")
            
            return min_instance_id, instance_url

    async def complete_request(self, request_id: str):
        """
        Mark a request as completed.
        
        Args:
            request_id: Unique request identifier
        """
        async with self.select_lock:
            if request_id not in self.request_to_instance:
                print(f"[UnifiedEngineManager] Warning: Request {request_id} not tracked")
                return
            
            instance_id = self.request_to_instance[request_id]
            self.instance_metrics[instance_id].complete_request()
            
            # Clean up tracking
            del self.request_to_instance[request_id]
            if request_id in self.request_to_lora:
                del self.request_to_lora[request_id]
            
            print(f"[UnifiedEngineManager] Request {request_id} completed on "
                  f"Instance {instance_id} (active: {self.instance_metrics[instance_id].active_requests})")

    def get_stats(self) -> Dict:
        """Get current manager statistics."""
        return {
            "num_instances": self.num_instances,
            "instance_metrics": [
                {
                    "instance_id": m.instance_id,
                    "active_requests": m.active_requests,
                    "total_completed": m.total_completed
                }
                for m in self.instance_metrics
            ],
            "instance_request_counts": [m.active_requests for m in self.instance_metrics],
        }

    def start_background_loop(self):
        """Compatibility method - does nothing in unified mode."""
        pass

    def stop_background_loop(self):
        """Compatibility method - does nothing in unified mode."""
        pass
    
    def reset_stats(self):
        """重置所有实例的统计数据。"""
        for m in self.instance_metrics:
            m.active_requests = 0
            m.total_completed = 0
        print("[UnifiedEngineManager] All instance stats have been reset.")


def create_engine_manager_unified(
    num_instances: int,
    num_loras: int,
    instance_urls: List[str],
    **kwargs
) -> UnifiedEngineManager:
    """
    Create a UnifiedEngineManager instance.
    
    Args:
        num_instances: Number of instances
        num_loras: Number of LoRA adapters
        instance_urls: List of instance URLs
        **kwargs: Additional arguments (ignored)
        
    Returns:
        UnifiedEngineManager instance
    """
    return UnifiedEngineManager(
        num_instances=num_instances,
        num_loras=num_loras,
        instance_urls=instance_urls,
        **kwargs
    )
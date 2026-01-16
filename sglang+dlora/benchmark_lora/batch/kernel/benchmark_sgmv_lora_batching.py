"""
LoRA SGMV Kernel 基础性能测试
测试同一 LoRA vs 不同 LoRA 的批处理性能差异（简化版本）
"""

import torch
import time
from typing import List, Dict
from dataclasses import dataclass

from sglang.srt.lora. backend.chunked_backend import ChunkedSgmvLoRABackend


@dataclass
class SimpleConfig:
    """简化的测试配置 - 减少内存使用"""
    batch_size: int = 16          # 减小 batch size
    seq_len: int = 128            # 减小序列长度  
    hidden_dim:  int = 1024       # 减小隐藏层维度
    lora_rank: int = 32          # 减小 LoRA rank
    num_loras: int = 4
    test_iterations: int = 10    # 减少测试次数
    device: str = "cuda:1"
    max_chunk_size: int = 32     # 减小 chunk size


class MinimalServerArgs:
    """最小化的 ServerArgs"""
    def __init__(self, max_lora_chunk_size: int = 32):
        self.max_lora_chunk_size = max_lora_chunk_size
        self.model_path = "/tmp/dummy"


class SimpleForwardMode:
    """简化的 ForwardMode"""
    def is_extend(self) -> bool:
        return True


class SimpleForwardBatch:
    """简化的 ForwardBatch"""
    def __init__(self, batch_size: int, seq_len: int):
        self.batch_size = batch_size
        self. extend_seq_lens_cpu = [seq_len] * batch_size
        self.forward_mode = SimpleForwardMode()
        self.extend_num_tokens = batch_size * seq_len


def create_lora_weights(config: SimpleConfig, device: torch.device):
    """创建 LoRA 权重"""
    weights_a = torch.randn(
        config.num_loras, 
        config.lora_rank, 
        config.hidden_dim,
        dtype=torch.float16, 
        device=device
    )
    
    weights_b = torch.randn(
        config.num_loras, 
        config.hidden_dim, 
        config.lora_rank,
        dtype=torch.float16, 
        device=device
    )
    
    return weights_a, weights_b


def create_input(config: SimpleConfig, device: torch.device):
    """创建输入数据"""
    total_tokens = config.batch_size * config.seq_len
    return torch.randn(
        total_tokens, 
        config. hidden_dim,
        dtype=torch. float16, 
        device=device
    )


def prepare_backend(backend, config: SimpleConfig, weight_indices: List[int]):
    """准备后端批次信息"""
    forward_batch = SimpleForwardBatch(config.batch_size, config.seq_len)
    lora_ranks = [config.lora_rank] * config.num_loras
    scalings = [1.0] * config.num_loras
    
    backend.prepare_lora_batch(
        forward_batch=forward_batch,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        batch_info=None
    )


def run_lora_forward(backend, weights_a, weights_b, x, config: SimpleConfig):
    """运行 LoRA 前向传播"""
    # LoRA A (shrink)
    lora_a_output = backend.run_lora_a_sgemm(x, weights_a)
    
    # LoRA B (expand)  
    output_offset = torch.tensor([0, config.hidden_dim], dtype=torch. int32, device=x. device)
    output = backend.run_lora_b_sgemm(
        x=lora_a_output,
        weights=weights_b, 
        output_offset=output_offset,
        base_output=None
    )
    
    return output


def benchmark_scenario(backend, weights_a, weights_b, config: SimpleConfig, 
                      weight_indices: List[int], scenario_name: str) -> float:
    """测试单个场景"""
    print(f"\n🔄 Testing {scenario_name}")
    print(f"   Weight indices: {weight_indices}")
    
    # 准备后端
    prepare_backend(backend, config, weight_indices)
    
    # 创建输入
    x = create_input(config, backend.device)
    
    # Warmup (只做1次)
    try:
        _ = run_lora_forward(backend, weights_a, weights_b, x, config)
        torch.cuda.synchronize()
        print("   ✓ Warmup completed")
    except Exception as e: 
        print(f"   ❌ Warmup failed: {e}")
        raise
    
    # 性能测试
    torch.cuda.synchronize()
    start_time = time. perf_counter()
    
    for _ in range(config. test_iterations):
        _ = run_lora_forward(backend, weights_a, weights_b, x, config)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # 计算平均时间
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / config. test_iterations
    
    print(f"   ⏱️  Average time: {avg_time_ms:.4f} ms")
    
    return avg_time_ms


def main():
    """主测试函数"""
    
    # 配置 - 使用更小的参数以避免内存问题
    config = SimpleConfig(
        batch_size=16,
        seq_len=128, 
        hidden_dim=1024,
        lora_rank=32,
        num_loras=4,
        test_iterations=10000,
        device="cuda:1",
        max_chunk_size=8
    )
    
    print("🚀 LoRA SGMV Kernel 基础性能测试")
    print("=" * 60)
    print(f"配置:")
    print(f"  Device: {config.device}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Sequence Length: {config. seq_len}") 
    print(f"  Hidden Dimension: {config.hidden_dim}")
    print(f"  LoRA Rank: {config.lora_rank}")
    print(f"  Test Iterations: {config.test_iterations}")
    
    try:
        # 初始化设备
        device = torch.device(config.device)
        torch.cuda.set_device(device)
        
        # 创建后端
        server_args = MinimalServerArgs(config.max_chunk_size)
        backend = ChunkedSgmvLoRABackend(
            max_loras_per_batch=config.num_loras,
            device=device,
            server_args=server_args
        )
        print("✓ Backend initialized")
        
        # 创建 LoRA 权重
        weights_a, weights_b = create_lora_weights(config, device)
        print(f"✓ LoRA weights created:  A{weights_a.shape}, B{weights_b.shape}")
        
        # 测试场景 1: 所有序列使用相同的 LoRA (lora_0)
        weight_indices_same = [0] * config.batch_size
        time_same = benchmark_scenario(
            backend, weights_a, weights_b, config, 
            weight_indices_same, "Same LoRA (all use lora_0)"
        )
        
        # 测试场景 2: 每个序列使用不同的 LoRA  
        weight_indices_diff = [i % config.num_loras for i in range(config.batch_size)]
        time_diff = benchmark_scenario(
            backend, weights_a, weights_b, config,
            weight_indices_diff, "Different LoRAs (mixed)"
        )
        
        # 结果对比
        print("\n" + "=" * 60)
        print("📊 测试结果")
        print("=" * 60)
        print(f"Same LoRA time:       {time_same:.4f} ms")
        print(f"Different LoRA time: {time_diff:.4f} ms")
        
        if time_same > 0:
            overhead_pct = (time_diff - time_same) / time_same * 100
            speedup = time_same / time_diff
            
            print(f"Overhead:             {overhead_pct:+.2f}%")
            if overhead_pct > 0: 
                print(f"💡 Different LoRAs are {overhead_pct:.1f}% slower than same LoRA")
                print(f"   (验证了 dLoRA 论文中的额外计算开销)")
            else:
                print(f"💡 Different LoRAs are {abs(overhead_pct):.1f}% faster (unexpected!)")
        
        print("=" * 60)
        print("🎉 测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败:  {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__": 
    exit_code = main()
    exit(exit_code)
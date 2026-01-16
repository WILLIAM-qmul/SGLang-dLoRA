"""
LoRA SGMV Kernel Performance Benchmark
测试同一 LoRA vs 不同 LoRA 的批处理性能差异
"""

import torch
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import sys
import os

from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend
from sglang.srt.lora. utils import LoRABatchInfo  
from sglang.srt.server_args import ServerArgs


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    batch_size: int = 16
    seq_len: int = 128
    hidden_dim: int = 4096
    lora_rank: int = 64
    num_loras: int = 4
    warmup_iterations: int = 10
    test_iterations: int = 100
    device: str = "cuda:0"
    max_chunk_size: int = 128


class MockServerArgs:
    """模拟 ServerArgs 以避免复杂的依赖"""
    
    def __init__(self, max_lora_chunk_size: int = 128):
        self.max_lora_chunk_size = max_lora_chunk_size
        
        # 添加其他可能需要的默认属性
        self. model_path = "/tmp/dummy_model"  # 必需参数
        self. dtype = "auto"
        self.tp_size = 1
        self.pp_size = 1
        self.device = "cuda"


class MockForwardMode:
    """模拟 ForwardMode 对象"""
    
    def __init__(self, is_extend:  bool = True):
        self._is_extend = is_extend
    
    def is_extend(self) -> bool:
        return self._is_extend


class MockForwardBatch:
    """模拟 ForwardBatch 对象"""
    
    def __init__(self, batch_size: int, seq_lens_cpu: List[int]):
        self.batch_size = batch_size
        self. extend_seq_lens_cpu = seq_lens_cpu
        self.forward_mode = MockForwardMode(is_extend=True)  # 设置为属性
        
        # 计算 extend_num_tokens（扩展模式下的 token 总数）
        self.extend_num_tokens = sum(seq_lens_cpu)


class LoRABenchmark:
    """LoRA 性能基准测试类"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 检查设备可用性
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if self.device.index is not None and self.device.index >= torch.cuda. device_count():
            raise RuntimeError(f"GPU {config.device} is not available")
        
        # 设置设备
        torch.cuda.set_device(self.device)
        
        # 初始化模拟的 server_args
        server_args = MockServerArgs(max_lora_chunk_size=config.max_chunk_size)
        
        # 初始化后端
        self. backend = ChunkedSgmvLoRABackend(
            max_loras_per_batch=config.num_loras,
            device=self.device,
            server_args=server_args
        )
        
        # 创建 LoRA 权重
        self. weights_a, self.weights_b = self._create_lora_weights()
        
        print("✓ Benchmark initialized")
        print(f"  Device: {config.device}")
        print(f"  GPU Memory: {torch. cuda.get_device_properties(self.device).total_memory / 1e9:.1f} GB")
        print(f"  Batch Size:  {config.batch_size}")
        print(f"  Sequence Length: {config.seq_len}")
        print(f"  Hidden Dimension: {config.hidden_dim}")
        print(f"  LoRA Rank: {config.lora_rank}")
        print(f"  Number of LoRAs: {config.num_loras}")
    
    def _create_lora_weights(self) -> Tuple[torch. Tensor, torch. Tensor]:
        """创建 LoRA 权重 (A 和 B 矩阵)"""
        # LoRA A: (num_loras, lora_rank, hidden_dim) - 用于 shrink 操作
        weights_a = torch. randn(
            self.config. num_loras,
            self.config.lora_rank,
            self. config.hidden_dim,
            dtype=torch.float16,
            device=self.device
        )
        
        # LoRA B:  (num_loras, hidden_dim, lora_rank) - 用于 expand 操作  
        weights_b = torch.randn(
            self. config.num_loras,
            self.config.hidden_dim,
            self.config.lora_rank,
            dtype=torch.float16,
            device=self.device
        )
        
        print(f"✓ Created LoRA weights:")
        print(f"  weights_a shape: {weights_a.shape}")
        print(f"  weights_b shape: {weights_b.shape}")
        
        return weights_a, weights_b
    
    def _create_input(self) -> torch.Tensor:
        """创建输入张量"""
        total_tokens = self. config.batch_size * self. config.seq_len
        x = torch.randn(
            total_tokens,
            self.config.hidden_dim,
            dtype=torch.float16,
            device=self.device
        )
        return x
    
    def _prepare_backend_batch_info(self, weight_indices: List[int]):
        """准备后端的批次信息"""
        # 创建序列长度列表（每个序列的长度）
        seq_lens_cpu = [self.config.seq_len] * self.config.batch_size
        
        # 创建 LoRA ranks 和 scalings
        lora_ranks = [self.config.lora_rank] * self.config.num_loras
        scalings = [1.0] * self.config.num_loras
        
        # 创建一个模拟的 forward_batch 对象
        forward_batch = MockForwardBatch(self.config.batch_size, seq_lens_cpu)
        
        # 使用后端的 prepare_lora_batch 方法
        self.backend.prepare_lora_batch(
            forward_batch=forward_batch,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            batch_info=None
        )
    
    def _run_forward(self, x: torch.Tensor) -> torch.Tensor:
        """运行前向传播：LoRA A + LoRA B"""
        # LoRA A (shrink): x @ A^T -> (total_tokens, lora_rank)
        lora_a_output = self.backend. run_lora_a_sgemm(x, self.weights_a)
        
        # LoRA B (expand): lora_a_output @ B^T -> (total_tokens, hidden_dim)
        output_offset = torch.tensor(
            [0, self.config.hidden_dim],
            dtype=torch.int32,
            device=self.device
        )
        
        output = self.backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.weights_b,
            output_offset=output_offset,
            base_output=None
        )
        
        return output
    
    def benchmark_scenario(
        self,
        weight_indices: List[int],
        scenario_name: str
    ) -> Dict[str, float]:
        """对特定场景进行基准测试
        
        Args: 
            weight_indices: 每个序列使用的 LoRA 索引
            scenario_name: 场景名称（用于打印）
        
        Returns:
            包含性能指标的字典
        """
        config = self.config
        
        # 准备批次信息
        self._prepare_backend_batch_info(weight_indices)
        
        # 创建输入
        x = self._create_input()
        
        print(f"  Running {scenario_name} scenario...")
        print(f"    Weight indices: {weight_indices[: min(8, len(weight_indices))]}{'...' if len(weight_indices) > 8 else ''}")
        
        # Warmup
        for i in range(config.warmup_iterations):
            try:
                output = self._run_forward(x)
                torch.cuda.synchronize()
            except Exception as e: 
                print(f"    Warmup iteration {i+1}/{config.warmup_iterations} failed: {e}")
                raise
        
        print(f"    Warmup completed ({config.warmup_iterations} iterations)")
        
        # 测试
        torch.cuda.synchronize()
        start_time = time. perf_counter()
        
        for i in range(config. test_iterations):
            try:
                output = self._run_forward(x)
            except Exception as e: 
                print(f"    Test iteration {i+1}/{config.test_iterations} failed: {e}")
                raise
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # 计算指标
        total_time = (end_time - start_time) * 1000  # ms
        avg_time = total_time / config.test_iterations  # ms per iteration
        throughput = (config.batch_size * config.test_iterations) / (total_time / 1000)  # requests/sec
        
        # 验证输出形状
        if output is not None:
            expected_shape = (config.batch_size * config.seq_len, config.hidden_dim)
            if output.shape != expected_shape:
                print(f"    ⚠️  Warning: Output shape {output.shape} != expected {expected_shape}")
        
        return {
            'scenario': scenario_name,
            'avg_time_ms': avg_time,
            'throughput_rps': throughput,
            'total_time_ms':  total_time
        }
    
    def run_comparison(self) -> Dict[str, any]:
        """运行同一 LoRA vs 不同 LoRA 的对比测试"""
        
        print("\n" + "=" * 80)
        print("Running Benchmark:  Same LoRA vs Different LoRAs")
        print("=" * 80)
        
        # 场景 1: 所有请求使用同一个 LoRA (lora_0)
        print("\n[Scenario 1] All requests use the SAME LoRA (lora_0)")
        weight_indices_same = [0] * self.config. batch_size
        result_same = self. benchmark_scenario(weight_indices_same, "Same LoRA")
        print(f"  Average time: {result_same['avg_time_ms']:.4f} ms")
        print(f"  Throughput:  {result_same['throughput_rps']:.2f} requests/sec")
        
        # 场景 2: 每个请求使用不同的 LoRA
        print("\n[Scenario 2] Each request uses a DIFFERENT LoRA")
        weight_indices_diff = [i % self.config. num_loras for i in range(self.config.batch_size)]
        result_diff = self. benchmark_scenario(weight_indices_diff, "Different LoRAs")
        print(f"  Average time: {result_diff['avg_time_ms']:.4f} ms")
        print(f"  Throughput:  {result_diff['throughput_rps']:.2f} requests/sec")
        
        # 计算差异
        if result_same['avg_time_ms'] > 0:
            speedup = result_same['avg_time_ms'] / result_diff['avg_time_ms']
            overhead_pct = (result_diff['avg_time_ms'] - result_same['avg_time_ms']) / result_same['avg_time_ms'] * 100
        else:
            speedup = 1.0
            overhead_pct = 0.0
        
        print("\n" + "=" * 80)
        print("Results Summary")
        print("=" * 80)
        print(f"Same LoRA time:            {result_same['avg_time_ms']:.4f} ms")
        print(f"Different LoRAs time:      {result_diff['avg_time_ms']:.4f} ms")
        if speedup > 1:
            print(f"Performance ratio:        {speedup:.3f}x (Same is faster)")
        else:
            print(f"Performance ratio:        {1/speedup:.3f}x (Different is faster)")
        print(f"Overhead:                  {overhead_pct:+.2f}%")
        print("=" * 80)
        
        return {
            'same_lora': result_same,
            'different_loras': result_diff,
            'speedup': speedup,
            'overhead_pct': overhead_pct
        }
    
    def run_batch_size_sweep(self, batch_sizes: List[int]) -> List[Dict]: 
        """扫描不同的 batch size"""
        
        print("\n" + "=" * 80)
        print("Batch Size Sweep")
        print("=" * 80)
        
        results = []
        original_batch_size = self.config.batch_size
        
        for bs in batch_sizes: 
            print(f"\nTesting batch size: {bs}")
            self.config.batch_size = bs
            
            # 场景 1: 同一 LoRA
            weight_indices_same = [0] * bs
            result_same = self.benchmark_scenario(weight_indices_same, f"Same-{bs}")
            
            # 场景 2: 不同 LoRA
            weight_indices_diff = [i % self.config.num_loras for i in range(bs)]
            result_diff = self.benchmark_scenario(weight_indices_diff, f"Diff-{bs}")
            
            if result_same['avg_time_ms'] > 0:
                overhead_pct = (result_diff['avg_time_ms'] - result_same['avg_time_ms']) / result_same['avg_time_ms'] * 100
            else: 
                overhead_pct = 0.0
            
            results.append({
                'batch_size':  bs,
                'same_lora_ms': result_same['avg_time_ms'],
                'diff_lora_ms': result_diff['avg_time_ms'],
                'overhead_pct': overhead_pct
            })
            
            print(f"  Same LoRA:         {result_same['avg_time_ms']:.4f} ms")
            print(f"  Different LoRA:   {result_diff['avg_time_ms']:.4f} ms")
            print(f"  Overhead:         {overhead_pct:+.2f}%")
        
        # 恢复原始 batch size
        self.config.batch_size = original_batch_size
        
        return results
    
    def plot_results(self, sweep_results: List[Dict], save_path: str = "lora_benchmark. png"):
        """绘制结果图表"""
        
        if not sweep_results: 
            print("⚠️  No results to plot")
            return
            
        batch_sizes = [r['batch_size'] for r in sweep_results]
        same_lora_times = [r['same_lora_ms'] for r in sweep_results]
        diff_lora_times = [r['diff_lora_ms'] for r in sweep_results]
        overheads = [r['overhead_pct'] for r in sweep_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 子图 1: 时间对比
        ax1.plot(batch_sizes, same_lora_times, 'o-', label='Same LoRA', linewidth=2, markersize=8, color='blue')
        ax1.plot(batch_sizes, diff_lora_times, 's-', label='Different LoRAs', linewidth=2, markersize=8, color='red')
        ax1.set_xlabel('Batch Size', fontsize=12)
        ax1.set_ylabel('Average Time (ms)', fontsize=12)
        ax1.set_title('LoRA SGMV Kernel Performance', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # 子图 2: 开销百分比
        ax2.plot(batch_sizes, overheads, 'D-', color='red', linewidth=2, markersize=8)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Batch Size', fontsize=12)
        ax2.set_ylabel('Overhead (%)', fontsize=12)
        ax2.set_title('Performance Overhead (Different vs Same)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        plt.tight_layout()
        
        try:
            plt. savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Plot saved to:  {save_path}")
        except Exception as e:
            print(f"\n⚠️  Failed to save plot: {e}")
        
        plt.close()


def main():
    """主函数"""
    
    # 配置
    config = BenchmarkConfig(
        batch_size=8,
        seq_len=128,
        hidden_dim=4096,
        lora_rank=64,
        num_loras=4,
        warmup_iterations=5,   # 减少 warmup 迭代次数
        test_iterations=50,    # 减少测试迭代次数以加快调试
        device="cuda:1",       # 根据你的实际情况调整
        max_chunk_size=128
    )
    
    try:
        # 创建基准测试
        benchmark = LoRABenchmark(config)
        
        # 运行主要对比测试
        comparison_results = benchmark. run_comparison()
        
        # 运行 batch size 扫描  
        batch_sizes = [4, 8, 16, 32]  # 减少测试的 batch size 数量
        sweep_results = benchmark. run_batch_size_sweep(batch_sizes)
        
        # 绘制结果
        benchmark.plot_results(sweep_results)
        
        print("\n" + "=" * 80)
        print("🎉 Benchmark Complete!")
        print("=" * 80)
        
        # 打印关键发现
        overhead = comparison_results['overhead_pct']
        if overhead > 5:
            print(f"💡 Key Finding: Different LoRAs have {overhead:.1f}% overhead compared to same LoRA")
        elif overhead > 0:
            print(f"💡 Key Finding:  Different LoRAs have minimal {overhead:.1f}% overhead")
        else:
            print(f"💡 Key Finding: Different LoRAs are actually {abs(overhead):.1f}% faster (unexpected!)")
        
    except Exception as e: 
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":  
    exit_code = main()
    sys.exit(exit_code)
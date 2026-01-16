"""
未合并推理性能对比测试：Triton vs Chunked SGMV 后端
测试场景：WX + BAX（完整的未合并推理流程）
对比维度：同一LoRA vs 不同LoRA，以及不同后端的性能
"""

import torch
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.lora.backend.base_backend import BaseLoRABackend


class BackendType(Enum):
    """后端类型枚举"""
    TRITON = "triton"
    CHUNKED_SGMV = "csgmv"


@dataclass
class UnmergedConfig: 
    """未合并推理测试配置"""
    batch_size: int = 16
    seq_len: int = 128
    hidden_dim: int = 1024
    lora_rank:  int = 32
    num_loras: int = 4
    test_iterations: int = 1000
    device: str = "cuda:1"
    max_chunk_size: int = 8


class MinimalServerArgs:
    def __init__(self, max_lora_chunk_size: int = 32):
        self.max_lora_chunk_size = max_lora_chunk_size
        self.model_path = "/tmp/dummy"


class SimpleForwardMode: 
    def is_extend(self) -> bool:
        return True


class SimpleForwardBatch:
    def __init__(self, batch_size: int, seq_len: int, device: torch.device):
        self.batch_size = batch_size
        self.extend_seq_lens_cpu = [seq_len] * batch_size
        self.forward_mode = SimpleForwardMode()
        self.extend_num_tokens = batch_size * seq_len
        # Triton backend 需要 extend_seq_lens tensor
        self.extend_seq_lens = torch.tensor(
            self.extend_seq_lens_cpu, 
            dtype=torch.int32, 
            device=device
        )


class UnmergedLoRAInferenceTester:
    """未合并LoRA推理测试器 - 支持多后端"""
    
    def __init__(self, config: UnmergedConfig, backend_type: BackendType):
        self.config = config
        self.backend_type = backend_type
        self.device = torch.device(config.device)
        torch.cuda.set_device(self.device)
        
        # 初始化后端
        self. backend = self._create_backend()
        
        # 创建权重
        self.base_weight = self._create_base_weight()
        self.lora_weights_a, self.lora_weights_b = self._create_lora_weights()
        
        print(f"✓ {backend_type.value. upper()} Backend Tester initialized")
        print(f"  Base weight shape: {self.base_weight. shape}")
        print(f"  LoRA A shape: {self.lora_weights_a.shape}")
        print(f"  LoRA B shape: {self.lora_weights_b.shape}")
    
    def _create_backend(self) -> BaseLoRABackend:
        """根据类型创建后端"""
        if self.backend_type == BackendType.TRITON:
            return TritonLoRABackend(
                max_loras_per_batch=self.config.num_loras,
                device=self.device
            )
        elif self.backend_type == BackendType.CHUNKED_SGMV:
            server_args = MinimalServerArgs(self.config.max_chunk_size)
            return ChunkedSgmvLoRABackend(
                max_loras_per_batch=self.config. num_loras,
                device=self.device,
                server_args=server_args
            )
        else:
            raise ValueError(f"Unknown backend type:  {self.backend_type}")
    
    def _create_base_weight(self) -> torch.Tensor:
        """创建基础模型权重 W"""
        return torch.randn(
            self.config.hidden_dim,
            self.config.hidden_dim,
            dtype=torch.float16,
            device=self.device
        )
    
    def _create_lora_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建 LoRA 权重 A 和 B"""
        weights_a = torch. randn(
            self.config.num_loras,
            self.config.lora_rank,
            self.config.hidden_dim,
            dtype=torch.float16,
            device=self.device
        )
        
        weights_b = torch.randn(
            self.config. num_loras,
            self.config.hidden_dim,
            self.config.lora_rank,
            dtype=torch.float16,
            device=self.device
        )
        
        return weights_a, weights_b
    
    def _create_input(self) -> torch.Tensor:
        """创建输入数据"""
        total_tokens = self.config. batch_size * self.config. seq_len
        return torch. randn(
            total_tokens,
            self.config.hidden_dim,
            dtype=torch.float16,
            device=self.device
        )
    
    def _prepare_backend(self, weight_indices: List[int]):
        """准备后端批次信息"""
        forward_batch = SimpleForwardBatch(
            self.config.batch_size, 
            self.config.seq_len,
            self.device
        )
        lora_ranks = [self.config.lora_rank] * self.config.num_loras
        scalings = [1.0] * self.config.num_loras
        
        self.backend.prepare_lora_batch(
            forward_batch=forward_batch,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            batch_info=None
        )
    
    def unmerged_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        完整的未合并推理：WX + BAX
        这是 SGLang 中实际使用的方式
        """
        # 1. 基础模型计算：WX
        base_output = torch.mm(x, self.base_weight. T)
        
        # 2. LoRA 计算：BAX (使用 backend kernel)
        # LoRA A (shrink): x @ A^T -> (total_tokens, lora_rank)
        lora_a_output = self.backend.run_lora_a_sgemm(x, self.lora_weights_a)
        
        # LoRA B (expand): lora_a_output @ B^T -> (total_tokens, hidden_dim)
        if self.backend_type == BackendType.TRITON:
            # Triton backend 不需要 output_offset
            lora_output = self.backend.run_lora_b_sgemm(
                x=lora_a_output,
                weights=self.lora_weights_b,
                base_output=base_output
            )
        else:
            # Chunked SGMV backend 需要 output_offset
            output_offset = torch.tensor(
                [0, self.config. hidden_dim], 
                dtype=torch.int32, 
                device=x.device
            )
            lora_output = self.backend. run_lora_b_sgemm(
                x=lora_a_output,
                weights=self.lora_weights_b,
                output_offset=output_offset,
                base_output=base_output
            )
        
        return lora_output
    
    def benchmark_unmerged_inference(
        self, 
        weight_indices: List[int], 
        scenario_name: str
    ) -> Dict[str, float]:
        """测试未合并推理性能"""
        print(f"\n🔄 Testing {scenario_name}")
        print(f"   Backend: {self.backend_type. value. upper()}")
        print(f"   Weight indices: {weight_indices[:8]}{'...' if len(weight_indices) > 8 else ''}")
        
        # 准备后端
        self._prepare_backend(weight_indices)
        
        # 创建输入
        x = self._create_input()
        
        # Warmup
        try:
            for _ in range(10):  # 多次warmup确保稳定
                _ = self.unmerged_inference(x)
            torch.cuda.synchronize()
            print("   ✓ Warmup completed")
        except Exception as e:
            print(f"   ❌ Warmup failed: {e}")
            raise
        
        # 性能测试
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(self.config. test_iterations):
            _ = self.unmerged_inference(x)
        
        torch. cuda.synchronize()
        end_time = time.perf_counter()
        
        # 计算指标
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / self.config.test_iterations
        avg_time_us = avg_time_ms * 1000
        throughput = (self.config.batch_size * self.config.test_iterations) / (total_time_ms / 1000)
        
        print(f"   ⏱️  Average time: {avg_time_us:.1f} us")
        print(f"   🚀 Throughput: {throughput:.1f} requests/sec")
        
        return {
            'scenario': scenario_name,
            'backend': self.backend_type.value,
            'avg_time_us': avg_time_us,
            'throughput_rps': throughput,
            'weight_indices': weight_indices. copy()
        }


class MultiBackendComparison:
    """多后端对比测试管理器"""
    
    def __init__(self, config: UnmergedConfig):
        self.config = config
        self.testers = {
            BackendType.TRITON: UnmergedLoRAInferenceTester(config, BackendType.TRITON),
            BackendType.CHUNKED_SGMV: UnmergedLoRAInferenceTester(config, BackendType.CHUNKED_SGMV)
        }
        self.results = {}
    
    def run_backend_comparison(self, backend_type: BackendType) -> Dict: 
        """运行单个后端的对比测试"""
        print(f"\n{'='*70}")
        print(f"🔬 测试后端: {backend_type.value.upper()}")
        print(f"{'='*70}")
        
        tester = self.testers[backend_type]
        
        # 场景 1: 所有请求使用同一个 LoRA
        print(f"\n📊 场景 1: 所有 {self.config.batch_size} 个请求都使用 LoRA_0")
        weight_indices_same = [0] * self.config.batch_size
        result_same = tester.benchmark_unmerged_inference(
            weight_indices_same, 
            f"Same LoRA (all lora_0)"
        )
        
        # 场景 2: 每个请求使用不同的 LoRA
        print(f"\n📊 场景 2: 每个请求使用不同的 LoRA")
        weight_indices_diff = [
            i % self.config.num_loras 
            for i in range(self.config.batch_size)
        ]
        result_diff = tester.benchmark_unmerged_inference(
            weight_indices_diff, 
            f"Different LoRAs (mixed)"
        )
        
        # 计算性能差异
        overhead_pct = 0
        throughput_loss = 0
        if result_same['avg_time_us'] > 0:
            overhead_pct = (
                (result_diff['avg_time_us'] - result_same['avg_time_us']) 
                / result_same['avg_time_us'] * 100
            )
            throughput_loss = (
                (result_same['throughput_rps'] - result_diff['throughput_rps']) 
                / result_same['throughput_rps'] * 100
            )
        
        return {
            'backend': backend_type.value,
            'same_lora':  result_same,
            'different_lora': result_diff,
            'overhead_pct':  overhead_pct,
            'throughput_loss_pct': throughput_loss
        }
    
    def run_all_backends_comparison(self):
        """运行所有后端的对比测试"""
        print("\n" + "🚀 多后端未合并推理性能对比测试")
        print("=" * 70)
        print("测试内容：完整的未合并推理 WX + BAX")
        print("测试后端：Triton vs Chunked SGMV")
        print("对比场景：同一LoRA vs 不同LoRA")
        print("=" * 70)
        
        # 测试每个后端
        for backend_type in [BackendType.TRITON, BackendType.CHUNKED_SGMV]:
            self.results[backend_type] = self.run_backend_comparison(backend_type)
        
        # 综合分析
        self._print_comprehensive_analysis()
    
    def _print_comprehensive_analysis(self):
        """打印综合分析结果"""
        print("\n" + "=" * 70)
        print("📊 综合性能分析")
        print("=" * 70)
        
        # 1. 各后端的基本性能指标
        print("\n1️⃣ 各后端性能指标对比")
        print("-" * 70)
        
        for backend_type in [BackendType. TRITON, BackendType. CHUNKED_SGMV]: 
            result = self.results[backend_type]
            print(f"\n【{backend_type.value.upper()} Backend】")
            
            print(f"  同一LoRA场景:")
            print(f"    平均时间:   {result['same_lora']['avg_time_us']:.1f} us")
            print(f"    吞吐量:    {result['same_lora']['throughput_rps']:.1f} req/s")
            
            print(f"  不同LoRA场景:")
            print(f"    平均时间:  {result['different_lora']['avg_time_us']:.1f} us")
            print(f"    吞吐量:    {result['different_lora']['throughput_rps']:.1f} req/s")
            
            print(f"  性能差异:")
            print(f"    时间开销:   {result['overhead_pct']:+.2f}%")
            print(f"    吞吐损失:  {result['throughput_loss_pct']:+.2f}%")
        
        # 2. 后端间的性能对比
        print("\n2️⃣ 后端间性能对比")
        print("-" * 70)
        
        triton_result = self.results[BackendType.TRITON]
        chunked_result = self.results[BackendType.CHUNKED_SGMV]
        
        # 同一LoRA场景对比
        same_lora_speedup = (
            chunked_result['same_lora']['avg_time_us'] 
            / triton_result['same_lora']['avg_time_us']
        )
        print(f"\n同一LoRA场景:")
        print(f"  Triton:         {triton_result['same_lora']['avg_time_us']:.1f} us")
        print(f"  Chunked SGMV:   {chunked_result['same_lora']['avg_time_us']:.1f} us")
        print(f"  加速比:         {same_lora_speedup:.2f}x " 
              f"({'Triton更快' if same_lora_speedup > 1 else 'Chunked更快'})")
        
        # 不同LoRA场景对比
        diff_lora_speedup = (
            chunked_result['different_lora']['avg_time_us'] 
            / triton_result['different_lora']['avg_time_us']
        )
        print(f"\n不同LoRA场景:")
        print(f"  Triton:         {triton_result['different_lora']['avg_time_us']:.1f} us")
        print(f"  Chunked SGMV:  {chunked_result['different_lora']['avg_time_us']:.1f} us")
        print(f"  加速比:        {diff_lora_speedup:.2f}x "
              f"({'Triton更快' if diff_lora_speedup > 1 else 'Chunked更快'})")
        
        # 3. 关键发现
        print("\n3️⃣ 关键发现")
        print("-" * 70)
        
        triton_overhead = triton_result['overhead_pct']
        chunked_overhead = chunked_result['overhead_pct']
        
        print(f"\n📌 不同LoRA带来的性能开销:")
        print(f"  Triton:        {triton_overhead:+.2f}%")
        print(f"  Chunked SGMV:  {chunked_overhead:+.2f}%")
        
        if abs(triton_overhead - chunked_overhead) > 5:
            better_backend = "Triton" if triton_overhead < chunked_overhead else "Chunked SGMV"
            worse_backend = "Chunked SGMV" if better_backend == "Triton" else "Triton"
            print(f"\n  💡 {better_backend} 在处理不同LoRA时的性能退化更小")
            print(f"     {better_backend} 对 LoRA 多样性的处理更优")
        else:
            print(f"\n  ✅ 两种后端在处理不同LoRA时的性能退化相近")
        
        # 4. 使用建议
        print("\n4️⃣ 使用建议")
        print("-" * 70)
        
        if same_lora_speedup > 1.2:
            print(f"• 对于同一LoRA批次：优先选择 Triton (快 {(same_lora_speedup-1)*100:.1f}%)")
        elif same_lora_speedup < 0.8:
            print(f"• 对于同一LoRA批次：优先选择 Chunked SGMV (快 {(1/same_lora_speedup-1)*100:.1f}%)")
        else:
            print(f"• 对于同一LoRA批次：两种后端性能相近，可任选")
        
        if diff_lora_speedup > 1.2:
            print(f"• 对于混合LoRA批次：优先选择 Triton (快 {(diff_lora_speedup-1)*100:.1f}%)")
        elif diff_lora_speedup < 0.8:
            print(f"• 对于混合LoRA批次：优先选择 Chunked SGMV (快 {(1/diff_lora_speedup-1)*100:.1f}%)")
        else:
            print(f"• 对于混合LoRA批次：两种后端性能相近，可任选")
        
        if triton_overhead > 10 or chunked_overhead > 10:
            print(f"• ⚠️  不同LoRA带来的性能开销较大，建议使用dLoRA等优化策略")
        
        # 5. 测试配置
        print("\n5️⃣ 测试配置")
        print("-" * 70)
        print(f"  批次大小:       {self.config.batch_size}")
        print(f"  序列长度:      {self.config. seq_len}")
        print(f"  总token数:     {self.config.batch_size * self.config.seq_len}")
        print(f"  LoRA数量:      {self.config.num_loras}")
        print(f"  LoRA维度:      {self.config. lora_rank}")
        print(f"  隐藏层维度:    {self.config.hidden_dim}")
        print(f"  测试迭代次数:   {self.config.test_iterations}")
        
        print("=" * 70)
    
    def run_batch_size_analysis(self):
        """分析不同batch size下各后端的性能"""
        print("\n🔬 Batch Size 影响分析 (多后端)")
        print("=" * 70)
        
        original_batch_size = self.config.batch_size
        batch_sizes = [4, 8, 16, 32]
        
        all_results = {
            BackendType.TRITON: [],
            BackendType.CHUNKED_SGMV: []
        }
        
        for bs in batch_sizes:
            print(f"\n{'='*60}")
            print(f"测试 Batch Size: {bs}")
            print(f"{'='*60}")
            
            self.config.batch_size = bs
            
            # 为每个后端测试
            for backend_type in [BackendType.TRITON, BackendType.CHUNKED_SGMV]:
                print(f"\n{backend_type.value.upper()} Backend:")
                tester = self.testers[backend_type]
                
                # 同一LoRA
                weight_indices_same = [0] * bs
                result_same = tester.benchmark_unmerged_inference(
                    weight_indices_same, 
                    f"Same-BS{bs}"
                )
                
                # 不同LoRA
                weight_indices_diff = [i % self.config.num_loras for i in range(bs)]
                result_diff = tester.benchmark_unmerged_inference(
                    weight_indices_diff, 
                    f"Diff-BS{bs}"
                )
                
                overhead = (
                    (result_diff['avg_time_us'] - result_same['avg_time_us']) 
                    / result_same['avg_time_us'] * 100
                )
                
                all_results[backend_type].append({
                    'batch_size': bs,
                    'same_time_us': result_same['avg_time_us'],
                    'diff_time_us': result_diff['avg_time_us'],
                    'overhead_pct':  overhead
                })
        
        # 恢复原始配置
        self.config.batch_size = original_batch_size
        
        # 打印分析总结
        print(f"\n{'='*70}")
        print("📊 Batch Size 分析总结")
        print(f"{'='*70}")
        
        print(f"\n{'Batch Size':<12} {'Backend':<12} {'Same LoRA':<15} {'Diff LoRA':<15} {'Overhead': <10}")
        print("-" * 70)
        
        for bs in batch_sizes:
            for backend_type in [BackendType.TRITON, BackendType.CHUNKED_SGMV]:
                result = next(
                    r for r in all_results[backend_type] 
                    if r['batch_size'] == bs
                )
                print(f"{bs:<12} {backend_type.value. upper():<12} "
                      f"{result['same_time_us']:<15.1f} "
                      f"{result['diff_time_us']:<15.1f} "
                      f"{result['overhead_pct']: >+7.1f}%")
        
        print("=" * 70)
        
        return all_results


def main():
    """主测试函数"""
    
    # 配置
    config = UnmergedConfig(
        batch_size=8,
        seq_len=512,
        hidden_dim=4096,
        lora_rank=64,
        num_loras=4,
        test_iterations=100000,
        device="cuda:1",
        max_chunk_size=16
    )
    
    print("🔬 SGLang 多后端未合并推理性能测试")
    print("=" * 70)
    print("目标：对比 Triton 和 Chunked SGMV 两种后端的性能")
    print("场景：同一LoRA vs 不同LoRA")
    print("方法：完整的未合并推理 (WX + BAX)")
    print("=" * 70)
    
    try:
        # 创建多后端对比管理器
        comparison = MultiBackendComparison(config)
        
        # 运行主要对比测试
        comparison.run_all_backends_comparison()
        
        # # 运行batch size分析
        # print("\n" + "="*70)
        # print("进行 Batch Size 影响分析...")
        # print("="*70)
        # batch_analysis = comparison.run_batch_size_analysis()
        
        print("\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__": 
    exit_code = main()
    exit(exit_code)
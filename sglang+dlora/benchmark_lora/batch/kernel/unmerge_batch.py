"""
未合并推理性能对比测试：同一LoRA vs 不同LoRA
测试场景：WX + BAX（完整的未合并推理流程）
"""

import torch
import time
from typing import List, Dict
from dataclasses import dataclass

from sglang.srt. lora. backend. chunked_backend import ChunkedSgmvLoRABackend
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.lora.backend.base_backend import BaseLoRABackend


@dataclass
class UnmergedConfig:
    """未合并推理测试配置"""
    batch_size: int = 16
    seq_len: int = 128
    hidden_dim: int = 1024
    lora_rank: int = 32
    num_loras: int = 4
    test_iterations: int = 1000
    device: str = "cuda:1"
    max_chunk_size:  int = 8


class MinimalServerArgs: 
    def __init__(self, max_lora_chunk_size: int = 32):
        self.max_lora_chunk_size = max_lora_chunk_size
        self.model_path = "/tmp/dummy"


class SimpleForwardMode:
    def is_extend(self) -> bool:
        return True


class SimpleForwardBatch:
    def __init__(self, batch_size: int, seq_len: int):
        self.batch_size = batch_size
        self.extend_seq_lens_cpu = [seq_len] * batch_size
        self.forward_mode = SimpleForwardMode()
        self.extend_num_tokens = batch_size * seq_len


class UnmergedLoRAInferenceTester:
    """未合并LoRA推理测试器"""
    
    def __init__(self, config:  UnmergedConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.cuda.set_device(self.device)
        
        # 初始化后端
        server_args = MinimalServerArgs(config. max_chunk_size)
        self.backend = ChunkedSgmvLoRABackend(
            max_loras_per_batch=config.num_loras,
            device=self.device,
            server_args=server_args
        )
        # self.backend = TritonLoRABackend(
        #     max_loras_per_batch=self.config.num_loras,
        #     device=self.device
        # )
        
        # 创建权重
        self. base_weight = self._create_base_weight()
        self.lora_weights_a, self.lora_weights_b = self._create_lora_weights()
        
        print("✓ Unmerged LoRA Inference Tester initialized")
        print(f"  Base weight shape: {self.base_weight.shape}")
        print(f"  LoRA A shape: {self.lora_weights_a.shape}")
        print(f"  LoRA B shape: {self.lora_weights_b.shape}")
    
    def _create_base_weight(self) -> torch.Tensor:
        """创建基础模型权重 W"""
        return torch.randn(
            self.config. hidden_dim,
            self.config. hidden_dim,
            dtype=torch.float16,
            device=self.device
        )
    
    def _create_lora_weights(self) -> tuple:
        """创建 LoRA 权重 A 和 B"""
        weights_a = torch. randn(
            self.config.num_loras,
            self.config. lora_rank,
            self.config. hidden_dim,
            dtype=torch. float16,
            device=self.device
        )
        
        weights_b = torch.randn(
            self.config.num_loras,
            self. config.hidden_dim,
            self.config.lora_rank,
            dtype=torch.float16,
            device=self.device
        )
        
        return weights_a, weights_b
    
    def _create_input(self) -> torch.Tensor:
        """创建输入数据"""
        total_tokens = self.config. batch_size * self.config.seq_len
        return torch.randn(
            total_tokens,
            self.config.hidden_dim,
            dtype=torch.float16,
            device=self.device
        )
    
    def _prepare_backend(self, weight_indices: List[int]):
        """准备后端批次信息"""
        forward_batch = SimpleForwardBatch(self.config.batch_size, self.config.seq_len)
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
        base_output = torch.mm(x, self. base_weight. T)  # (total_tokens, hidden_dim)
        
        # 2. LoRA 计算：BAX (使用 SGMV kernel)
        # LoRA A (shrink): x @ A^T -> (total_tokens, lora_rank)
        lora_a_output = self.backend.run_lora_a_sgemm(x, self.lora_weights_a)
        
        # LoRA B (expand): lora_a_output @ B^T -> (total_tokens, hidden_dim)
        output_offset = torch.tensor([0, self.config.hidden_dim], dtype=torch.int32, device=x.device)
        lora_output = self.backend. run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.lora_weights_b,
            output_offset=output_offset,
            base_output=base_output  # 会自动加到 base_output 上：WX + BAX
        )
        
        return lora_output
    
    def benchmark_unmerged_inference(self, weight_indices: List[int], scenario_name: str) -> Dict[str, float]:
        """测试未合并推理性能"""
        print(f"\n🔄 Testing {scenario_name}")
        print(f"   Weight indices: {weight_indices[: 8]}{'...' if len(weight_indices) > 8 else ''}")
        
        # 准备后端
        self._prepare_backend(weight_indices)
        
        # 创建输入
        x = self._create_input()
        
        # Warmup
        try:
            _ = self.unmerged_inference(x)
            torch.cuda.synchronize()
            print("   ✓ Warmup completed")
        except Exception as e:
            print(f"   ❌ Warmup failed: {e}")
            raise
        
        # 性能测试
        torch.cuda.synchronize()
        start_time = time. perf_counter()
        
        for _ in range(self. config.test_iterations):
            _ = self.unmerged_inference(x)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # 计算指标
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / self.config. test_iterations
        avg_time_us = avg_time_ms * 1000
        throughput = (self.config.batch_size * self. config.test_iterations) / (total_time_ms / 1000)
        
        print(f"   ⏱️  Average time: {avg_time_us:.1f} us")
        print(f"   🚀 Throughput: {throughput:.1f} requests/sec")
        
        return {
            'scenario': scenario_name,
            'avg_time_us': avg_time_us,
            'throughput_rps': throughput,
            'weight_indices': weight_indices.copy()
        }
    
    def run_comparison_test(self):
        """运行对比测试：同一LoRA vs 不同LoRA"""
        print("\n" + "🚀 未合并推理性能对比测试")
        print("=" * 70)
        print("测试内容：完整的未合并推理 WX + BAX")
        print("对比场景：同一LoRA vs 不同LoRA 在 SGMV kernel 中的性能")
        print("=" * 70)
        
        # 场景 1: 所有请求使用同一个 LoRA (lora_0)
        print(f"\n📊 场景 1: 所有 {self.config.batch_size} 个请求都使用 LoRA_0")
        weight_indices_same = [0] * self.config.batch_size
        result_same = self. benchmark_unmerged_inference(weight_indices_same, "Same LoRA (all lora_0)")
        
        # 场景 2: 每个请求使用不同的 LoRA
        print(f"\n📊 场景 2: 每个请求使用不同的 LoRA (循环使用 LoRA_0 到 LoRA_{self.config.num_loras-1})")
        weight_indices_diff = [i % self. config.num_loras for i in range(self.config. batch_size)]
        result_diff = self.benchmark_unmerged_inference(weight_indices_diff, "Different LoRAs (mixed)")
        
        # 详细分析
        print("\n" + "=" * 70)
        print("📈 性能分析结果")
        print("=" * 70)
        
        # 基本指标
        print(f"同一LoRA (lora_0):")
        print(f"  平均时间:      {result_same['avg_time_us']:.1f} ms")
        print(f"  吞吐量:        {result_same['throughput_rps']:.1f} requests/sec")
        
        print(f"\n不同LoRA (混合):")
        print(f"  平均时间:      {result_diff['avg_time_us']:.1f} ms") 
        print(f"  吞吐量:       {result_diff['throughput_rps']:.1f} requests/sec")
        
        # 性能差异分析
        if result_same['avg_time_us'] > 0:
            overhead_pct = (result_diff['avg_time_us'] - result_same['avg_time_us']) / result_same['avg_time_us'] * 100
            throughput_loss = (result_same['throughput_rps'] - result_diff['throughput_rps']) / result_same['throughput_rps'] * 100
            
            print(f"\n💡 关键发现:")
            print(f"  时间开销:     {overhead_pct:+.2f}%")
            print(f"  吞吐量损失:   {throughput_loss:+.2f}%")
            
            if overhead_pct > 5:
                print(f"  🔍 不同LoRA比同一LoRA慢 {overhead_pct:.1f}%")
                print(f"     这验证了dLoRA论文中关于SGMV kernel性能差异的观点")
                print(f"     原因：不同LoRA需要访问不同的权重，缓存局部性较差")
            elif overhead_pct > 1:
                print(f"  ✅ 性能差异较小 ({overhead_pct:.1f}%)，SGMV kernel优化良好")
            else:
                print(f"  🤔 性能差异很小，可能受到其他因素影响")
        
        # 详细的权重索引信息
        print(f"\n🔍 详细信息:")
        print(f"  批次大小:     {self.config. batch_size}")
        print(f"  序列长度:     {self.config. seq_len}")
        print(f"  总token数:    {self.config. batch_size * self. config.seq_len}")
        print(f"  LoRA数量:      {self.config.num_loras}")
        print(f"  LoRA维度:     {self.config.lora_rank}")
        
        unique_loras_diff = len(set(weight_indices_diff))
        print(f"  场景1使用LoRA: 1个 (lora_0)")
        print(f"  场景2使用LoRA:  {unique_loras_diff}个 (lora_0 到 lora_{unique_loras_diff-1})")
        
        print("=" * 70)
        
        return {
            'same_lora': result_same,
            'different_lora': result_diff,
            'overhead_pct': overhead_pct if result_same['avg_time_us'] > 0 else 0,
            'throughput_loss_pct': throughput_loss if result_same['avg_time_us'] > 0 else 0
        }
    
    def run_batch_size_analysis(self):
        """分析不同batch size下的性能差异"""
        print("\n🔬 Batch Size 影响分析")
        print("=" * 50)
        
        original_batch_size = self.config.batch_size
        batch_sizes = [4, 8, 16, 32, 64]
        results = []
        
        for bs in batch_sizes: 
            if bs > 64:  # 避免内存超限
                continue
                
            print(f"\n测试 Batch Size: {bs}")
            self.config.batch_size = bs
            
            # 同一LoRA
            weight_indices_same = [0] * bs
            result_same = self.benchmark_unmerged_inference(weight_indices_same, f"Same-BS{bs}")
            
            # 不同LoRA  
            weight_indices_diff = [i % self.config.num_loras for i in range(bs)]
            result_diff = self.benchmark_unmerged_inference(weight_indices_diff, f"Diff-BS{bs}")

            overhead = (result_diff['avg_time_us'] - result_same['avg_time_us']) / result_same['avg_time_us'] * 100

            results.append({
                'batch_size': bs,
                'same_time': result_same['avg_time_us'],
                'diff_time': result_diff['avg_time_us'],
                'overhead_pct': overhead
            })

            print(f"  同一LoRA:  {result_same['avg_time_us']:. 1f}ms")
            print(f"  不同LoRA: {result_diff['avg_time_us']:.1f}ms")
            print(f"  开销: {overhead: +.1f}%")
        
        # 恢复原始配置
        self. config.batch_size = original_batch_size
        
        print(f"\n📊 Batch Size 分析总结:")
        for r in results:
            print(f"  BS={r['batch_size']:2d}: 开销={r['overhead_pct']: +5.1f}%")
        
        return results


def main():
    """主测试函数"""
    
    # 配置
    config = UnmergedConfig(
        batch_size=8,
        seq_len=512,
        hidden_dim=4096,
        lora_rank=64,
        num_loras=8,
        test_iterations=100000,  # 增加迭代次数获得更稳定的结果
        device="cuda:1",
        max_chunk_size=16
    )
    
    print("🔬 SGLang 未合并推理性能测试")
    print("=" * 70)
    print("目标：测试 SGMV kernel 在处理同一LoRA vs 不同LoRA时的性能差异")
    print("方法：完整的未合并推理 (WX + BAX)")
    print("=" * 70)
    
    try:
        # 创建测试器
        tester = UnmergedLoRAInferenceTester(config)
        
        # 运行主要对比测试
        comparison_results = tester.run_comparison_test()
        
        # 运行batch size分析
        # batch_analysis = tester.run_batch_size_analysis()
        
        print("\n🎉 测试完成!")
        print("\n💡 主要结论:")
        overhead = comparison_results['overhead_pct']
        if overhead > 10:
            print(f"1. 不同LoRA有显著的性能开销 ({overhead:.1f}%)")
            print("2. 这支持了dLoRA论文中动态批处理的必要性")
        elif overhead > 3:
            print(f"1. 不同LoRA有中等的性能开销 ({overhead:.1f}%)")
            print("2. SGMV kernel优化较好，但仍有优化空间")  
        else: 
            print(f"1. 不同LoRA的性能开销很小 ({overhead:.1f}%)")
            print("2. 当前配置下SGMV kernel表现良好")
        
        print("3. 可以通过调整batch size和LoRA配置来进一步优化")
        
    except Exception as e: 
        print(f"\n❌ 测试失败:  {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
"""
简化版 LoRA Benchmark 脚本
测试场景：
1. 8个请求都使用 lora0 (同一LoRA)
2. 8个请求随机使用不同LoRA (混合LoRA)
"""

import argparse
import asyncio
import aiohttp
import json
import random
import sys
import time
import traceback
from typing import List, Optional, Tuple
from dataclasses import dataclass
from tqdm. asyncio import tqdm
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from sglang.bench_serving import (
    _create_bench_client_session,
    RequestFuncInput,
    RequestFuncOutput,
    remove_prefix,
)

from sglang.srt. instances.lora_config_paths import LORA_PATH, NUM_LORAS


@dataclass
class BenchmarkMetrics:
    """Benchmark 性能指标"""
    scenario: str
    num_requests: int
    completed: int
    failed: int
    total_time_s: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_ttft_ms: float
    median_ttft_ms: float
    total_input_tokens: int
    total_output_tokens: int
    throughput_rps: float
    output_throughput_tps: float


def get_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    """获取 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    return tokenizer


def prepare_test_requests(
    num_requests: int = 8,
    prompt_len: int = 256,
    output_len: int = 128
) -> List[Tuple[str, int, int]]:
    """
    准备测试请求
    
    Returns:
        List of (prompt, prompt_len, output_len)
    """
    requests = []
    
    # 生成简单的测试 prompts
    test_prompts = [
        "Write a short story about artificial intelligence.",
        "Explain the concept of machine learning in simple terms.",
        "What are the benefits of using LoRA for model fine-tuning?",
        "Describe a futuristic city powered by renewable energy.",
        "How does natural language processing work?",
        "Explain quantum computing to a high school student.",
        "What is the future of autonomous vehicles?",
        "Describe the process of training a neural network.",
    ]
    
    for i in range(num_requests):
        prompt = test_prompts[i % len(test_prompts)]
        requests.append((prompt, prompt_len, output_len))
    
    return requests


async def async_request_sglang(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """向 SGLang 服务器发送请求"""
    api_url = request_func_input.api_url
    prompt = request_func_input.prompt
    
    async with _create_bench_client_session() as session:
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": request_func_input.output_len,
                "temperature": 0.0,  # 确定性输出
            }
        }
        
        # 如果指定了 LoRA，添加到 payload
        if request_func_input.lora_name:
            payload["lora_path"] = request_func_input. lora_name
        
        headers = {"Content-Type": "application/json"}
        
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        
        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        
        try: 
            async with session.post(
                url=api_url, 
                json=payload, 
                headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        
                        chunk = remove_prefix(
                            chunk_bytes. decode("utf-8"), 
                            "data: "
                        ).strip()
                        
                        if chunk == "[DONE]":
                            break
                        
                        try:
                            data = json. loads(chunk)
                        except Exception: 
                            continue
                        
                        # 检查错误
                        if isinstance(data, dict) and "error" in data:
                            output.error = str(data. get("error"))
                            output.success = False
                            break
                        
                        # 提取生成的文本
                        text_piece = ""
                        if isinstance(data, dict):
                            text_piece = data.get("text", "")
                        
                        if text_piece:
                            timestamp = time.perf_counter()
                            if ttft == 0.0:
                                ttft = timestamp - st
                                output.ttft = ttft
                            else:
                                output.itl. append(timestamp - most_recent_timestamp)
                            
                            most_recent_timestamp = timestamp
                            generated_text += text_piece
                    
                    output.generated_text = generated_text
                    output.success = True
                    output.latency = time.perf_counter() - st
                    output.output_len = len(generated_text. split())
                else:
                    output.error = f"HTTP {response.status}: {response.reason}"
                    output.success = False
                    
        except Exception as e:
            output.success = False
            output.error = f"Request failed: {str(e)}"
            exc_info = sys.exc_info()
            output.error += "\n" + "". join(traceback.format_exception(*exc_info))
        
        if pbar:
            pbar. update(1)
        
        return output


def calculate_metrics(
    outputs: List[RequestFuncOutput],
    scenario: str,
    total_time:  float
) -> BenchmarkMetrics: 
    """计算性能指标"""
    successful = [o for o in outputs if o. success]
    num_completed = len(successful)
    num_failed = len(outputs) - num_completed
    
    if num_completed == 0:
        raise ValueError("所有请求都失败了！")
    
    # 延迟统计
    latencies_ms = [o.latency * 1000 for o in successful]
    latencies_ms.sort()
    
    mean_latency = sum(latencies_ms) / len(latencies_ms)
    median_latency = latencies_ms[len(latencies_ms) // 2]
    p95_idx = int(len(latencies_ms) * 0.95)
    p95_latency = latencies_ms[p95_idx] if p95_idx < len(latencies_ms) else latencies_ms[-1]
    p99_idx = int(len(latencies_ms) * 0.99)
    p99_latency = latencies_ms[p99_idx] if p99_idx < len(latencies_ms) else latencies_ms[-1]
    
    # TTFT 统计
    ttfts_ms = [o.ttft * 1000 for o in successful if o.ttft > 0]
    mean_ttft = sum(ttfts_ms) / len(ttfts_ms) if ttfts_ms else 0
    ttfts_ms.sort()
    median_ttft = ttfts_ms[len(ttfts_ms) // 2] if ttfts_ms else 0
    
    # Token 统计
    total_input = sum(o.prompt_len for o in successful)
    total_output = sum(o.output_len for o in successful)
    
    # 吞吐量
    throughput_rps = num_completed / total_time
    output_throughput_tps = total_output / total_time
    
    return BenchmarkMetrics(
        scenario=scenario,
        num_requests=len(outputs),
        completed=num_completed,
        failed=num_failed,
        total_time_s=total_time,
        mean_latency_ms=mean_latency,
        median_latency_ms=median_latency,
        p95_latency_ms=p95_latency,
        p99_latency_ms=p99_latency,
        mean_ttft_ms=mean_ttft,
        median_ttft_ms=median_ttft,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        throughput_rps=throughput_rps,
        output_throughput_tps=output_throughput_tps
    )


async def run_benchmark_scenario(
    api_url: str,
    base_model_id: str,
    requests: List[Tuple[str, int, int]],
    lora_assignment: str,  # "same" or "random"
    disable_tqdm: bool = False
) -> BenchmarkMetrics:
    """
    运行单个 benchmark 场景
    
    Args:
        api_url: SGLang 服务器 API URL
        base_model_id:  基础模型 ID
        requests: 测试请求列表
        lora_assignment: LoRA 分配策略 ("same" 或 "random")
        disable_tqdm:  是否禁用进度条
    """
    scenario_name = f"{lora_assignment.upper()}_LORA"
    print(f"\n{'='*70}")
    print(f"场景: {scenario_name}")
    print(f"{'='*70}")
    
    if lora_assignment == "same": 
        print("所有 8 个请求都使用 lora0")
    else:
        print(f"8 个请求随机使用 lora0 到 lora{NUM_LORAS-1}")
    
    # 准备请求
    tasks = []
    pbar = None if disable_tqdm else tqdm(total=len(requests), desc=scenario_name)
    
    start_time = time.perf_counter()
    
    for idx, (prompt, prompt_len, output_len) in enumerate(requests):
        # 确定使用哪个 LoRA
        if lora_assignment == "same":
            lora_name = "lora0"
        else:   # random
            lora_id = random.randint(0, NUM_LORAS - 1)
            lora_name = f"lora{lora_id}"
        
        request_input = RequestFuncInput(
            model=base_model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            lora_name=lora_name,
            image_data=None,
            extra_request_body={}
        )
        
        # 创建异步任务
        task = asyncio.create_task(
            async_request_sglang(
                request_func_input=request_input,
                pbar=pbar
            )
        )
        tasks.append(task)
    
    # 等待所有请求完成
    outputs = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start_time
    
    if pbar:
        pbar.close()
    
    # 计算指标
    metrics = calculate_metrics(outputs, scenario_name, total_time)
    
    # 打印结果
    print_metrics(metrics)
    
    return metrics


def print_metrics(metrics: BenchmarkMetrics):
    """打印性能指标"""
    print(f"\n{'-'*70}")
    print(f"场景: {metrics.scenario}")
    print(f"{'-'*70}")
    print(f"总请求数:         {metrics.num_requests}")
    print(f"成功:             {metrics.completed}")
    print(f"失败:            {metrics.failed}")
    print(f"总耗时:          {metrics.total_time_s:.2f} s")
    print(f"\n延迟统计 (ms):")
    print(f"  平均延迟:      {metrics.mean_latency_ms:.2f}")
    print(f"  中位延迟:      {metrics.median_latency_ms:.2f}")
    print(f"  P95 延迟:      {metrics.p95_latency_ms:.2f}")
    print(f"  P99 延迟:      {metrics.p99_latency_ms:.2f}")
    print(f"\nTTFT 统计 (ms):")
    print(f"  平均 TTFT:     {metrics.mean_ttft_ms:.2f}")
    print(f"  中位 TTFT:     {metrics.median_ttft_ms:.2f}")
    print(f"\nToken 统计:")
    print(f"  总输入 tokens:   {metrics.total_input_tokens}")
    print(f"  总输出 tokens:   {metrics.total_output_tokens}")
    print(f"\n吞吐量:")
    print(f"  请求吞吐量:    {metrics.throughput_rps:.2f} req/s")
    print(f"  Token 吞吐量:  {metrics.output_throughput_tps:.2f} tok/s")
    print(f"{'='*70}")


def compare_results(
    same_lora_metrics: BenchmarkMetrics,
    random_lora_metrics:  BenchmarkMetrics
):
    """对比两个场景的结果"""
    print(f"\n{'='*70}")
    print("性能对比分析")
    print(f"{'='*70}")
    
    # 延迟对比
    latency_diff_pct = (
        (random_lora_metrics.mean_latency_ms - same_lora_metrics.mean_latency_ms)
        / same_lora_metrics.mean_latency_ms * 100
    )
    
    print(f"\n平均延迟:")
    print(f"  同一LoRA (lora0):     {same_lora_metrics. mean_latency_ms:.2f} ms")
    print(f"  随机LoRA (混合):      {random_lora_metrics. mean_latency_ms:.2f} ms")
    print(f"  差异:                  {latency_diff_pct:+.2f}%")
    
    # TTFT 对比
    ttft_diff_pct = (
        (random_lora_metrics.mean_ttft_ms - same_lora_metrics.mean_ttft_ms)
        / same_lora_metrics.mean_ttft_ms * 100
    )
    
    print(f"\n平均 TTFT:")
    print(f"  同一LoRA (lora0):     {same_lora_metrics. mean_ttft_ms:.2f} ms")
    print(f"  随机LoRA (混合):      {random_lora_metrics.mean_ttft_ms:.2f} ms")
    print(f"  差异:                 {ttft_diff_pct:+.2f}%")
    
    # 吞吐量对比
    throughput_diff_pct = (
        (same_lora_metrics.throughput_rps - random_lora_metrics.throughput_rps)
        / same_lora_metrics.throughput_rps * 100
    )
    
    print(f"\n请求吞吐量:")
    print(f"  同一LoRA (lora0):     {same_lora_metrics.throughput_rps:.2f} req/s")
    print(f"  随机LoRA (混合):      {random_lora_metrics.throughput_rps:.2f} req/s")
    print(f"  吞吐量损失:           {throughput_diff_pct:+.2f}%")
    
    # 结论
    print(f"\n💡 关键发现:")
    if latency_diff_pct > 5:
        print(f"  • 随机LoRA比同一LoRA延迟高 {latency_diff_pct:.1f}%")
        print(f"  • 这验证了 LoRA 多样性带来的性能开销")
        print(f"  • 建议使用 dLoRA 等优化策略来缓解")
    elif latency_diff_pct > 1:
        print(f"  • 随机LoRA有轻微的性能开销 ({latency_diff_pct:.1f}%)")
        print(f"  • 当前配置下开销可接受")
    else:
        print(f"  • 两种场景性能相近，开销很小")
        print(f"  • SGMV kernel 优化良好")
    
    print(f"{'='*70}")


async def main_async(args):
    """主测试函数"""
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 准备测试请求
    print("准备测试请求...")
    requests = prepare_test_requests(
        num_requests=args.num_requests,
        prompt_len=args.prompt_len,
        output_len=args.output_len
    )
    print(f"✓ 准备了 {len(requests)} 个测试请求")
    
    # 服务器 URL
    api_url = f"http://{args.host}:{args. port}/generate"
    base_model_id = LORA_PATH["base"]
    
    print(f"\n服务器配置:")
    print(f"  URL: {api_url}")
    print(f"  模型: {base_model_id}")
    print(f"  LoRA 数量: {NUM_LORAS}")
    print(f"  LoRA 后端: {args.lora_backend}")
    
    # 测试服务器连接
    print(f"\n测试服务器连接...")
    try:
        async with _create_bench_client_session() as session:
            async with session.get(f"http://{args.host}:{args.port}/health") as resp:
                if resp.status == 200:
                    print("✓ 服务器连接正常")
                else:
                    print(f"⚠️  服务器返回状态码: {resp.status}")
    except Exception as e: 
        print(f"❌ 无法连接到服务器: {e}")
        print(f"请确保服务器已启动:  http://{args.host}:{args.port}")
        return 1
    
    # 场景 1: 所有请求使用 lora0
    print(f"\n{'='*70}")
    print("开始测试场景 1: 所有请求使用 lora0")
    print(f"{'='*70}")
    
    same_lora_metrics = await run_benchmark_scenario(
        api_url=api_url,
        base_model_id=base_model_id,
        requests=requests,
        lora_assignment="same",
        disable_tqdm=args.disable_tqdm
    )
    
    # 等待一下再开始下一个场景
    print("\n等待 2 秒后开始下一个场景...")
    await asyncio.sleep(2)
    
    # 场景 2: 随机使用不同 LoRA
    print(f"\n{'='*70}")
    print("开始测试场景 2: 随机使用不同 LoRA")
    print(f"{'='*70}")
    
    random_lora_metrics = await run_benchmark_scenario(
        api_url=api_url,
        base_model_id=base_model_id,
        requests=requests,
        lora_assignment="random",
        disable_tqdm=args.disable_tqdm
    )
    
    # 对比结果
    compare_results(same_lora_metrics, random_lora_metrics)
    
    # 保存结果
    if args.output_file:
        results = {
            "config": {
                "num_requests": args.num_requests,
                "prompt_len":  args.prompt_len,
                "output_len": args.output_len,
                "lora_backend": args.lora_backend,
                "num_loras": NUM_LORAS,
            },
            "same_lora":  {
                "scenario": same_lora_metrics.scenario,
                "mean_latency_ms": same_lora_metrics.mean_latency_ms,
                "median_latency_ms": same_lora_metrics.median_latency_ms,
                "mean_ttft_ms": same_lora_metrics.mean_ttft_ms,
                "throughput_rps": same_lora_metrics.throughput_rps,
            },
            "random_lora": {
                "scenario": random_lora_metrics.scenario,
                "mean_latency_ms": random_lora_metrics.mean_latency_ms,
                "median_latency_ms": random_lora_metrics.median_latency_ms,
                "mean_ttft_ms": random_lora_metrics.mean_ttft_ms,
                "throughput_rps": random_lora_metrics. throughput_rps,
            }
        }
        
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ 结果已保存到: {args.output_file}")
    
    print("\n🎉 测试完成!")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="简化版 LoRA Benchmark - 测试同一LoRA vs 随机LoRA"
    )
    
    # 服务器配置
    parser. add_argument("--host", type=str, default="127.0.0.1",
                       help="SGLang 服务器地址")
    parser.add_argument("--port", type=int, default=30001,
                       help="SGLang 服务器端口")
    
    # 测试配置
    parser.add_argument("--num-requests", type=int, default=8,
                       help="每个场景的请求数量")
    parser.add_argument("--prompt-len", type=int, default=256,
                       help="输入 prompt 长度")
    parser.add_argument("--output-len", type=int, default=128,
                       help="输出 token 长度")
    
    # LoRA 配置
    parser.add_argument("--lora-backend", type=str, default="csgmv",
                       choices=["triton", "csgmv"],
                       help="LoRA 后端")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--output-file", type=str, default=None,
                       help="保存结果到 JSON 文件")
    parser.add_argument("--disable-tqdm", action="store_true",
                       help="禁用进度条")
    
    args = parser.parse_args()
    
    try:
        exit_code = asyncio.run(main_async(args))
        return exit_code
    except KeyboardInterrupt: 
        print("\n\n测试被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
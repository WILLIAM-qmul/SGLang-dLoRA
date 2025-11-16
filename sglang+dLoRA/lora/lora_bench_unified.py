# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import asyncio
import json
import random
import resource
import sys
import time
import traceback
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
from launch_server import LORA_PATH, NUM_LORAS
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from sglang.bench_serving import (
    _create_bench_client_session,
    RequestFuncInput,
    RequestFuncOutput,
    calculate_metrics,
    get_request,
    get_tokenizer,
    remove_prefix,
    sample_random_requests,
    sample_sharegpt_requests,
)

# Import trace generation logic
from workload_generator.trace import Trace
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator

global args


# trace新增: trace 回放异步生成器
async def get_trace_requests(
    input_requests: List[Tuple[str, int, int]],
    workload: List[Tuple[int, float]],
) -> AsyncGenerator[Tuple[int, Any], None]:
    """
    workload: List[(model_id, inter_arrival_interval_seconds)]
    逐条按照间隔 sleep,再产出 (model_id, request)
    """
    assert len(workload) >= len(input_requests), "workload 长度需 >= 请求数"
    for idx, (model_id, interval) in enumerate(workload[: len(input_requests)]):
        await asyncio.sleep(interval)
        yield model_id, input_requests[idx]


# set ignore_eos True by default
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url

    prompt = request_func_input.prompt

    async with _create_bench_client_session() as session:
        if args.base_only:
            payload = {
                "text": prompt,
                "sampling_params": {"max_new_tokens": request_func_input.output_len},
            }
        else:
            payload = {
                "text": prompt,
                "sampling_params": {"max_new_tokens": request_func_input.output_len},
                "lora_path": request_func_input.lora_name,
            }
        headers = {"Authorization": ""}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st

        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        latency = time.perf_counter() - st
                        if chunk == "[DONE]":
                            pass
                        else:
                            data = json.loads(chunk)

                            if data["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                    output.output_len = request_func_input.output_len
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


ASYNC_REQUEST_FUNCS = {
    "sglang": async_request_openai_completions,
}


async def benchmark(
    backend: str,
    api_url: str,
    base_model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    disable_tqdm: bool,
    extra_request_body: Dict[str, Any],
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # 处理实例 URL
    if args.inference_architecture == "dlora":
        # dLoRA 模式: 使用统一的 Unified Server
        unified_server_url = args.unified_server_url or f"http://{args.host}:{args.port}"
        print(f"\n[Benchmark] Using dLoRA Unified Server: {unified_server_url}")
        print(f"  - Migration Type: {args.migration_type}")
        print(f"  - Migration Interval: {args.migration_interval}s")
        print(f"  - Migration Threshold: {args.migration_req_threshold}")
        
        # 所有请求都发送到统一服务器
        static_lora_routing: Dict[str, str] = {}
        for lora_idx in range(NUM_LORAS):
            static_lora_routing[f"lora{lora_idx}"] = f"{unified_server_url}/generate"
        static_lora_routing["dummy"] = f"{unified_server_url}/generate"
        
    else:
        # SGLang 模式: 静态均分 LoRA -> 实例
        raw_worker_urls = args.worker_urls or []
        worker_base_urls: List[str] = []
        for u in raw_worker_urls:
            u = u.rstrip("/")
            if u.endswith("/generate"):
                u = u[: -len("/generate")]
            worker_base_urls.append(u)
        if not worker_base_urls:
            base_api = api_url.rstrip("/")
            if base_api.endswith("/generate"):
                base_api = base_api[: -len("/generate")]
            worker_base_urls = [base_api]

        num_instances = len(worker_base_urls)

        static_lora_routing: Dict[str, str] = {}
        for lora_idx in range(NUM_LORAS):
            inst_id = lora_idx % num_instances
            static_lora_routing[f"lora{lora_idx}"] = f"{worker_base_urls[inst_id]}/generate"
        static_lora_routing["dummy"] = f"{worker_base_urls[0]}/generate"
        
        print("\n[Benchmark] 静态 LoRA 路由 (sglang):")
        for k, v in static_lora_routing.items():
            if k.startswith("lora"):
                print(f"  {k} -> {v}")

    # 初始测试请求
    print("Starting initial single prompt test run...")
    test_request = input_requests[0]
    test_lora_name = "dummy" if args.base_only else "lora0"
    test_api_url = static_lora_routing[test_lora_name]

    test_input = RequestFuncInput(
        model=base_model_id,
        prompt=test_request.prompt,
        api_url=test_api_url,
        prompt_len=test_request.prompt_len,
        output_len=test_request.output_len,
        lora_name=test_lora_name,
        image_data=None,
        extra_request_body=extra_request_body,
    )
    test_pbar = tqdm(total=1, desc="Test request")
    test_output = await request_func(request_func_input=test_input, pbar=test_pbar)
    test_pbar.close()
    
    if not test_output.success:
        raise ValueError(f"Initial test run failed: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    # 主 benchmark 循环
    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    request_idx = 0

    if args.use_trace:
        # Trace 回放模式
        workload = Trace(
            trace_name=args.trace_name,
            trace_dir=args.trace_path,
            start_time=args.trace_start_time,
            end_time=args.trace_end_time,
            need_sort=args.trace_need_sort,
        ).replay_to_workload(
            num_models=NUM_LORAS,
            num_reqs=len(input_requests),
            arrival_distribution=args.trace_arrival_distribution,
            interval_minutes=args.trace_interval_minutes,
            tot_rate=args.trace_total_rate,
            cv=args.trace_cv,
            map_stride=args.trace_map_stride,
        )
        
        request_generator = get_trace_requests(input_requests, workload)
        
        async for model_id, request in request_generator:
            lora_path = "dummy" if args.base_only else f"lora{model_id}"
            target_api_url = static_lora_routing[lora_path]
            
            req_input = RequestFuncInput(
                model=base_model_id,
                prompt=request.prompt,
                api_url=target_api_url,
                prompt_len=request.prompt_len,
                output_len=request.output_len,
                lora_name=lora_path,
                image_data=None,
                extra_request_body=extra_request_body,
            )
            tasks.append(asyncio.create_task(request_func(request_func_input=req_input, pbar=pbar)))
            request_idx += 1
    
    else:
        # 非 trace 模式
        prompts_per_lora = max(1, len(input_requests) // NUM_LORAS)
        
        async for request in get_request(input_requests, request_rate):
            if args.base_only:
                lora_path = "dummy"
                lora_id = 0
            else:
                if args.lora_assignment_strategy == "random":
                    lora_id = random.randint(0, NUM_LORAS - 1)
                elif args.lora_assignment_strategy == "sequential":
                    lora_id = min(request_idx // prompts_per_lora, NUM_LORAS - 1)
                else:
                    lora_id = 0
                lora_path = f"lora{lora_id}"

            target_api_url = static_lora_routing[lora_path]

            req_input = RequestFuncInput(
                model=base_model_id,
                prompt=request.prompt,
                api_url=target_api_url,
                prompt_len=request.prompt_len,
                output_len=request.output_len,
                lora_name=lora_path,
                image_data=None,
                extra_request_body=extra_request_body,
            )
            tasks.append(asyncio.create_task(request_func(request_func_input=req_input, pbar=pbar)))
            request_idx += 1

    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        backend=backend,
    )

    # 打印结果
    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Architecture:", args.inference_architecture))
    print("{:<40} {:<10}".format("Traffic request rate:", request_rate))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10}".format(
            "Total generated tokens (retokenized):", metrics.total_output_retokenized
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format("Total throughput (tok/s):", metrics.total_throughput)
    )
    print("{s:{c}^{n}}".format(s="End-to-End Latency", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms)
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Median E2E Latency (ms):", metrics.median_e2e_latency_ms
        )
    )
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print(
        "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    )
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("=" * 50)

    if (
        metrics.median_ttft_ms is not None
        and metrics.mean_itl_ms is not None
        and metrics.output_throughput is not None
    ):
        result = {
            "backend": args.backend,
            "architecture": args.inference_architecture,
            "migration_type": args.migration_type if args.inference_architecture == "dlora" else "N/A",
            "request_rate": request_rate,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "total_output_tokens_retokenized": metrics.total_output_retokenized,
            "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
            "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
            "median_ttft_ms": metrics.median_ttft_ms,
            "median_itl_ms": metrics.median_itl_ms,
            "output_throughput": metrics.output_throughput,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "random_range_ratio": args.random_range_ratio,
            "duration": benchmark_duration,
            "completed": metrics.completed,
        }
    else:
        print(f"Error running benchmark for request rate: {request_rate}")
        print("-" * 30)

    # 确定输出文件名
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        arch_suffix = f"_{args.inference_architecture}"
        if args.inference_architecture == "dlora":
            arch_suffix += f"_{args.migration_type}"
        output_file_name = f"{args.backend}{arch_suffix}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl"

    # 追加结果到 JSONL 文件
    with open(output_file_name, "a") as file:
        file.write(json.dumps(result) + "\n")

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "total_output_tokens_retokenized": metrics.total_output_retokenized,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
    }
    return result


def run_benchmark(args_: argparse.Namespace):
    global args
    args = args_

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set url
    if args.port is None:
        if args.inference_architecture == "dlora":
            args.port = 8000  # Unified server 默认端口
        else:
            args.port = 30000  # SGLang 默认端口

    api_url = (
        f"{args.base_url}/generate"
        if args.base_url
        else f"http://{args.host}:{args.port}/generate"
    )

    print(f"{args}\n")

    # Read dataset
    backend = args.backend
    base_model_id = args.model = LORA_PATH["base"]
    tokenizer_id = args.model

    tokenizer = get_tokenizer(tokenizer_id)

    input_requests = sample_sharegpt_requests(
        dataset_path="/workspace/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json",
        num_requests=args.num_prompts,
        tokenizer=tokenizer,
    )

    return asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_model_id=base_model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=args.request_rate,
            disable_tqdm=False,
            extra_request_body={},
        )
    )


def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online lora serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        default="sglang",
        help="Must specify a backend, depending on the LLM Inference Engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Default host is 127.0.0.1."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="If not set, the default port is configured according to architecture.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to process. Default is 100.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range of sampled ratio of input/output length, "
        "used only for random dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
    )
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    
    # LoRA 分配策略
    parser.add_argument(
        "--lora-assignment-strategy",
        type=str,
        default="sequential",
        choices=["random", "sequential"],
        help="Strategy to assign LoRA models to prompts. 'random' assigns a LoRA randomly to each prompt. "
        "'sequential' divides prompts evenly among LoRA models in sequence.",
    )
    
    # Multi-instance args (仅用于 sglang 架构)
    parser.add_argument(
        "--worker-urls",
        type=str,
        nargs="+",
        default=None,
        help="多个实例地址(仅 sglang 模式),例如: http://127.0.0.1:30001 http://127.0.0.1:30002",
    )
    
    # Trace 控制参数
    parser.add_argument("--use-trace", action="store_true", help="启用真实 trace 回放调度")
    parser.add_argument("--trace-name", type=str, default="azure_v2", choices=["azure_v1", "azure_v2"], help="trace 数据集名称")
    parser.add_argument("--trace-path", type=str, default="/workspace/datasets/maf2/", help="trace 根目录")
    parser.add_argument("--trace-start-time", type=str, default="0.0.0", help="起始时间 day.hour.min")
    parser.add_argument("--trace-end-time", type=str, default="0.0.60", help="结束时间 day.hour.min")
    parser.add_argument("--trace-interval-minutes", type=int, default=60, help="聚合直方图的时间粒度 (minutes)")
    parser.add_argument("--trace-arrival-distribution", type=str, default="gamma", choices=["gamma", "exponential"], help="到达分布类型")
    parser.add_argument("--trace-total-rate", type=float, default=4.0, help="整体总到达率")
    parser.add_argument("--trace-cv", type=float, default=10.0, help="Gamma 分布的变异系数")
    parser.add_argument("--trace-map-stride", type=int, default=1, help="函数映射到模型的步长")
    parser.add_argument("--trace-need-sort", action="store_true", help="是否按调用总次数排序函数")
    
    # 推理架构选择
    parser.add_argument(
        "--inference-architecture",
        type=str,
        default="sglang",
        choices=["sglang", "dlora"],
        help="LoRA serving architecture: 'sglang' for static assignment, 'dlora' for dynamic scheduling",
    )
    
    # dLoRA 参数 (仅用于 dlora 架构)
    parser.add_argument(
        "--unified-server-url",
        type=str,
        default=None,
        help="dLoRA Unified Server URL (默认: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--migration-type",
        type=str,
        default="period_mig",
        choices=["dispatch_only", "dispatch_mig", "period_mig", "1", "2", "3"],
        help="迁移策略 (仅 dlora 模式)",
    )
    parser.add_argument(
        "--migration-interval",
        type=int,
        default=10,
        help="周期迁移间隔秒 (仅 dlora 模式)",
    )
    parser.add_argument(
        "--migration-req-threshold",
        type=int,
        default=16,
        help="负载迁移请求阈值 (仅 dlora 模式)",
    )
    parser.add_argument(
        "--max-loras-per-batch",
        type=int,
        default=8,
        help="批次可合并的最大 LoRA 数量",
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=16,
        help="实例最大并发请求数",
    )
    
    args = parser.parse_args()
    run_benchmark(args)
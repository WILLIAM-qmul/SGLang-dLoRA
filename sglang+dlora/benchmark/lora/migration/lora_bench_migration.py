# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0

import argparse
import asyncio
import aiohttp
import json
import random
import resource
import sys
import time
import traceback
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator
from collections import defaultdict
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

import psutil
import GPUtil

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

from sglang.srt.workload_generator.trace import Trace
from sglang.srt.instances.lora_config_paths import LORA_PATH, NUM_LORAS

global args


async def get_trace_requests(
    input_requests: List[Tuple[str, int, int]],
    workload: List[Tuple[int, float]],
) -> AsyncGenerator[Tuple[int, Any], None]:
    """workload: List[(model_id, inter_arrival_interval_seconds)]"""
    assert len(workload) >= len(input_requests), "workload 长度需 >= 请求数"
    for idx, (model_id, interval) in enumerate(workload[: len(input_requests)]):
        await asyncio.sleep(interval)
        yield model_id, input_requests[idx]


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """Send request to SGLang server."""
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

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ").strip()
                        latency = time.perf_counter() - st

                        if chunk == "[DONE]":
                            continue

                        try:
                            data = json.loads(chunk)
                        except Exception:
                            # Ignore non-JSON SSE events
                            continue

                        # unified server / backend error events
                        if isinstance(data, dict) and "error" in data:
                            output.error = str(data.get("error"))
                            output.success = False
                            break

                        text_piece = ""
                        if isinstance(data, dict):
                            text_piece = data.get("text") or ""

                        if text_piece:
                            timestamp = time.perf_counter()
                            if ttft == 0.0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)

                            most_recent_timestamp = timestamp
                            generated_text += text_piece

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

    use_unified_server = args.inference_architecture == "dlora"
    
    if use_unified_server:
        # dLoRA 模式: 所有请求都通过 Unified Server (动态负载均衡)
        unified_server_url = args.unified_server_url or f"http://{args.host}:{args.port}"
        print(f"\n[Benchmark] Using dLoRA Unified Server: {unified_server_url}")
        print(f"  - Routing Strategy: Least-Loaded (实时动态负载均衡)")
        print(f"  - Mode: {'Trace Replay' if args.use_trace else 'Poisson Arrival'}")
        
        backend_instance_urls = [
            f"http://{args.host}:{args.sglang_port + i}"
            for i in range(args.num_instances)
        ]
        
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

        # 按块均分 LoRA 到实例
        static_lora_routing: Dict[str, str] = {}
        base = NUM_LORAS // num_instances
        remainder = NUM_LORAS % num_instances
        lora_idx = 0
        for inst_id in range(num_instances):
            count = base + (1 if inst_id < remainder else 0)
            for _ in range(count):
                static_lora_routing[f"lora{lora_idx}"] = f"{worker_base_urls[inst_id]}/generate"
                lora_idx += 1
        while lora_idx < NUM_LORAS:
            static_lora_routing[f"lora{lora_idx}"] = f"{worker_base_urls[0]}/generate"
            lora_idx += 1
        static_lora_routing["dummy"] = f"{worker_base_urls[0]}/generate"
        
        backend_instance_urls = worker_base_urls
        
        print("\n[Benchmark] 静态 LoRA 路由 (sglang):")
        for k, v in static_lora_routing.items():
            if k.startswith("lora"):
                print(f"  {k} -> {v}")
                

    # 初始测试请求
    print("\nStarting initial single prompt test run...")
    test_request = input_requests[0]
    test_lora_name = "dummy" if args.base_only else "lora0"
    
    if use_unified_server:
        test_api_url = f"{unified_server_url}/generate"
    else:
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
        
    
    # 通知服务器 benchmark 开始
    server_url = args.unified_server_url if args.inference_architecture == "dlora" else f"http://{args.host}:{args.port}"
    bench_info = {
        "backend": backend,
        "architecture": args.inference_architecture,
        "num_prompts": len(input_requests),
        "request_rate": request_rate,
        "use_trace": args.use_trace,
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/start_benchmark",
                json={"info": json.dumps(bench_info)}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"\n✅ Server notified:  {result. get('message')}")
                else:
                    print(f"\n⚠️  Failed to notify server (start): {resp.status}")
    except Exception as e:
        print(f"\n⚠️  Failed to notify server (start): {e}")

    # 主 benchmark 循环
    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    request_idx = 0

    if args.use_trace:
        # Trace 回放模式 - 支持动态路由
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
            
            # 关键: 根据架构选择路由方式
            if use_unified_server:
                # dLoRA: 所有请求都通过 Unified Server (实时负载均衡)
                target_api_url = f"{unified_server_url}/generate"
            else:
                # SGLang: 使用静态路由
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
        # 非 trace 模式 (Poisson 到达)
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

            # 关键: 根据架构选择路由方式
            if use_unified_server:
                # dLoRA: 所有请求都通过 Unified Server (实时负载均衡)
                target_api_url = f"{unified_server_url}/generate"
            else:
                # SGLang: 使用静态路由
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
        
    # 新增: 通知服务器 benchmark 结束
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/end_benchmark",
                json={"info": json.dumps(bench_info)}
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"\n✅ Server notified: {result.get('message')}")
                else: 
                    print(f"\n⚠️  Failed to notify server (end): {resp.status}")
    except Exception as e: 
        print(f"\n⚠️  Failed to notify server (end): {e}")

    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        backend=backend,
    )

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
    print("{:<40} {:<10}".format("Architecture:", args.inference_architecture))
    print("{:<40} {:<10}".format("Traffic request rate:", request_rate))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("\n{s:{c}^{n}}".format(s=" Performance Metrics ", n=50, c="-"))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{s:{c}^{n}}".format(s="Latency Metrics", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean E2E Latency (ms):", metrics.mean_e2e_latency_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("=" * 50)

    result = {
        "backend": args.backend,
        "architecture": args.inference_architecture,
        "routing_strategy": "least_loaded" if use_unified_server else "static",
        "request_rate": request_rate,
        "completed": metrics.completed,
        "duration": benchmark_duration,
        "total_input_tokens": metrics.total_input,
        "total_generated_tokens": metrics.total_output,
        "total_throughput": metrics.total_throughput,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_e2e_latency_ms": metrics.mean_e2e_latency_ms,
        "median_e2e_latency_ms": metrics.median_e2e_latency_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
    }
    
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        trace_suffix = "_trace" if args.use_trace else ""
        output_file_name = f"{args.backend}_{args.inference_architecture}{trace_suffix}_{args.trace_name}_{args.trace_total_rate}_{args.trace_map_stride}_{args.num_prompts}.jsonl"

    with open(output_file_name, "a") as file:
        file.write(json.dumps(result) + "\n")

    return result


def run_benchmark(args_: argparse.Namespace):
    global args
    args = args_

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set url and port based on architecture
    if args.port is None:
        if args.inference_architecture == "dlora":
            args.port = 8000
        else:
            args.port = 30001

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
    parser.add_argument("--backend", type=str, default="sglang",
                       choices=list(ASYNC_REQUEST_FUNCS.keys()))
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, help="Port (auto-set based on architecture if not provided)")
    parser.add_argument("--num-prompts", type=int, default=4000)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--seed", type=int, default=1)
    
    # LoRA assignment
    parser.add_argument("--lora-assignment-strategy", type=str, default="sequential",
                       choices=["random", "sequential"])
    
    # Multi-instance args (sglang mode)
    parser.add_argument("--worker-urls", type=str, nargs="+",
                       default=["http://127.0.0.1:30001", "http://127.0.0.1:30002"])
    
    # Trace parameters
    parser.add_argument("--use-trace", action="store_true")
    parser.add_argument("--trace-name", type=str, default="azure_v2", choices=["azure_v1", "azure_v2"])
    parser.add_argument("--trace-path", type=str, default="/workspace/datasets/maf2/")
    parser.add_argument("--trace-start-time", type=str, default="0.0.0")
    parser.add_argument("--trace-end-time", type=str, default="0.2.0")
    parser.add_argument("--trace-interval-minutes", type=int, default=60)
    parser.add_argument("--trace-arrival-distribution", type=str, default="gamma",
                       choices=["gamma", "exponential"])
    parser.add_argument("--trace-total-rate", type=float, default=40.0)
    parser.add_argument("--trace-cv", type=float, default=1.0)
    parser.add_argument("--trace-map-stride", type=int, default=1)
    parser.add_argument("--trace-need-sort", action="store_true")
    
    # Architecture selection
    parser.add_argument("--inference-architecture", type=str, default="dlora",
                       choices=["sglang", "dlora"],
                       help="'sglang' for static assignment, 'dlora' for dynamic load balancing")
    
    # dLoRA parameters
    parser.add_argument("--unified-server-url", type=str, default=None,
                       help="Unified Server URL for dLoRA mode (default: http://127.0.0.1:8000)")
    parser.add_argument("--num-instances", type=int, default=2,
                       help="Number of backend SGLang instances")
    parser.add_argument("--sglang-port", type=int, default=30001,
                       help="Starting port for SGLang instances (used in dLoRA mode)")
    
    args = parser.parse_args()
    run_benchmark(args)
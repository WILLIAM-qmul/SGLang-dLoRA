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

from sglang.bench_serving import ( # 导入sglang.bench_serving中的工具函数和类
    # AIOHTTP_TIMEOUT, # aiohttp超时时间
    _create_bench_client_session,  # 使用 bench_serving 提供的会话工厂，包含超时和缓冲区设置
    RequestFuncInput, # 请求输入数据结构
    RequestFuncOutput, # 请求输出数据结构
    calculate_metrics, # 计算评测指标函数
    get_request, # 获取请求生成器
    get_tokenizer, # 获取分词器函数
    remove_prefix, # 移除字符串前缀函数
    sample_random_requests, # 随机采样请求函数
    sample_sharegpt_requests, # 采样ShareGPT请求函数
)

"""
1. sample_random_requests：负责“生成请求样本”。输出是一个 List[DatasetRow]（每个 DatasetRow 包含 prompt、prompt_len、output_len、可选 timestamp 等）。不负责什么时候发送，只负责构建请求内容和长度分布。

2. get_request：负责“调度/发送请求”。它是一个异步生成器（async generator），接收 input_requests（即 sample_random_requests 的返回值）和 request_rate / use_trace_timestamps 等参数，按策略逐个 yield DatasetRow：
use_trace_timestamps=True：按每个 DatasetRow.timestamp 回放（并支持 slowdown_factor）。
否则按 request_rate（Poisson / 指数分布抽样间隔）或 request_rate==inf（全部立即 yield）来控制发送时间间隔。
"""

global args


# set ignore_eos True by default
async def async_request_openai_completions(
    request_func_input: RequestFuncInput, # 请求输入数据
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url # 获取API地址
    # assert api_url.endswith(
    #     "completions"
    # ), "OpenAI Completions API URL must end with 'completions'."

    prompt = request_func_input.prompt # 获取请求的prompt文本

    # async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session: # 创建异步HTTP会话
    async with _create_bench_client_session() as session:  # 使用 bench_serving 的会话工厂（含超时和read_bufsize）
        # payload = {
        #     "model": request_func_input.model,
        #     "prompt": prompt,
        #     "temperature": 0.0,
        #     "best_of": 1,
        #     "max_tokens": request_func_input.output_len,
        #     "stream": not args.disable_stream,
        #     "ignore_eos": not args.disable_ignore_eos,
        #     **request_func_input.extra_request_body,
        # }
        # headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        if args.base_only:
            payload = { # 如果只用基础模型
                "text": prompt,
                "sampling_params": {"max_new_tokens": request_func_input.output_len},
            }
        else: # 否则使用Lora模型
            payload = {
                "text": prompt,
                "sampling_params": {"max_new_tokens": request_func_input.output_len},
                "lora_path": request_func_input.lora_name,
            }
        headers = {"Authorization": ""}

        output = RequestFuncOutput() # 创建请求输出对象
        output.prompt_len = request_func_input.prompt_len # 设置prompt长度

        generated_text = "" # 初始化生成文本
        ttft = 0.0 # 初始化首token时间
        st = time.perf_counter() # 记录起始时间
        most_recent_timestamp = st # 记录最近一次token时间
        '''
        chunk_bytes是从 HTTP 响应流中读取到的原始字节数据，每次循环会获取一段内容（通常是模型生成的部分文本或控制信息），类型为 bytes。

        chunk_bytes.strip() 是去除这段字节数据前后的空白字符（如换行、空格等），得到仍为 bytes 类型的内容。

        chunk 是将 chunk_bytes 解码为字符串（utf-8），并去除前缀 "data: " 后的结果。它通常是一个 JSON 字符串或特殊标志（如 [DONE]），用于后续解析。
        '''
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response: # 发送POST请求
                if response.status == 200: # 如果响应成功
                    async for chunk_bytes in response.content: # 异步读取响应内容
                        chunk_bytes = chunk_bytes.strip() # 去除空白
                        if not chunk_bytes: # 跳过空内容
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ") # 去除前缀
                        latency = time.perf_counter() - st # 计算延迟
                        if chunk == "[DONE]": # 如果收到结束标志
                            pass
                        else: # 解析JSON数据
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["text"]: # 如果有生成文本
                                # if data["choices"][0]["text"]:
                                timestamp = time.perf_counter() # 当前时间戳
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st # 计算TTFT
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp) # 记录token间延迟

                                most_recent_timestamp = timestamp # 更新最近token时间
                                # generated_text += data["choices"][0]["text"]
                                generated_text += data["text"] # 拼接生成文本

                    output.generated_text = generated_text # 保存生成文本
                    output.success = True # 标记成功
                    output.latency = latency # 保存延迟
                    output.output_len = request_func_input.output_len # 保存输出长度
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar: # 如果有进度条
        pbar.update(1) # 更新进度
    return output # 返回请求结果


ASYNC_REQUEST_FUNCS = {
    "sglang": async_request_openai_completions, # 定义后端与请求函数的映射
}


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    disable_tqdm: bool,
    extra_request_body: Dict[str, Any],
):
    if backend in ASYNC_REQUEST_FUNCS: # 检查后端是否支持
        request_func = ASYNC_REQUEST_FUNCS[backend] # 获取对应请求函数
    else:
        raise ValueError(f"Unknown backend: {backend}") # 不支持则报错

    print("Starting initial single prompt test run...") # 打印测试信息
    test_request = input_requests[0] # 取第一个请求做测试
    test_lora_name = "dummy"
    if not args.base_only:
        test_lora_name = "lora0"
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_request.prompt,
        api_url=api_url,
        prompt_len=test_request.prompt_len,
        output_len=test_request.output_len,
        lora_name=test_lora_name,  # the lora_name argument is dummy if base_only is True, since no lora model is used
        image_data=None,
        extra_request_body=extra_request_body,
    )
    test_output = await request_func(request_func_input=test_input) # 执行测试请求
    if not test_output.success: # 如果测试失败
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else: # 测试通过
        print("Initial test run completed. Starting main benchmark run...")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests)) # 创建进度条

    benchmark_start_time = time.perf_counter() # 记录基准测试开始时间
    '''
    创建一个空列表，用来存放所有异步任务（每个任务代表一次请求）。
    '''
    tasks: List[asyncio.Task] = [] # 初始化异步任务列表
    '''
    异步循环，从 get_request 这个生成器里按设定速率依次获取请求数据（request）。
    '''
    request_idx = 0
    prompts_per_lora = len(input_requests) // NUM_LORAS
    async for request in get_request(input_requests, request_rate): # 按请求速率生成请求
        lora_path = "dummy"
        if not args.base_only:
            if args.lora_assignment_strategy == "random":
                lora_path = f"lora{random.randint(0, NUM_LORAS - 1)}"
            elif args.lora_assignment_strategy == "sequential":
                lora_index = min(request_idx // prompts_per_lora, NUM_LORAS - 1)
                lora_path = f"lora{lora_index}"

        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.output_len,
            lora_name=lora_path,
            image_data=None,
            extra_request_body=extra_request_body,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
        request_idx += 1
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks) # 并发执行所有请求
    """
    请求是随时并发发送的，收集结果是一次性完成的。
    """

    if pbar is not None:
        pbar.close() # 关闭进度条

    benchmark_duration = time.perf_counter() - benchmark_start_time # 计算测试总时长

    metrics, output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        backend=backend,
    ) # 计算各项评测指标

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Backend:", backend))
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

    # Determine output file name
    # 确定输出文件名
    if args.output_file:
        output_file_name = args.output_file
    else:
        now = datetime.now().strftime("%m%d")
        output_file_name = f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_{args.random_output_len}.jsonl"

    # Append results to a JSONL file
    # 追加结果到JSONL文件
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
    return result # 返回最终结果


def run_benchmark(args_: argparse.Namespace):
    global args
    args = args_

    # Set global environments
    set_ulimit()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set url
    if args.port is None:
        args.port = {
            "sglang": 30000,
        }.get(args.backend, 30000)

    # api_url = (
    #     f"{args.base_url}/v1/completions"
    #     if args.base_url
    #     else f"http://{args.host}:{args.port}/v1/completions"
    # )
    api_url = (
        f"{args.base_url}/generate"
        if args.base_url
        else f"http://{args.host}:{args.port}/generate"
    )

    print(f"{args}\n")

    # Read dataset
    # 读取数据集
    backend = args.backend
    model_id = args.model = LORA_PATH["base"]
    tokenizer_id = args.model

    tokenizer = get_tokenizer(tokenizer_id)

    # input_requests = sample_random_requests(
    #     input_len=args.random_input_len,
    #     output_len=args.random_output_len,
    #     num_prompts=args.num_prompts,
    #     range_ratio=args.random_range_ratio,
    #     tokenizer=tokenizer,
    #     dataset_path="/workspace/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json",
    # ) # 随机生成请求列表
    input_requests = sample_sharegpt_requests(
        dataset_path="/workspace/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json",
        num_requests=args.num_prompts,
        tokenizer=tokenizer,
    )

    return asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=args.request_rate,
            disable_tqdm=False,
            extra_request_body={},
        )
    ) # 执行基准测试并返回结果


def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


if __name__ == "__main__": # 主程序入口
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
        "--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0."
    )
    parser.add_argument(
        "--port",
        type=int,
        help="If not set, the default port is configured according to its default value for different LLM Inference Engines.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process. Default is 1000.",
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
    parser.add_argument(
        "--lora-assignment-strategy",
        type=str,
        default="sequential",
        choices=["random", "sequential"],
        help="Strategy to assign LoRA models to prompts. 'random' assigns a LoRA randomly to each prompt. "
        "'sequential' divides prompts evenly among LoRA models in sequence.",
    )
    args = parser.parse_args()
    run_benchmark(args)

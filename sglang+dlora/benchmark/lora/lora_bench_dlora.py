"""
LoRA Benchmark with dLoRA-style Engine Manager.
Usage:
    python benchmarks/lora_bench_multi.py \\
        --num-instances 2 \\
        --num-loras 4 \\
        --num-requests 100 \\
        --trace-name azure_v2
"""

import argparse
import asyncio
import json
import time
from typing import List, Tuple

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from sglang.srt.workload_generator.trace import Trace
from sglang.srt.managers.engine_manager import (
    create_engine_manager,
    ExecType,
    MigrationType
)


def _normalize_endpoint(url: str) -> str:
    """Normalize API endpoint URL."""
    if not url.startswith("http"):
        url = "http://" + url
    return url.rstrip("/")


async def async_request_generate(
    model_id: int,
    api_url: str,
    prompt_len: int,
    output_len: int,
    pbar: tqdm,
) -> Tuple[float, int, int, str]:
    """Send async request to SGLang endpoint."""
    
    # Generate dummy prompt
    prompt = "test " * prompt_len
    
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": output_len,
            "temperature": 0.0,
        },
        "lora_path": f"lora{model_id}",
    }
    
    headers = {"User-Agent": "Benchmark Client"}
    request_start_time = time.perf_counter()
    
    timeout = aiohttp.ClientTimeout(total=3600)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url=f"{api_url}/generate", 
                json=payload, 
                headers=headers
            ) as response:
                if response.status == 200:
                    output = await response.json()
                    request_end_time = time.perf_counter()
                    request_latency = request_end_time - request_start_time
                    
                    output_text = output.get("text", [""])[0] if isinstance(output.get("text"), list) else output.get("text", "")
                    output_token_count = len(output_text.split())
                    
                    if pbar:
                        pbar.update(1)
                    
                    return (request_latency, prompt_len, output_token_count, output_text)
                else:
                    error_text = await response.text()
                    if pbar:
                        pbar.update(1)
                    return (0, prompt_len, 0, f"Error {response.status}: {error_text}")
    except Exception as e:
        if pbar:
            pbar.update(1)
        return (0, prompt_len, 0, f"Exception: {str(e)}")


async def benchmark(
    engine_manager,
    workload: List[Tuple[int, float]],
    prompt_len: int = 100,
    output_len: int = 50,
    disable_tqdm: bool = False,
) -> Tuple[List[float], List[int], List[int]]:
    """Run benchmark with engine manager."""
    
    print(f"\n[Benchmark] Starting with {len(workload)} requests...")
    print(f"[Benchmark] Prompt length: {prompt_len}, Output length: {output_len}")
    
    pbar = None if disable_tqdm else tqdm(total=len(workload), desc="Processing requests")
    
    benchmark_start_time = time.perf_counter()
    tasks = []
    
    async def send_request(idx: int, model_id: int, delay: float):
        """Send a single request after delay."""
        if delay > 0:
            await asyncio.sleep(delay)
        
        # Select instance using engine manager
        request_id = f"req_{idx}"
        instance_id, instance_url = await engine_manager.select_instance(
            request_id, model_id
        )
        
        # Send request
        return await async_request_generate(
            model_id, instance_url, prompt_len, output_len, pbar
        )
    
    # Schedule all requests
    cumulative_time = 0.0
    for i, (model_id, interval) in enumerate(workload):
        task = asyncio.create_task(
            send_request(i, model_id, cumulative_time)
        )
        tasks.append(task)
        cumulative_time += interval
    
    # Wait for all requests
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    benchmark_end_time = time.perf_counter()
    benchmark_duration = benchmark_end_time - benchmark_start_time
    
    if pbar:
        pbar.close()
    
    # Process results
    latencies = []
    prompt_lens = []
    output_lens = []
    errors = 0
    
    for result in results:
        if isinstance(result, Exception):
            errors += 1
            continue
        
        latency, p_len, o_len, _ = result
        if latency > 0:
            latencies.append(latency)
            prompt_lens.append(p_len)
            output_lens.append(o_len)
        else:
            errors += 1
    
    print(f"\n[Benchmark] Completed in {benchmark_duration:.2f}s")
    print(f"[Benchmark] Successful: {len(latencies)}, Errors: {errors}, Total: {len(workload)}")
    
    return latencies, prompt_lens, output_lens


def run_benchmark(args: argparse.Namespace):
    """Main benchmark runner."""
    print("="*70)
    print("dLoRA-style LoRA Benchmark with Engine Manager")
    print("="*70)
    print(f"Instances: {args.num_instances}")
    print(f"LoRAs: {args.num_loras}")
    print(f"Requests: {args.num_requests}")
    print(f"Trace: {args.trace_name}")
    print(f"Migration Type: {args.migration_type}")
    print("="*70)
    
    async def run():
        # Create engine manager
        migration_type_map = {
            1: MigrationType.DISPATCH_ONLY,
            2: MigrationType.DISPATCH_MIG,
            3: MigrationType.PERIOD_MIG,
        }
        
        engine_manager = create_engine_manager(
            backend="sglang",
            num_instances=args.num_instances,
            num_loras=args.num_loras,
            base_url=args.host,
            base_port=args.port,
            exec_type=ExecType.LORA,
            migration_type=migration_type_map.get(args.migration_type, MigrationType.PERIOD_MIG),
            max_loras_per_batch=args.max_loras_per_batch,
            max_running_requests=args.max_running_requests,
            migration_interval=args.migration_interval,
            migration_req_threshold=args.migration_req_threshold,
        )
        
        # Start background loop
        if args.migration_type == 3:  # PERIOD_MIG
            engine_manager.start_background_loop()
        
        # Generate workload from trace
        print("\n[Trace] Generating workload...")
        trace = Trace(
            args.trace_name, 
            args.trace_dir, 
            args.start_time, 
            args.end_time,
            need_sort=args.need_sort
        )
        workload = trace.replay_to_workload(
            num_models=args.num_loras,
            num_reqs=args.num_requests,
            arrival_distribution="gamma",
            interval_minutes=args.interval_minutes,
            tot_rate=args.tot_rate,
            cv=args.cv,
            map_stride=args.map_stride,
        )
        
        print(f"[Trace] Generated {len(workload)} requests")
        
        # Run benchmark
        latencies, prompt_lens, output_lens = await benchmark(
            engine_manager=engine_manager,
            workload=workload,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
            disable_tqdm=args.disable_tqdm,
        )
        
        # Print statistics
        if latencies:
            print("\n" + "="*70)
            print("Benchmark Results")
            print("="*70)
            print(f"Total requests: {len(latencies)}")
            print(f"Mean latency: {np.mean(latencies):.3f}s")
            print(f"Median latency: {np.median(latencies):.3f}s")
            print(f"P50 latency: {np.percentile(latencies, 50):.3f}s")
            print(f"P90 latency: {np.percentile(latencies, 90):.3f}s")
            print(f"P95 latency: {np.percentile(latencies, 95):.3f}s")
            print(f"P99 latency: {np.percentile(latencies, 99):.3f}s")
            print(f"Min latency: {np.min(latencies):.3f}s")
            print(f"Max latency: {np.max(latencies):.3f}s")
            print(f"Throughput: {len(latencies) / (time.time() - benchmark_start_time):.2f} req/s")
            print("="*70)
        
        # Print engine stats
        print("\nEngine Manager Statistics:")
        stats = engine_manager.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Save results
        if args.save_result:
            result = {
                "config": vars(args),
                "latencies": latencies,
                "prompt_lens": prompt_lens,
                "output_lens": output_lens,
                "statistics": {
                    "count": len(latencies),
                    "mean": float(np.mean(latencies)) if latencies else 0,
                    "median": float(np.median(latencies)) if latencies else 0,
                    "p50": float(np.percentile(latencies, 50)) if latencies else 0,
                    "p90": float(np.percentile(latencies, 90)) if latencies else 0,
                    "p95": float(np.percentile(latencies, 95)) if latencies else 0,
                    "p99": float(np.percentile(latencies, 99)) if latencies else 0,
                    "min": float(np.min(latencies)) if latencies else 0,
                    "max": float(np.max(latencies)) if latencies else 0,
                },
                "engine_stats": stats,
            }
            
            with open(args.result_filename, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.result_filename}")
        
        # Stop engine manager
        engine_manager.stop_background_loop()
        print("\n[Benchmark] Done!")
    
    # Save start time for throughput calculation
    benchmark_start_time = time.time()
    asyncio.run(run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dLoRA-style LoRA Benchmark")
    
    # Backend configuration
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    
    # Instance configuration
    parser.add_argument("--num-instances", type=int, default=2)
    parser.add_argument("--num-loras", type=int, default=4)
    parser.add_argument("--max-loras-per-batch", type=int, default=8)
    parser.add_argument("--max-running-requests", type=int, default=16)
    
    # Migration configuration
    parser.add_argument("--migration-type", type=int, default=3,
                       choices=[1, 2, 3],
                       help="1: DISPATCH_ONLY, 2: DISPATCH_MIG, 3: PERIOD_MIG")
    parser.add_argument("--migration-interval", type=int, default=10)
    parser.add_argument("--migration-req-threshold", type=int, default=16)
    
    # Workload configuration
    parser.add_argument("--trace-name", type=str, default="azure_v2")
    parser.add_argument("--trace-dir", type=str, default="/workspace/datasets/maf2/")
    parser.add_argument("--start-time", type=str, default="0.0.0")
    parser.add_argument("--end-time", type=str, default="0.6.0")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--tot-rate", type=float, default=4.0)
    parser.add_argument("--cv", type=float, default=1.0)
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument("--map-stride", type=int, default=1)
    parser.add_argument("--need-sort", action="store_true")
    
    # Request configuration
    parser.add_argument("--prompt-len", type=int, default=100)
    parser.add_argument("--output-len", type=int, default=50)
    
    # Output configuration
    parser.add_argument("--save-result", action="store_true")
    parser.add_argument("--result-filename", type=str, default="dlora_benchmark_result.json")
    parser.add_argument("--disable-tqdm", action="store_true")
    
    args = parser.parse_args()
    run_benchmark(args)
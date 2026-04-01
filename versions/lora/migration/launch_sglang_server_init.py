# File: benchmark/lora/migration/launch_sglang_server.py
"""
Launch Unified Server with EngineManager for dynamic load balancing.  
Fully adapted from dLoRA to SGLang architecture.
With CPU and GPU utilization monitoring.
"""

import argparse
import subprocess
import time
import logging
import asyncio
import uvicorn
import aiohttp
import json
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sglang.srt. instances. lora_config_paths import LORA_PATH, NUM_LORAS

import psutil
import GPUtil


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# ========== 监控相关全局变量 ==========
monitoring_data:  List[Dict[str, Any]] = []
monitoring_lock = asyncio.Lock()
monitoring_task: Optional[asyncio. Task] = None
stop_monitoring_event = asyncio. Event()
current_phase = "warmup"  # warmup, benchmarking, idle
benchmark_count = 0
monitoring_output_file: Optional[str] = None


def build_sglang_cmd(args, port: int, gpu_id: int) -> str:
    """Build SGLang server launch command."""
    base_path = LORA_PATH["base"]
    
    cmd = f"python -m sglang.launch_server --model-path {base_path} "
    
    if not args.base_only:
        cmd += "--lora-paths "
        for i in range(NUM_LORAS):
            lora_name = f"lora{i}"
            lora_path = LORA_PATH[lora_name]
            cmd += f"{lora_name}={lora_path} "
    
    cmd += f"--max-loras-per-batch {args.max_loras_per_batch} "
    cmd += f"--max-running-requests {args.max_running_requests} "
    cmd += f"--lora-backend {args.lora_backend} "
    cmd += f"--tp-size {args.tp_size} "
    cmd += f"--host {args.host} --port {port} "
    
    if args.disable_custom_all_reduce:
        cmd += "--disable-custom-all-reduce "
    if args.enable_mscclpp:
        cmd += "--enable-mscclpp "
    
    return cmd. strip()


def launch_sglang_instances(args):
    """Launch multiple SGLang server instances."""
    procs = []
    instance_urls = []
    
    logger.info("=" * 86)
    logger.info("Launching SGLang Server Instances")
    logger.info("=" * 86)
    logger.info(f"Number of instances: {args.num_instances}")
    logger.info(f"Number of LoRAs: {NUM_LORAS}")
    logger.info(f"Starting port: {args.sglang_port}")
    logger.info("=" * 86)
    
    for i in range(args. num_instances):
        port = args.sglang_port + i
        gpu_id = i
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} " + build_sglang_cmd(args, port, gpu_id)
        
        logger.info(f"\n[Instance {i}] Launching on GPU {gpu_id}, Port {port}")
        logger.info(f"  Command: {cmd}")
        
        proc = subprocess.Popen(cmd, shell=True)
        procs.append(proc)
        
        instance_url = f"http://{args.host}:{port}"
        instance_urls.append(instance_url)
        
        time.sleep(16)
    
    logger.info("\n" + "=" * 86)
    logger.info("✓ All SGLang Instances Launched!")
    logger.info("=" * 86)
    logger.info("\nInstance URLs:")
    for i, url in enumerate(instance_urls):
        logger.info(f"  Instance {i}: {url}")
    logger.info("=" * 86)
    
    return procs, instance_urls


# ========== 监控功能 ==========
async def monitor_utilization_loop():
    """持续监控 CPU 和每个 GPU 的使用率"""
    global monitoring_data, current_phase
    
    logger.info("[Monitor] Starting utilization monitoring loop...")
    
    while not stop_monitoring_event.is_set():
        try:
            timestamp = datetime.now().isoformat()
            
            # CPU 使用率
            cpu_util = psutil.cpu_percent(interval=None)
            
            # 每个 GPU 的使用率
            gpu_utils = {}
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_utils[f"gpu_{gpu.id}"] = {
                        "utilization": gpu. load * 100,  # 转换为百分比
                        "memory_used_mb": gpu.memoryUsed,
                        "memory_total_mb": gpu.memoryTotal,
                        "memory_util": gpu. memoryUtil * 100,
                        "temperature": gpu.temperature,
                    }
            except Exception as e: 
                logger.warning(f"[Monitor] Failed to get GPU info: {e}")
            
            # 记录数据
            data_point = {
                "timestamp": timestamp,
                "phase": current_phase,
                "benchmark_count": benchmark_count,
                "cpu_utilization": cpu_util,
                "gpus": gpu_utils,
            }
            
            async with monitoring_lock:
                monitoring_data.append(data_point)
            
            # 每秒采样一次
            try:
                await asyncio.wait_for(stop_monitoring_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
                
        except Exception as e: 
            logger.error(f"[Monitor] Error in monitoring loop: {e}")
            await asyncio. sleep(10)
    
    logger.info("[Monitor] Monitoring loop stopped")


def save_monitoring_data():
    """保存监控数据到文件"""
    global monitoring_data, monitoring_output_file
    
    if not monitoring_output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        monitoring_output_file = f"monitoring_sglang_{timestamp}.jsonl"
    
    try: 
        with open(monitoring_output_file, "w") as f:
            for data_point in monitoring_data: 
                f.write(json. dumps(data_point) + "\n")
        
        logger.info(f"[Monitor] Saved {len(monitoring_data)} data points to {monitoring_output_file}")
        
        # 打印统计摘要
        phases = {}
        for dp in monitoring_data:
            phase = dp["phase"]
            if phase not in phases:
                phases[phase] = 0
            phases[phase] += 1
        
        logger.info(f"[Monitor] Data points by phase: {phases}")
        
    except Exception as e:
        logger.error(f"[Monitor] Failed to save monitoring data: {e}")


@app.post("/start_benchmark")
async def start_benchmark(request: Request):
    """通知服务器 benchmark 开始"""
    global current_phase, benchmark_count
    
    try:
        data = await request.json()
        bench_info = data.get("info", "")
        
        async with monitoring_lock:
            benchmark_count += 1
            current_phase = "benchmarking"
        
        logger.info("=" * 80)
        logger.info(f"[Monitor] 📊 BENCHMARK #{benchmark_count} STARTED")
        logger.info(f"[Monitor] Info: {bench_info}")
        logger.info("=" * 80)
        
        return JSONResponse({
            "status": "ok",
            "benchmark_count": benchmark_count,
            "message": f"Benchmark #{benchmark_count} started"
        })
        
    except Exception as e:
        logger. error(f"[Monitor] Error in start_benchmark: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/end_benchmark")
async def end_benchmark(request: Request):
    """通知服务器 benchmark 结束"""
    global current_phase, benchmark_count
    
    try:
        data = await request.json()
        bench_info = data.get("info", "")
        
        async with monitoring_lock:
            current_phase = "idle"
        
        logger.info("=" * 80)
        logger.info(f"[Monitor] ✅ BENCHMARK #{benchmark_count} COMPLETED")
        logger.info(f"[Monitor] Info: {bench_info}")
        logger.info(f"[Monitor] Switching to IDLE phase")
        logger.info("=" * 80)
        
        # 保存当前数据（追加模式）
        save_monitoring_data()
        
        return JSONResponse({
            "status": "ok",
            "benchmark_count": benchmark_count,
            "message": f"Benchmark #{benchmark_count} completed"
        })
        
    except Exception as e:
        logger. error(f"[Monitor] Error in end_benchmark: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/monitoring_status")
async def get_monitoring_status():
    """获取当前监控状态"""
    global current_phase, benchmark_count, monitoring_data
    
    async with monitoring_lock:
        data_count = len(monitoring_data)
    
    return JSONResponse({
        "current_phase": current_phase,
        "benchmark_count": benchmark_count,
        "data_points_collected": data_count,
        "monitoring_file": monitoring_output_file,
    })


@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy"})


def run_unified_server(args):
    """Run the unified server with monitoring."""
    global monitoring_task, monitoring_output_file
    
    # 设置监控输出文件
    if args.monitoring_file:
        monitoring_output_file = args.monitoring_file
    
    # Launch SGLang instances
    procs, instance_urls = launch_sglang_instances(args)
    
    @app.on_event("startup")
    async def startup():
        global monitoring_task
        
        # 启动监控任务
        logger.info("[Monitor] Starting utilization monitoring...")
        monitoring_task = asyncio.create_task(monitor_utilization_loop())
        
        logger.info(f"[Monitor] Monitoring output: {monitoring_output_file or 'auto-generated'}")
        logger.info("=" * 80)
        logger.info("Server ready!  Waiting for benchmark requests...")
        logger.info("Send POST to /start_benchmark to begin tracking")
        logger.info("=" * 80)
    
    @app.on_event("shutdown")
    async def shutdown():
        global monitoring_task
        
        # 停止监控
        if monitoring_task: 
            logger.info("[Monitor] Stopping monitoring...")
            stop_monitoring_event.set()
            await monitoring_task
            
            # 保存最终数据
            save_monitoring_data()
        
        # 终止 SGLang 实例
        for proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass
    
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Terminating...")


if __name__ == "__main__":
    parser = argparse. ArgumentParser(
        description="Launch Unified Server for SGLang Multi-Instance Serving"
    )
    
    # Unified server config
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-instances", type=int, default=2)
    parser.add_argument("--sglang-port", type=int, default=30001)
    
    # Monitoring config
    parser.add_argument(
        "--monitoring-file",
        type=str,
        default=None,
        help="Output file for monitoring data (default:  auto-generated with timestamp)",
    )
    
    # SGLang config
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--max-loras-per-batch", type=int, default=8)
    parser.add_argument("--max-running-requests", type=int, default=2)
    parser.add_argument("--lora-backend", type=str, default="csgmv")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--enable-mscclpp", action="store_true")
    
    args = parser.parse_args()
    run_unified_server(args)
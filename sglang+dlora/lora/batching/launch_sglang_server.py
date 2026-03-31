# File: benchmark/lora/batching/launch_sglang_server.py
"""
Launch SGLang Server Instances (No Resource Monitoring)
"""

import argparse
import subprocess
import time
import logging
from typing import List

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sglang.srt.instances.lora_config_paths import LORA_PATH, NUM_LORAS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()


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
        
    if args.enable_credit_schedule:
        cmd += "--enable-credit-schedule "

    return cmd.strip()


def launch_sglang_instances(args):
    """Launch multiple SGLang server instances."""
    procs = []
    instance_urls = []

    logger.info("=" * 86)
    logger.info("Launching SGLang Server Instances")
    logger.info("=" * 86)
    logger.info(f"Number of instances: {args.num_instances}")
    logger.info(f"Number of LoRAs:  {NUM_LORAS}")
    logger.info(f"Starting port: {args.sglang_port}")
    logger.info("=" * 86)

    for i in range(args.num_instances):
        port = args.sglang_port + i
        gpu_id = i
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} " + build_sglang_cmd(args, port, gpu_id)

        logger.info(f"\n[Instance {i}] 在 GPU {gpu_id} 上启动，端口 {port}")
        logger.info(f"  命令: {cmd}")

        proc = subprocess.Popen(cmd, shell=True)
        procs.append(proc)

        instance_url = f"http://{args.host}:{port}"
        instance_urls.append(instance_url)

        logger.info(f"  等待 {args.instance_delay} 秒...")
        time.sleep(args.instance_delay)

    logger.info("\n" + "=" * 86)
    logger.info("✓ 所有 SGLang 实例已启动!")
    logger.info("=" * 86)
    logger.info("\n实例 URLs:")
    for i, url in enumerate(instance_urls):
        logger.info(f"  实例 {i}:  {url}")
    logger.info("=" * 86)

    return procs, instance_urls


@app.get("/health")
async def health():
    """
    Health check endpoint
    """
    return JSONResponse({"status": "healthy"})


def run_unified_server(args):
    """
    Run the unified SGLang server (no resource monitoring).
    """
    # 1. run SGLang instances
    procs, instance_urls = launch_sglang_instances(args)

    # 2. wait for warmup time
    logger.info(f"等待实例就绪 ({args.warmup_time} 秒)...")
    time.sleep(args.warmup_time)

    logger.info("[Server] 实例就绪")

    # 3. setup FastAPI events
    @app.on_event("startup")
    async def startup():
        logger.info("=" * 80)
        logger.info("🚀 SGLang 服务器已就绪!")
        logger.info("=" * 80)

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("正在关闭服务器...")

        for i, proc in enumerate(procs):
            try:
                logger.info(f"终止实例 {i}...")
                proc.terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"强制终止实例 {i}")
                proc.kill()
            except Exception as e:
                logger.error(f"终止实例 {i} 失败: {e}")

        logger.info("服务器已关闭")

    # 4. run Uvicorn server
    try:
        logger.info(f"\n启动 FastAPI 服务器在 {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在退出...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch SGLang Multi-Instance Server (No Resource Monitoring)"
    )

    # Server
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-instances", type=int, default=2)
    parser.add_argument("--sglang-port", type=int, default=30001)

    # Instance
    parser.add_argument("--instance-delay", type=float, default=16.0,
                        help="每个实例启动后的等待时间 (秒)")
    parser.add_argument("--warmup-time", type=float, default=10.0,
                        help="所有实例启动后的额外等待时间 (秒)")

    # SGLang
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--max-loras-per-batch", type=int, default=8)
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument("--lora-backend", type=str, default="csgmv")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--enable-mscclpp", action="store_true")
    
    # Batching
    parser.add_argument("--enable-credit-schedule", action="store_true")

    args = parser.parse_args()
    run_unified_server(args)
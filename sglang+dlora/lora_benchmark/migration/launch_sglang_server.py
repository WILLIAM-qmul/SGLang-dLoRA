# File: benchmark/lora/migration/launch_sglang_server.py
"""
Launch Unified Server with EngineManager for dynamic load balancing. 
Fully adapted from dLoRA to SGLang architecture.
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

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sglang.srt.instances.lora_config_paths import LORA_PATH, NUM_LORAS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


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
    
    return cmd.strip()


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
    
    for i in range(args.num_instances):
        port = args.sglang_port + i
        gpu_id = i
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} " + build_sglang_cmd(args, port, gpu_id)
        
        logger.info(f"\n[Instance {i}] Launching on GPU {gpu_id}, Port {port}")
        logger.info(f" Command: {cmd}")
        
        proc = subprocess.Popen(cmd, shell=True)
        procs.append(proc)
        
        instance_url = f"http://{args.host}:{port}"
        instance_urls.append(instance_url)
        
        time.sleep(16)
    
    logger.info("\n" + "=" * 86)
    logger. info("✓ All SGLang Instances Launched!")
    logger.info("=" * 86)
    logger.info("\nInstance URLs:")
    for i, url in enumerate(instance_urls):
        logger.info(f" Instance {i}: {url}")
    logger.info("=" * 86)
    
    return procs, instance_urls


def run_unified_server(args):
    """Run the unified server with EngineManager."""
    global manager
    
    # Launch SGLang instances
    procs, instance_urls = launch_sglang_instances(args)
    input("All instances launched. Press Ctrl+C to terminate...\n")


if __name__ == "__main__":
    parser = argparse. ArgumentParser(
        description="Launch Unified Server for SGLang Multi-Instance Serving"
    )
    
    # Unified server config
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-instances", type=int, default=2)
    parser.add_argument("--sglang-port", type=int, default=30001)
    
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
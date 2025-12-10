# File: benchmark/lora/launch_unified_server.py
"""
Launch Unified Server with EngineManager for dynamic load balancing. 
Fully adapted from dLoRA to SGLang architecture.
"""

import argparse
import subprocess
import time
import asyncio
import uvicorn
import aiohttp
import json
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sglang.srt.instances.engine_manager import EngineManager, MigrationType


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# LoRA paths configuration
NUM_LORAS = 4
LORA_PATH = {
    "base": "/workspace/models/Llama-2-7b-hf",
    "lora0": "/workspace/models/llama-2-7b-chat-lora-adaptor",
    "lora1": "/workspace/models/llama-2-7b-LORA-data-analyst",
    "lora2": "/workspace/models/llama2-stable-7b-lora",
    "lora3": "/workspace/models/llava-llama-2-7b-chat-lightning-lora-preview",
}

app = FastAPI()
manager: EngineManager = None


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
        logger. info(f"  Command: {cmd}")
        
        proc = subprocess.Popen(cmd, shell=True)
        procs.append(proc)
        
        instance_url = f"http://{args.host}:{port}"
        instance_urls.append(instance_url)
        
        time.sleep(3)
    
    logger.info("\n" + "=" * 86)
    logger. info("✓ All SGLang Instances Launched!")
    logger.info("=" * 86)
    logger.info("\nInstance URLs:")
    for i, url in enumerate(instance_urls):
        logger.info(f"  Instance {i}: {url}")
    logger.info("=" * 86)
    
    return procs, instance_urls


@app.post("/generate")
async def generate(request: Request):
    """
    Generate completion with dynamic load balancing.
    Routes requests using EngineManager.
    """
    request_dict = await request.json()
    text = request_dict.get("text", "")
    sampling_params = request_dict.get("sampling_params", {})
    lora_path = request_dict.get("lora_path", "lora0")
    
    # Extract model ID
    try:
        model_id = int(lora_path.replace("lora", "")) if lora_path. startswith("lora") else 0
    except:
        model_id = 0
    
    # Generate unique request ID
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    
    # Select instance using manager
    instance_id, instance_url = await manager.select_engine(request_id, model_id)
    
    # Forward request
    target_url = f"{instance_url}/generate"
    
    async def stream_from_backend():
        """Stream response from backend instance."""
        try:
            timeout = aiohttp.ClientTimeout(total=300, connect=10, sock_read=300)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {
                    "text": text,
                    "sampling_params": sampling_params,
                    "lora_path": lora_path,
                    "stream": True,
                }
                
                async with session.post(target_url, json=payload) as resp:
                    if resp.status == 200:
                        async for chunk in resp.content.iter_any():
                            yield chunk
                    else:
                        error_msg = {"error": f"Backend returned {resp.status}"}
                        yield (json.dumps(error_msg) + "\n").encode("utf-8")
        except Exception as e:
            error_msg = {"error": str(e)}
            yield (json. dumps(error_msg) + "\n").encode("utf-8")
        finally:
            await manager.complete_request(request_id)
    
    return StreamingResponse(
        stream_from_backend(),
        media_type="text/event-stream"
    )


@app.get("/get_manager_stats")
async def get_manager_stats():
    """Get manager statistics."""
    return JSONResponse(manager.get_stats())


@app.post("/reset_manager_stats")
async def reset_manager_stats():
    """Reset manager statistics."""
    await manager.reset_stats()
    return JSONResponse({"status": "ok"})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "num_instances": manager.num_instances,
        "migration_enabled": manager.is_running(),
    })


async def shutdown_handler():
    """Cleanup on shutdown"""
    if manager:
        await manager.close()


def run_unified_server(args):
    """Run the unified server with EngineManager."""
    global manager
    
    # Launch SGLang instances
    procs, instance_urls = launch_sglang_instances(args)
    
    # Wait for instances to be ready
    logger.info("\nWaiting for instances to be ready...")
    time.sleep(20)
    
    # Create engine manager
    migration_type = MigrationType(args.migration_type)
    
    manager = EngineManager(
        num_instances=args.num_instances,
        num_models=NUM_LORAS,
        instance_urls=instance_urls,
        migration_type=migration_type,
        migration_interval=args.migration_interval,
        lora_capacity_per_engine=args.max_loras_per_batch,
        max_running_requests=args.max_running_requests,
    )
    
    @app.on_event("startup")
    async def start_manager_background_loop():
        if migration_type != MigrationType.DISPATCH_ONLY:
            manager.start_background_loop()
    
    @app.on_event("shutdown")
    async def shutdown():
        await shutdown_handler()
        for proc in procs:
            proc.terminate()
    
    logger.info(f"\n[Unified Server] Starting on port {args.port}")
    logger.info(f"[Unified Server] Migration type: {migration_type.name}")
    
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("\n\nTerminating...")


if __name__ == "__main__":
    parser = argparse. ArgumentParser(
        description="Launch Unified Server for SGLang Multi-Instance Serving"
    )
    
    # Unified server config
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-instances", type=int, default=2)
    parser.add_argument("--sglang-port", type=int, default=30001)
    
    # Migration config
    parser. add_argument("--migration-type", type=int, default=3,
                       help="1=DISPATCH_ONLY, 2=DISPATCH_MIG, 3=PERIOD_MIG")
    parser.add_argument("--migration-interval", type=float, default=10.0)
    
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
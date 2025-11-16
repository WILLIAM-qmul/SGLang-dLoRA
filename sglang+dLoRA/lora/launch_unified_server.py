"""
Launch Unified dLoRA-style Server with Engine Manager
"""

import argparse
import asyncio
import subprocess
import time
import os
import signal
import sys
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from sglang.srt.instances.engine_manager_unified import (
    UnifiedEngineManager,
    ExecType,
    MigrationType,
)

# Global manager instance
engine_manager: UnifiedEngineManager = None

# Model paths
NUM_LORAS = 4
LORA_PATH = { # 定义模型路径字典
    "base": "/workspace/models/Llama-2-7b-hf",
    # "lora": "winddude/wizardLM-LlaMA-LoRA-7B",
    # "lora": "/workspace/models/fingpt-mt_llama2-7b_lora",
    "lora0": "/workspace/models/llama-2-7b-chat-lora-adaptor",
    "lora1": "/workspace/models/llama-2-7b-LORA-data-analyst",
    "lora2": "/workspace/models/llama2-stable-7b-lora",
    "lora3": "/workspace/models/llava-llama-2-7b-chat-lightning-lora-preview",
    "lora4": "/workspace/models/MUFFIN-Llama2-lora-7B",
}


def build_instance_cmd(args, port: int) -> str:
    """Build command to launch a single SGLang instance."""
    base_path = LORA_PATH["base"]
    # lora_path = LORA_PATH["lora"]

    if args.base_only:
        cmd = f"python3 -m sglang.launch_server --model-path {base_path} "
    else:
        cmd = f"python3 -m sglang.launch_server --model-path {base_path} --lora-paths "
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


def launch_sglang_instances(args) -> List[subprocess.Popen]:
    """Launch multiple SGLang server instances."""
    procs = []
    cuda_devices = list(range(args.num_instances))
    
    print("="*70)
    print("Launching SGLang Server Instances")
    print("="*70)
    print(f"Number of instances: {args.num_instances}")
    print(f"Number of LoRAs: {NUM_LORAS}")
    print("="*70)
    
    for i in range(args.num_instances):
        port = args.sglang_port + i
        cuda_device = cuda_devices[i]
        cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} " + build_instance_cmd(args, port)
        
        print(f"\n[Instance {i}] Launching on GPU {cuda_device}, Port {port}")
        print(f"  Command: {cmd}")
        
        proc = subprocess.Popen(cmd, shell=True)
        procs.append(proc)
        time.sleep(3)  # Stagger launches
    
    print("\n" + "="*70)
    print("✓ All SGLang Instances Launched")
    print("="*70)
    
    return procs


# FastAPI app for the unified server
app = FastAPI(title="dLoRA-style Unified Server")


@app.on_event("startup")
async def startup_event():
    """Initialize the engine manager on startup."""
    global engine_manager
    
    # Get config from app state (set by main function)
    args = app.state.args
    
    # Build instance URLs
    instance_urls = [
        f"http://{args.host}:{args.sglang_port + i}"
        for i in range(args.num_instances)
    ]
    
    # Map migration type
    mig_map = {
        "dispatch_only": MigrationType.DISPATCH_ONLY,
        "dispatch_mig": MigrationType.DISPATCH_MIG,
        "period_mig": MigrationType.PERIOD_MIG,
        "1": MigrationType.DISPATCH_ONLY,
        "2": MigrationType.DISPATCH_MIG,
        "3": MigrationType.PERIOD_MIG,
    }
    mig_type = mig_map.get(str(args.migration_type), MigrationType.PERIOD_MIG)
    
    # Create engine manager
    engine_manager = UnifiedEngineManager(
        exec_type=ExecType.LORA,
        migration_type=mig_type,
        num_instances=args.num_instances,
        num_loras=NUM_LORAS,
        instance_urls=instance_urls,
        max_loras_per_batch=args.max_loras_per_batch,
        max_running_requests=args.max_running_requests,
        migration_interval=args.migration_interval,
        migration_req_threshold=args.migration_req_threshold,
    )
    
    # Initialize manager
    await engine_manager.initialize()
    
    # Start periodic migration if needed
    if mig_type == MigrationType.PERIOD_MIG:
        engine_manager.start_background_loop()
    
    print(f"\n[UnifiedServer] Engine Manager initialized and ready!")
    print(f"  Stats: {engine_manager.get_stats()}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global engine_manager
    
    if engine_manager:
        engine_manager.stop_background_loop()
        await engine_manager.close()
    
    print("[UnifiedServer] Engine Manager stopped")


@app.post("/generate")
async def generate(request: Request):
    """
    Unified /generate endpoint that routes requests via Engine Manager.
    """
    global engine_manager
    
    request_dict = await request.json()
    prompt = request_dict.pop("prompt", request_dict.pop("text", None))
    model_id = int(request_dict.pop("model_id", 0))
    stream = request_dict.pop("stream", False)
    
    # Get lora_path if provided, otherwise use model_id
    lora_path = request_dict.pop("lora_path", None)
    if lora_path:
        # Extract lora_id from lora_path (e.g., "lora0" -> 0)
        if lora_path.startswith("lora"):
            model_id = int(lora_path.replace("lora", ""))
    
    # Select instance via Engine Manager
    instance_id, instance_url = await engine_manager.select_instance(
        request_id=f"req_{time.time()}",
        lora_id=model_id
    )
    
    print(f"[UnifiedServer] Routing request to instance {instance_id} ({instance_url})")
    
    # Prepare request for backend instance
    backend_payload = {
        "text": prompt,
        "sampling_params": request_dict.get("sampling_params", {}),
        "lora_path": f"lora{model_id}",
    }
    
    # Forward request to selected instance
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{instance_url}/generate",
            json=backend_payload
        ) as resp:
            if stream:
                # Stream response
                async def stream_results():
                    async for chunk in resp.content:
                        yield chunk
                
                return StreamingResponse(stream_results(), media_type="text/event-stream")
            else:
                # Non-streaming response
                result = await resp.json()
                return JSONResponse(result)


@app.get("/get_manager_stats")
async def get_manager_stats():
    """Get Engine Manager statistics."""
    global engine_manager
    return JSONResponse(engine_manager.get_stats())


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})


def main():
    parser = argparse.ArgumentParser(
        description="Launch dLoRA-style Unified Server with Engine Manager"
    )
    
    # SGLang instance configuration
    parser.add_argument("--base-only", action="store_true",
                       help="Launch base model only without LoRA")
    parser.add_argument("--max-loras-per-batch", type=int, default=8,
                       help="Maximum LoRA adapters per batch")
    parser.add_argument("--max-running-requests", type=int, default=16,
                       help="Maximum concurrent requests per instance")
    parser.add_argument("--lora-backend", type=str, default="csgmv",
                       help="LoRA backend implementation")
    parser.add_argument("--tp-size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--disable-custom-all-reduce", action="store_true",
                       help="Disable custom all-reduce")
    parser.add_argument("--enable-mscclpp", action="store_true",
                       help="Enable MSCCL++")
    
    # Multi-instance configuration
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host address")
    parser.add_argument("--sglang-port", type=int, default=30001,
                       help="Starting port for SGLang instances")
    parser.add_argument("--num-instances", type=int, default=2,
                       help="Number of SGLang instances")
    
    # Unified server configuration
    parser.add_argument("--port", type=int, default=8000,
                       help="Port for the unified server")
    
    # Engine Manager configuration
    parser.add_argument("--migration-type", type=str, default="period_mig",
                       choices=["dispatch_only", "dispatch_mig", "period_mig", "1", "2", "3"],
                       help="Migration strategy")
    parser.add_argument("--migration-interval", type=int, default=10,
                       help="Migration interval in seconds")
    parser.add_argument("--migration-req-threshold", type=int, default=16,
                       help="Request threshold for migration")
    
    args = parser.parse_args()
    
    # Launch SGLang instances
    sglang_procs = launch_sglang_instances(args)
    
    # Store args in app state
    app.state.args = args
    
    # Setup signal handlers to clean up subprocesses
    def signal_handler(signum, frame):
        print("\n\nTerminating all processes...")
        for proc in sglang_procs:
            proc.terminate()
        for proc in sglang_procs:
            proc.wait()
        print("✓ All processes terminated.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Launch unified server
    print(f"\n{'='*70}")
    print(f"Launching Unified Server on http://{args.host}:{args.port}")
    print(f"{'='*70}\n")
    
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
    finally:
        # Clean up on exit
        for proc in sglang_procs:
            proc.terminate()
        for proc in sglang_procs:
            proc.wait()


if __name__ == "__main__":
    main()
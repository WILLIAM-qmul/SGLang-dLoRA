"""
Launch Unified Server with UnifiedEngineManager.
Manages multiple SGLang instances on port 8000.
"""

import argparse
import subprocess
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import asyncio
from typing import AsyncGenerator
import aiohttp
import json

from sglang.srt.instances.engine_manager_unified import create_engine_manager_unified

# Constants
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

app = FastAPI()
manager = None


def build_sglang_cmd(args, port: int, gpu_id: int) -> str:
    """Build SGLang server launch command."""
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


def launch_sglang_instances(args):
    """Launch multiple SGLang server instances."""
    procs = []
    instance_urls = []
    
    print("="*70)
    print("Launching SGLang Server Instances")
    print("="*70)
    print(f"Number of instances: {args.num_instances}")
    print(f"Number of LoRAs: {NUM_LORAS}")
    print(f"Starting port: {args.sglang_port}")
    print("="*70)
    
    for i in range(args.num_instances):
        port = args.sglang_port + i
        gpu_id = i
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} " + build_sglang_cmd(args, port, gpu_id)
        
        print(f"\n[Instance {i}] Launching on GPU {gpu_id}, Port {port}")
        print(f"  Command: {cmd}")
        
        proc = subprocess.Popen(cmd, shell=True)
        procs.append(proc)
        
        instance_url = f"http://{args.host}:{port}"
        instance_urls.append(instance_url)
        
        time.sleep(3)  # Stagger launches
    
    print("\n" + "="*70)
    print("✓ All SGLang Instances Launched!")
    print("="*70)
    print("\nInstance URLs:")
    for i, url in enumerate(instance_urls):
        print(f"  Instance {i}: {url}")
    print("="*70)
    
    return procs, instance_urls


@app.post("/generate")
async def generate(request: Request):
    """
    Generate completion with load balancing.
    
    Request format:
    {
        "text": "prompt text",
        "sampling_params": {"max_new_tokens": 128},
        "lora_path": "lora0" (optional)
    }
    """
    request_dict = await request.json()
    text = request_dict.get("text", "")
    sampling_params = request_dict.get("sampling_params", {})
    lora_path = request_dict.get("lora_path", "lora0")
    
    # Extract LoRA ID from lora_path (e.g., "lora0" -> 0)
    try:
        lora_id = int(lora_path.replace("lora", "")) if lora_path.startswith("lora") else 0
    except:
        lora_id = 0
    
    # Generate unique request ID
    import uuid
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    
    # Select instance using manager
    instance_id, instance_url = await manager.select_instance(request_id, lora_id)
    
    # Forward request to selected instance
    target_url = f"{instance_url}/generate"
    
    # Create streaming response
    async def stream_from_backend():
        """Stream response from backend instance."""
        try:
            # Use a longer timeout for streaming
            timeout = aiohttp.ClientTimeout(total=300, connect=10, sock_read=300)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {
                    "text": text,
                    "sampling_params": sampling_params,
                    "lora_path": lora_path
                }
                
                async with session.post(target_url, json=payload) as resp:
                    if resp.status == 200:
                        # Stream chunks from backend
                        async for chunk in resp.content.iter_any():
                            yield chunk
                    else:
                        # Return error response
                        error_msg = {"error": f"Backend returned {resp.status}"}
                        yield (json.dumps(error_msg) + "\n").encode("utf-8")
        except Exception as e:
            # Return error response
            error_msg = {"error": str(e)}
            yield (json.dumps(error_msg) + "\n").encode("utf-8")
        finally:
            # Mark request as completed
            await manager.complete_request(request_id)
    
    return StreamingResponse(
        stream_from_backend(),
        media_type="text/event-stream"
    )


@app.get("/get_manager_stats")
async def get_manager_stats():
    """Get manager statistics."""
    return JSONResponse(manager.get_stats())


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({"status": "healthy", "num_instances": manager.num_instances})


def run_unified_server(args):
    """Run the unified server."""
    global manager
    
    # Launch SGLang instances
    procs, instance_urls = launch_sglang_instances(args)
    
    # Wait for instances to be ready
    print("\nWaiting for instances to be ready...")
    time.sleep(15)
    
    # Create unified manager
    manager = create_engine_manager_unified(
        num_instances=args.num_instances,
        num_loras=NUM_LORAS,
        instance_urls=instance_urls
    )
    
    print(f"\n[Unified Server] Starting on port {args.port}")
    print(f"[Unified Server] Routing strategy: Least-Loaded (动态负载均衡)")
    
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            timeout_keep_alive=60
        )
    except KeyboardInterrupt:
        print("\n\nTerminating all instances...")
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait()
        print("✓ All instances terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Unified Server for SGLang Multi-Instance Serving"
    )
    
    # Unified server config
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host address for unified server")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port for unified server")
    parser.add_argument("--num-instances", type=int, default=2,
                       help="Number of SGLang instances to launch")
    parser.add_argument("--sglang-port", type=int, default=30001,
                       help="Starting port for SGLang instances (port+i for instance i)")
    
    # SGLang server config
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
    
    args = parser.parse_args()
    run_unified_server(args)
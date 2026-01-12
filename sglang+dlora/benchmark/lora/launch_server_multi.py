"""Launch server with Engine Manager integration."""
import argparse
import os

import subprocess
import time

NUM_LORAS = 4 # 定义Lora模型数量
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


"""Launch multiple SGLang server instances for dLoRA-style serving."""
def build_cmd(args, port: int) -> str:
    """Build server launch command."""
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


def launch_server(args):
    """Launch multiple server instances."""
    procs = []
    cuda_devices = list(range(args.num_instances))
    
    print("="*70)
    print("Launching SGLang Server Instances for dLoRA-style Serving")
    print("="*70)
    print(f"Backend: {args.backend}")
    print(f"Number of instances: {args.num_instances}")
    print(f"Number of LoRAs: {NUM_LORAS}")
    print(f"Max LoRAs per batch: {args.max_loras_per_batch}")
    print(f"Max running requests: {args.max_running_requests}")
    print("="*70)
    
    # Launch instances
    for i in range(args.num_instances):
        port = args.port + i
        cuda_device = cuda_devices[i]
        cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} " + build_cmd(args, port)
        
        print(f"\n[Instance {i}] Launching on GPU {cuda_device}, Port {port}")
        print(f"  Command: {cmd}")
        
        proc = subprocess.Popen(cmd, shell=True)
        procs.append(proc)
        time.sleep(3)  # Stagger launches to avoid conflicts
    
    # Print instance URLs
    worker_urls = [
        f"http://{args.host}:{args.port + i}" 
        for i in range(args.num_instances)
    ]
    
    print("\n" + "="*70)
    print("✓ All Server Instances Launched Successfully!")
    print("="*70)
    print("\nInstance URLs:")
    for i, url in enumerate(worker_urls):
        print(f"  Instance {i}: {url}")
    
    # Wait for processes
    try:
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        print("\n\nTerminating all server instances...")
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait()
        print("✓ All instances terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Multiple SGLang Servers for dLoRA-style Serving"
    )
    
    # Server configuration
    parser.add_argument("--backend", type=str, default="sglang", 
                       choices=["sglang", "dlora"])
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
                       help="Host address for instances")
    parser.add_argument("--port", type=int, default=30001,
                       help="Starting port (each instance uses port+i)")
    parser.add_argument("--num-instances", type=int, default=2,
                       help="Number of server instances to launch")
    
    args = parser.parse_args()
    
    launch_server(args)
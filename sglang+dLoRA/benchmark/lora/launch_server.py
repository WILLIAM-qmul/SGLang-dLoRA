import argparse
import os

import subprocess

NUM_LORAS = 2 # 定义Lora模型数量
LORA_PATH = { # 定义模型路径字典
    "base": "/workspace/models/Llama-2-7b-hf",
    # "lora": "winddude/wizardLM-LlaMA-LoRA-7B",
    # "lora": "/workspace/models/fingpt-mt_llama2-7b_lora",
    "lora": "/workspace/models/llama-2-7b-chat-lora-adaptor",
    "lora": "/workspace/models/llama-2-7b-LORA-data-analyst",
}


'''
multi-instance launch script for LoRA benchmark server
'''
def build_cmd(args, port: int) -> str:
    base_path = LORA_PATH["base"]
    lora_path = LORA_PATH["lora"]

    if args.base_only:  # 如果指定只使用基础模型
        cmd = f"python3 -m sglang.launch_server --model-path {base_path} "
    else:  # 否则加载Lora模型
        cmd = f"python3 -m sglang.launch_server --model-path {base_path} --lora-paths "
        for i in range(NUM_LORAS):
            lora_name = f"lora{i}"
            cmd += f"{lora_name}={lora_path} "

    # cmd += f"--disable-radix "
    # cmd += f"--dynamic-batching "
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
    procs = []
    cuda_devices = [0, 1]  # 你的实际 GPU 号
    assert args.num_instances <= len(cuda_devices), "实例数不能超过可用GPU数"
    # 循环启动多个模型实例
    for i in range(args.num_instances):
        port = args.port + i
        cuda_device = cuda_devices[i]
        # 为每个实例构建启动命令
        cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} " + build_cmd(args, port)
        print(f"[Instance {i}] Launching server on port {port} (CUDA_VISIBLE_DEVICES={cuda_device})...")
        print(f"  > {cmd}")
        # 使用 subprocess.Popen 在后台启动服务进程
        proc = subprocess.Popen(cmd, shell=True)
        procs.append(proc)

    # 打印启动路由器的建议命令
    worker_urls = " ".join([f"http://{args.host}:{args.port + i}" for i in range(args.num_instances)])
    print("\n" + "="*50)
    print("All server instances are launching.")
    print("Please start the router in a new terminal with the following command:")
    print(f"  python3 -m sglang_router.launch_router --port {args.router_port} --worker-urls {worker_urls}")
    print("="*50 + "\n")

    # 等待所有子进程结束 (可以通过 Ctrl+C 中断)
    try:
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        print("\nTerminating all server instances...")
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait()
        print("All instances terminated.")


'''
single instance launch script for LoRA benchmark server
'''
# def launch_server(args):
#     base_path = LORA_PATH["base"]
#     lora_path = LORA_PATH["lora"]

#     if args.base_only: # 如果指定只使用基础模型
#         cmd = f"python3 -m sglang.launch_server --model {base_path} "
#     else: # 否则加载Lora模型
#         cmd = f"python3 -m sglang.launch_server --model {base_path} --lora-paths "
#         for i in range(NUM_LORAS):
#             lora_name = f"lora{i}"
#             cmd += f"{lora_name}={lora_path} "
#     # cmd += f"--disable-radix "
#     # cmd += f"--dynamic-batching "
#     cmd += f"--max-loras-per-batch {args.max_loras_per_batch} "
#     cmd += f"--max-running-requests {args.max_running_requests} "
#     cmd += f"--lora-backend {args.lora_backend} "
#     cmd += f"--tp-size {args.tp_size} "
#     if args.disable_custom_all_reduce:
#         cmd += "--disable-custom-all-reduce"
#     if args.enable_mscclpp:
#         cmd += "--enable-mscclpp"
#     print(cmd)
#     os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( # 添加是否只加载基础模型参数
        "--base-only",
        action="store_true",
    )
    # parser.add_argument(
    #     "--num-loras",
    #     type=int,
    #     default=2,
    #     help="LoRA模型数量"
    # )
    parser.add_argument( # 添加每批最大Lora数量参数
        "--max-loras-per-batch",
        type=int,
        default=8,
    )
    parser.add_argument( # 添加最大并发请求数参数
        "--max-running-requests",
        type=int,
        default=8,
    )
    parser.add_argument( # 添加Lora后端类型参数
        "--lora-backend",
        type=str,
        # default="triton",
        default = "csgmv"
    )
    parser.add_argument( # 添加张量并行大小参数
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for distributed inference",
    )
    # disable_custom_all_reduce
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Disable custom all reduce when device does not support p2p communication",
    )
    parser.add_argument(
        "--enable-mscclpp",
        action="store_true",
        help="Enable using mscclpp for small messages for all-reduce kernel and fall back to NCCL.",
    )
    # multi-instance args
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for the server instances.")
    parser.add_argument("--port", type=int, default=30001, help="Starting port for the first server instance.")
    parser.add_argument("--num-instances", type=int, default=2, help="Number of identical server instances to launch.")
    parser.add_argument("--router-port", type=int, default=30000, help="The port for the central router.")
    args = parser.parse_args()

    launch_server(args)

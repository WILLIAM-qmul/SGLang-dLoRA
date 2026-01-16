"""
简化版 SGLang Server 启动脚本
只启动单个 SGLang 实例用于 LoRA 推理测试
"""

import argparse
import subprocess
import time
import logging

from sglang.srt. instances. lora_config_paths import LORA_PATH, NUM_LORAS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def build_sglang_cmd(args) -> str:
    """构建 SGLang 服务器启动命令"""
    base_path = LORA_PATH["base"]
    
    cmd = f"python -m sglang.launch_server --model-path {base_path} "
    
    # 添加 LoRA 路径
    if not args.base_only:
        cmd += "--lora-paths "
        for i in range(NUM_LORAS):
            lora_name = f"lora{i}"
            lora_path = LORA_PATH[lora_name]
            cmd += f"{lora_name}={lora_path} "
    
    # 基本配置
    cmd += f"--max-loras-per-batch {args.max_loras_per_batch} "
    cmd += f"--max-running-requests {args.max_running_requests} "
    cmd += f"--lora-backend {args.lora_backend} "
    cmd += f"--tp-size {args.tp_size} "
    cmd += f"--host {args.host} --port {args.port} "
    
    # 可选配置
    if args. disable_custom_all_reduce:
        cmd += "--disable-custom-all-reduce "
    if args.enable_mscclpp:
        cmd += "--enable-mscclpp "
    
    return cmd. strip()


def launch_sglang_server(args):
    """启动单个 SGLang 服务器实例"""
    
    logger.info("=" * 86)
    logger.info("启动 SGLang 服务器")
    logger.info("=" * 86)
    logger.info(f"模型路径: {LORA_PATH['base']}")
    logger.info(f"LoRA 数量: {NUM_LORAS if not args.base_only else 0}")
    logger.info(f"服务器地址: {args.host}:{args.port}")
    logger.info(f"GPU 设备: {args.gpu_id}")
    logger.info(f"LoRA 后端: {args.lora_backend}")
    logger.info(f"最大批次 LoRA 数: {args.max_loras_per_batch}")
    logger.info(f"最大运行请求数: {args.max_running_requests}")
    logger.info("=" * 86)
    
    # 构建启动命令
    cmd = f"CUDA_VISIBLE_DEVICES={args.gpu_id} " + build_sglang_cmd(args)
    
    logger.info(f"\n执行命令:")
    logger.info(f"  {cmd}")
    logger.info("\n")
    
    # 启动服务器
    proc = subprocess.Popen(cmd, shell=True)
    
    logger.info("✓ SGLang 服务器已启动!")
    logger.info("=" * 86)
    logger.info(f"\n服务器 URL: http://{args.host}:{args.port}")
    logger.info(f"生成接口:  http://{args.host}:{args.port}/generate")
    logger.info("\n按 Ctrl+C 停止服务器.. .\n")
    logger.info("=" * 86)
    
    return proc


def main():
    parser = argparse.ArgumentParser(
        description="启动单个 SGLang 服务器实例用于 LoRA 推理测试"
    )
    
    # 服务器配置
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="服务器监听地址")
    parser.add_argument("--port", type=int, default=30001,
                       help="服务器监听端口")
    parser.add_argument("--gpu-id", type=int, default=0,
                       help="使用的 GPU 设备 ID")
    
    # LoRA 配置
    parser.add_argument("--base-only", action="store_true",
                       help="只使用基础模型，不加载 LoRA")
    parser.add_argument("--max-loras-per-batch", type=int, default=8,
                       help="单批次最大 LoRA 数量")
    parser.add_argument("--max-running-requests", type=int, default=16,
                       help="最大并发请求数")
    parser.add_argument("--lora-backend", type=str, default="triton",
                       choices=["triton", "csgmv"],
                       help="LoRA 后端实现")
    
    # 模型配置
    parser.add_argument("--tp-size", type=int, default=1,
                       help="张量并行大小")
    parser.add_argument("--disable-custom-all-reduce", action="store_true",
                       help="禁用自定义 all-reduce")
    parser.add_argument("--enable-mscclpp", action="store_true",
                       help="启用 MSCCL++")
    
    args = parser.parse_args()
    
    try:
        proc = launch_sglang_server(args)
        
        # 等待用户中断
        proc.wait()
        
    except KeyboardInterrupt:
        logger.info("\n\n收到中断信号，正在停止服务器...")
        if proc: 
            proc.terminate()
            proc.wait()
        logger.info("✓ 服务器已停止")
    except Exception as e:
        logger.error(f"启动失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__": 
    exit(main())
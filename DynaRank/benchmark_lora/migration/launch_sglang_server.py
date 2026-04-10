# File: benchmark/lora/migration/launch_sglang_server.py
"""
Launch SGLang Server Instances with Resource Monitoring
"""

import argparse
import subprocess
import time
import logging
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sglang.srt.instances.lora_config_paths import LORA_PATH, NUM_LORAS

import psutil
import GPUtil


logging.basicConfig(
    level=logging. INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class ResourceMonitor: 
    """
    Resource Monitor with Manual Phase Control
    """
    
    def __init__(self, output_file: Optional[str] = None):
        self.monitoring_data: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        
        self.manual_phase = "warmup"
        self.benchmark_count = 0
        
        if output_file:
            self.output_file = output_file
        else:  
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"monitoring_sglang_{timestamp}.jsonl"
    
    def _collect_data(self) -> Dict[str, Any]:  
        """
        collect a single data point
        """
        timestamp = datetime.now().isoformat()
        
        # CPU's current utilization (percentage), non-blocking
        cpu_util = psutil.cpu_percent(interval=None)
        
        # CPU's memory usage
        vmem = psutil.virtual_memory()
        cpu_memory_used_mb = vmem.used / 1024 / 1024
        cpu_memory_total_mb = vmem.total / 1024 / 1024
        cpu_memory_util = vmem.percent
        
        # GPU's current utilization and memory usage
        gpu_utils = {}
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_data = {
                    "utilization": gpu.load * 100,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_util":  gpu.memoryUtil * 100,
                    "temperature": gpu.temperature,
                }
                gpu_utils[f"gpu_{gpu. id}"] = gpu_data
        except Exception as e:  
            logger. warning(f"[Monitor] 获取GPU信息失败: {e}")
        
        data_point = {
            "timestamp": timestamp,
            "phase": self.manual_phase,
            "benchmark_count": self.benchmark_count,
            "cpu_utilization": cpu_util,
            "cpu_memory_used_mb": cpu_memory_used_mb,
            "cpu_memory_total_mb": cpu_memory_total_mb,
            "cpu_memory_util": cpu_memory_util,
            "gpus": gpu_utils,
        }
        
        return data_point
    
    def _monitoring_loop(self):
        """
        Monitoring loop (thread version)
        """
        logger.info("[Monitor] 开始资源监控循环...")
        
        while not self.stop_event.is_set():
            try:
                data_point = self._collect_data()
                
                with self. lock:
                    self.monitoring_data.append(data_point)
                
                # each second collect
                self.stop_event.wait(1.0)
                
            except Exception as e:
                logger. error(f"[Monitor] 监控循环错误:  {e}")
                time.sleep(1.0)
        
        logger.info("[Monitor] 监控循环结束")
    
    def start_monitoring(self):
        """
        Start resource monitoring (threaded mode)
        """
        if self.thread and self.thread.is_alive():
            logger.warning("[Monitor] 监控已在运行")
            return
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        logger.info(f"[Monitor] 监控已启动，输出文件: {self.output_file}")
    
    def stop_monitoring(self):
        """
        Stop resource monitoring (threaded mode)
        """
        if self.thread and self.thread.is_alive():
            logger.info("[Monitor] 停止监控...")
            self.stop_event.set()
            self.thread. join(timeout=5.0)
            if self.thread.is_alive():
                logger.warning("[Monitor] 监控线程未能正常结束")
        
        # save data on stop
        self. save_data()
    
    def set_manual_phase(self, phase: str):
        """
        Set the current manual phase
        """
        with self.lock:
            old_phase = self. manual_phase
            self.manual_phase = phase
            if old_phase != phase:
                logger.info(f"[Monitor] 阶段变更:  {old_phase} -> {phase}")
    
    def increment_benchmark(self):
        """
        Increment benchmark count
        """
        with self.lock:
            self.benchmark_count += 1
            logger.info(f"[Monitor] 📊 Benchmark #{self.benchmark_count} 开始")
    
    def save_data(self):
        """
        Save monitoring data to output file
        """
        try:
            with self.lock:
                data_to_save = self.monitoring_data. copy()
            
            if not data_to_save:  
                logger.info("[Monitor] 无数据需要保存")
                return
            
            os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
            
            with open(self.output_file, "w") as f:
                for data_point in data_to_save:
                    f.write(json.dumps(data_point) + "\n")
            
            logger.info(f"[Monitor] 已保存 {len(data_to_save)} 个数据点到 {self.output_file}")
            
            phase_counts = {}
            for dp in data_to_save:
                phase = dp.get("phase", "unknown")
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            
            logger.info(f"[Monitor] 各阶段数据点:  {phase_counts}")
            
        except Exception as e:  
            logger. error(f"[Monitor] 保存数据失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:  
        """
        Get current monitoring status
        """
        with self.lock:
            data_count = len(self. monitoring_data)
            current_phase = self.manual_phase
            
        is_running = self.thread and self.thread.is_alive()
        
        status = {
            "monitoring_active": is_running,
            "current_phase": current_phase,
            "benchmark_count": self.benchmark_count,
            "data_points_collected": data_count,
            "output_file": self. output_file,
        }
            
        return status


# Global monitor instance
monitor:  Optional[ResourceMonitor] = None
# FastAPI app
app = FastAPI()


def build_sglang_cmd(args, port:  int, gpu_id: int) -> str:
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
    logger.info(f"Number of LoRAs:  {NUM_LORAS}")
    logger.info(f"Starting port: {args.sglang_port}")
    logger.info("=" * 86)
    
    global monitor
    if monitor:
        monitor.set_manual_phase("launching_instances")
    
    for i in range(args.num_instances):
        port = args.sglang_port + i
        gpu_id = i
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} " + build_sglang_cmd(args, port, gpu_id)
        
        logger.info(f"\n[Instance {i}] 在 GPU {gpu_id} 上启动，端口 {port}")
        logger.info(f"  命令: {cmd}")
        
        proc = subprocess.Popen(cmd, shell=True)
        procs. append(proc)
        
        instance_url = f"http://{args.host}:{port}"
        instance_urls.append(instance_url)
        
        logger.info(f"  等待 {args. instance_delay} 秒...")
        time.sleep(args.instance_delay)
    
    logger.info("\n" + "=" * 86)
    logger.info("✓ 所有 SGLang 实例已启动!")
    logger.info("=" * 86)
    logger.info("\n实例 URLs:")
    for i, url in enumerate(instance_urls):
        logger.info(f"  实例 {i}:  {url}")
    logger.info("=" * 86)
    
    if monitor:  
        monitor. set_manual_phase("idle")
    
    return procs, instance_urls


@app.post("/start_benchmark")
async def start_benchmark(request: Request):
    """
    Notify the server that a benchmark is starting.
    """
    global monitor
    
    try:
        data = await request.json()
        bench_info = data.get("info", "")
        
        if monitor:
            monitor. increment_benchmark()
            monitor.set_manual_phase("benchmarking")
        
        logger.info("=" * 80)
        logger.info(f"[Monitor] 📊 BENCHMARK #{monitor.benchmark_count} STARTED")
        logger.info(f"[Monitor] Info: {bench_info}")
        logger.info("=" * 80)
        
        return JSONResponse({
            "status":  "ok", 
            "message": f"Benchmark #{monitor.benchmark_count} 已开始",
            "info": bench_info
        })
        
    except Exception as e:
        logger.error(f"[API] start_benchmark 错误: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/end_benchmark") 
async def end_benchmark(request:  Request):
    """
    Notify the server that a benchmark is ending.
    """
    global monitor
    
    try:
        data = await request.json()
        bench_info = data.get("info", "")
        
        if monitor:
            monitor. set_manual_phase("idle")
            monitor. save_data()
        
        logger. info("=" * 80)
        logger.info(f"[Monitor] ✅ BENCHMARK #{monitor.benchmark_count} COMPLETED")
        logger.info(f"[Monitor] Info: {bench_info}")
        logger.info(f"[Monitor] 切换到 IDLE 阶段")
        logger.info("=" * 80)
        
        return JSONResponse({
            "status":  "ok",
            "message":  f"Benchmark #{monitor.benchmark_count} 已完成"
        })
        
    except Exception as e:
        logger.error(f"[API] end_benchmark 错误:  {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/monitoring_status")
async def get_monitoring_status():
    """
    Get current monitoring status
    """
    global monitor
    
    if not monitor:  
        return JSONResponse({"error": "监控器未初始化"}, status_code=500)
    
    return JSONResponse(monitor. get_status())


@app.get("/health")
async def health():
    """
    Health check endpoint
    """
    return JSONResponse({"status":  "healthy"})


def run_unified_server(args):
    """
    Run the unified SGLang server with resource monitoring.
    """
    global monitor
    
    # 1. initialize resource monitor
    logger.info("[Monitor] 初始化资源监控器...")
    monitor = ResourceMonitor(output_file=args. monitoring_file)
    monitor.start_monitoring()
    logger.info("[Monitor] 进入 WARMUP 阶段")
    
    # 2. run SGLang instances
    procs, instance_urls = launch_sglang_instances(args)
    
    # 3. wait for warmup time
    logger.info(f"等待实例就绪 ({args.warmup_time} 秒)...")
    time.sleep(args.warmup_time)
    
    # keep monitor in IDLE phase
    logger. info("[Monitor] 实例就绪，保持 IDLE 阶段")
    
    # 4. setup FastAPI events
    @app.on_event("startup")
    async def startup():
        logger.info("=" * 80)
        logger.info("🚀 SGLang 服务器已就绪!")
        logger.info("💡 发送 POST 请求到 /start_benchmark 开始跟踪")
        if monitor:
            status = monitor.get_status()
            logger.info(f"📊 监控状态: {status['data_points_collected']} 个数据点已收集")
            logger.info(f"📁 输出文件: {status['output_file']}")
        logger.info("=" * 80)
    
    @app.on_event("shutdown")
    async def shutdown():
        global monitor
        
        logger.info("正在关闭服务器...")
        
        if monitor: 
            monitor.stop_monitoring()
        
        for i, proc in enumerate(procs):
            try:
                logger.info(f"终止实例 {i}...")
                proc. terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger. warning(f"强制终止实例 {i}")
                proc.kill()
            except Exception as e:  
                logger. error(f"终止实例 {i} 失败: {e}")
        
        logger.info("服务器已关闭")
    
    # 5. run Uvicorn server
    try:
        logger. info(f"\n启动 FastAPI 服务器在 {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:  
        logger. info("收到中断信号，正在退出...")


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(
        description="Launch SGLang Multi-Instance Server with Resource Monitoring"
    )
    
    # Server
    parser. add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-instances", type=int, default=2)
    parser.add_argument("--sglang-port", type=int, default=30001)
    
    # Instance
    parser. add_argument("--instance-delay", type=float, default=16.0,
                       help="每个实例启动后的等待时间 (秒)")
    parser.add_argument("--warmup-time", type=float, default=10.0,
                       help="所有实例启动后的额外等待时间 (秒)")
    
    # Monitoring
    parser.add_argument("--monitoring-file", type=str, default=None,
                       help="监控数据输出文件 (默认自动生成)")
    
    # SGLang
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--max-loras-per-batch", type=int, default=8)
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument("--lora-backend", type=str, default="csgmv")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--enable-mscclpp", action="store_true")
    
    args = parser.parse_args()
    run_unified_server(args)
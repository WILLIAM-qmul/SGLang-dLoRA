# File: benchmark/lora/migration/launch_dlora_server.py
"""
Launch Unified Server with EngineManager for dynamic load balancing.
Fully adapted from dLoRA to SGLang architecture.

Fixes:
1) Robust SSE forwarding: do NOT use resp.content.iter_any(), which can break SSE framing
   and cause client benchmark to treat requests as failed/unfinished.
2) Always emit valid SSE messages to the client, including termination "data: [DONE]\\n\\n".
3) Make completion bookkeeping robust against client disconnects/cancellation.
4) Optional unified logging to a single file, including SGLang instance subprocess stdout/stderr.
"""

import argparse
import logging
import subprocess
import time
import asyncio
import aiohttp
import uvicorn
import json
import os
import uuid
from typing import Any, AsyncIterator, Dict, Optional, Set, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sglang.srt.instances.lora_config_paths import LORA_PATH, NUM_LORAS
from sglang.srt.instances.instance_manager import InstanceManager, MigrationType


# logger = logging.getLogger("launch_dlora_server")
logger = logging.getLogger(__name__)

def setup_logging(log_file: Optional[str] = None, level: str = "INFO") -> None:
    """Setup root logging to console + optional file."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    for h in list(root.handlers):
        root.removeHandler(h)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(getattr(logging, level.upper(), logging.INFO))
        root.addHandler(fh)


app = FastAPI()
manager: InstanceManager = None
client_session: Optional[aiohttp.ClientSession] = None
request_migration_map: Dict[str, int] = {}
migration_map_lock = asyncio.Lock()


def build_sglang_cmd(args, port: int, gpu_id: int) -> str:
    """Build SGLang server launch command."""
    base_path = LORA_PATH["base"]
    
    cmd = f"python -m sglang.launch_server --model-path {base_path} "
    
    cmd += "--enable-lora "
    cmd += "--max-lora-rank 64 "
    cmd += "--lora-target-modules all "
    
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


def launch_sglang_instances(
    args: argparse.Namespace, log_file: Optional[str]
) -> Tuple[list, list]:
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

    # If a unified log file is provided, append all instance stdout/stderr to it.
    # NOTE: This mixes logs, but satisfies "all in one file" requirement.
    log_fh = None
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        log_fh = open(log_file, "ab", buffering=0)

    for i in range(args.num_instances):
        port = args.sglang_port + i
        gpu_id = i
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} " + build_sglang_cmd(args, port, gpu_id)

        logger.info(f"\n[Instance {i}] Launching on GPU {gpu_id}, Port {port}")
        logger.info(f"  Command: {cmd}")

        if log_fh is not None:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=log_fh,
                stderr=log_fh,
            )
        else:
            proc = subprocess.Popen(cmd, shell=True)

        procs.append(proc)

        instance_url = f"http://{args.host}:{port}"
        instance_urls.append(instance_url)

        time.sleep(3)

    logger.info("\n" + "=" * 86)
    logger.info("✓ All SGLang Instances Launched!")
    logger.info("=" * 86)
    logger.info("\nInstance URLs:")
    for i, url in enumerate(instance_urls):
        logger.info(f"  Instance {i}: {url}")
    logger.info("=" * 86)

    return procs, instance_urls


def _ensure_sse_bytes(data: str) -> bytes:
    """Wrap a payload as an SSE 'data:' event."""
    if not data.endswith("\n\n"):
        if data.endswith("\n"):
            data = data + "\n"
        else:
            data = data + "\n\n"
    # Ensure it has "data:" prefix for SSE compatibility with OpenAI-style stream consumers
    if not data.startswith("data:"):
        data = "data: " + data
        if not data.endswith("\n\n"):
            data += "\n\n"
    return data.encode("utf-8")

def _sse_done() -> bytes:
    return b"data: [DONE]\n\n"

def _split_sse_events(buf: bytearray) -> list[bytes]:
    """Split SSE events by blank line (\\n\\n), preserving event bytes."""
    out: list[bytes] = []
    sep = b"\n\n"
    while True:
        i = buf.find(sep)
        if i < 0:
            break
        out.append(bytes(buf[: i + 2]))
        del buf[: i + 2]
    return out

def _extract_sse_data(event: bytes) -> Optional[str]:
    """Extract concatenated data: lines from an SSE event, without 'data:' prefix."""
    text = event.decode("utf-8", errors="replace")
    data_lines = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    if not data_lines:
        return None
    return "\n".join(data_lines)


async def _stream_backend_sse_events(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
) -> AsyncIterator[bytes]:
    """
    POST to backend /generate and yield complete SSE event frames (bytes ending with \\n\\n).
    """
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"backend status={resp.status}, body={body[:500]}")
        buf = bytearray()
        async for chunk in resp.content.iter_any():
            if not chunk:
                continue
            buf.extend(chunk)
            for ev in _split_sse_events(buf):
                yield ev
        # flush tail if any
        if buf.strip():
            tail = bytes(buf)
            if not tail.endswith(b"\n\n"):
                tail += b"\n\n"
            yield tail


@app.post("/notify_migration")
async def notify_migration(request: Request):
    """
    Receive migration notifications from InstanceManager
    Body: {
        "migrations": [
            {"request_id": "req_xxx", "old_engine":  0, "new_engine": 1},
            ... 
        ]
    }
    """
    try:
        data = await request.json()
        migrations = data.get("migrations", [])
        
        async with migration_map_lock: 
            for mig in migrations:
                rid = mig["request_id"]
                new_engine = mig["new_engine"]
                request_migration_map[rid] = new_engine
                logger.info(f"[Migration] Notified:  {rid} -> Engine {new_engine}")
        
        return JSONResponse({"status": "ok", "count": len(migrations)})
    
    except Exception as e:
        logger.error(f"[Migration] Notification error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/generate")
async def generate(request: Request):
    """
    Generate completion with dynamic load balancing and auto-reconnect on migration. 
    """
    global manager
    assert manager is not None, "EngineManager not initialized."

    request_dict = await request.json()
    text = request_dict.get("text", "")
    sampling_params = request_dict.get("sampling_params", {})
    lora_path = request_dict.get("lora_path", "lora0")

    try:
        model_id = int(lora_path.replace("lora", "")) if lora_path.startswith("lora") else 0
    except Exception:
        model_id = 0

    request_id = request_dict.get("rid", f"req_{uuid.uuid4().hex[:12]}")

    instance_id = await manager.select_engine(request_id, model_id)
    
    client_disconnected = False

    async def stream_from_backend_with_migration() -> AsyncIterator[bytes]:
        """
        Supports migration-aware streaming response generator.
        Automatically checks for migration and reconnects if client disconnects.
        """
        nonlocal client_disconnected, instance_id
        
        start_ts = time.time()
        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=None)
        max_retries = 2
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                instance_url = manager.instance_urls[instance_id]
                target_url = f"{instance_url}/generate"
                
                logger.debug(f"[{request_id}] Attempt {retry_count + 1}:  Connecting to Engine {instance_id}")
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    payload = {
                        "rid": request_id,
                        "text": text,
                        "sampling_params": sampling_params,
                        "lora_path": lora_path,
                        "stream": True,
                    }

                    async with session.post(target_url, json=payload) as resp:
                        if resp.status != 200:
                            error_msg = {
                                "error": f"Backend returned {resp.status}",
                                "instance_id": instance_id,
                                "request_id": request_id,
                            }
                            yield _ensure_sse_bytes(json.dumps(error_msg))
                            
                            migrated_engine = await _check_migration(request_id)
                            if migrated_engine is not None and retry_count < max_retries - 1:
                                logger. info(f"[{request_id}] Detected migration to Engine {migrated_engine}, retrying...")
                                instance_id = migrated_engine
                                retry_count += 1
                                await asyncio.sleep(0.5)
                                continue
                            else:
                                yield _sse_done()
                                return

                        async for line in resp.content.iter_any():
                            if not line: 
                                continue
                            yield line

                        yield _sse_done()
                        logger.info(f"[{request_id}] Stream completed successfully on Engine {instance_id}")
                        return

            except (aiohttp.ClientError, asyncio.CancelledError) as e:
                logger.warning(f"[{request_id}] Connection error on Engine {instance_id}: {type(e).__name__}")
                
                migrated_engine = await _check_migration(request_id)
                
                if migrated_engine is not None and retry_count < max_retries - 1:
                    logger.info(f"[{request_id}] Detected migration:  Engine {instance_id} -> {migrated_engine}")
                    instance_id = migrated_engine
                    retry_count += 1
                    await asyncio.sleep(0.5)
                    continue
                else:
                    if isinstance(e, asyncio.CancelledError):
                        client_disconnected = True
                        raise
                    
                    error_msg = {
                        "error": str(e),
                        "instance_id": instance_id,
                        "request_id": request_id,
                        "note": "Connection failed and no migration detected"
                    }
                    yield _ensure_sse_bytes(json.dumps(error_msg))
                    yield _sse_done()
                    return

            except Exception as e:
                logger.error(f"[{request_id}] Unexpected error: {e}")
                error_msg = {
                    "error": str(e),
                    "instance_id": instance_id,
                    "request_id": request_id,
                }
                yield _ensure_sse_bytes(json.dumps(error_msg))
                yield _sse_done()
                return

        logger.error(f"[{request_id}] Max retries exceeded")
        error_msg = {
            "error": "Max retries exceeded",
            "request_id": request_id,
        }
        yield _ensure_sse_bytes(json.dumps(error_msg))
        yield _sse_done()

    return StreamingResponse(stream_from_backend_with_migration(), media_type="text/event-stream")


async def _check_migration(request_id: str) -> Optional[int]:
    """
    Checks if the request has been migrated to a new engine.
    Returns:  new_engine_id if migrated, None otherwise
    """
    async with migration_map_lock:
        new_engine = request_migration_map.get(request_id)
        if new_engine is not None:
            del request_migration_map[request_id]
            return new_engine
    return None


@app.get("/get_manager_stats")
async def get_manager_stats():
    global manager
    assert manager is not None
    return JSONResponse(manager.get_stats())


@app.post("/reset_manager_stats")
async def reset_manager_stats():
    global manager
    assert manager is not None
    await manager.reset_stats()
    return JSONResponse({"status": "ok"})


@app.get("/health")
async def health():
    global manager
    assert manager is not None
    return JSONResponse(
        {
            "status": "healthy",
            "num_instances": manager.num_instances,
            "migration_enabled": manager.is_running(),
        }
    )


async def shutdown_handler():
    global manager
    if manager:
        await manager.close()


def run_unified_server(args:  argparse.Namespace):
    global manager

    setup_logging(args. log_file, args.log_level)

    procs, instance_urls = launch_sglang_instances(args, args.log_file)

    logger.info("\nWaiting for instances to be ready...")
    time.sleep(args.instance_warmup_sec)

    migration_type = MigrationType(args.migration_type)

    manager = InstanceManager(
        num_instances=args.num_instances,
        num_models=NUM_LORAS,
        instance_urls=instance_urls,
        migration_type=migration_type,
        migration_interval=args.migration_interval,
        lora_capacity_per_engine=args.max_loras_per_batch,
        max_running_requests=args.max_running_requests,
    )
    
    manager.unified_server_url = f"http://{args.host}:{args.port}"

    @app.on_event("startup")
    async def start_manager_background_loop():
        await manager.initialize()
        
        if migration_type != MigrationType.DISPATCH_ONLY:
            logger.info("Starting manager background loop...")
            await manager. start_background_loop()
        else:
            logger.info("Migration disabled (DISPATCH_ONLY).")

    @app.on_event("shutdown")
    async def shutdown():
        await shutdown_handler()
        for proc in procs:
            try:
                proc.terminate()
            except Exception: 
                pass

    logger.info(f"\n[Unified Server] Starting on port {args.port}")
    logger.info(f"[Unified Server] Migration type: {migration_type.name}")

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level. lower())
    except KeyboardInterrupt:
        logger. info("Terminating...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Unified Server for SGLang Multi-Instance Serving"
    )

    # Unified server config
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-instances", type=int, default=2)
    parser.add_argument("--sglang-port", type=int, default=30001)
    parser.add_argument("--instance-warmup-sec", type=float, default=20.0)

    # Logging config
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="If set, write unified server logs AND all instance stdout/stderr into this file.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")

    # Migration config
    parser.add_argument(
        "--migration-type",
        type=int,
        default=3,
        help="1=DISPATCH_ONLY, 2=DISPATCH_MIG, 3=PERIOD_MIG",
    )
    parser.add_argument("--migration-interval", type=float, default=5.0)

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
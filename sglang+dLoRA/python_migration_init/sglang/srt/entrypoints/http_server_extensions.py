# File: sglang+dLoRA/python/sglang/srt/entrypoints/http_server_extensions.py
"""
HTTP server extensions for engine manager integration.
Add these endpoints to the existing http_server.py
"""

from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


async def get_engine_stats(request: Request):
    """
    Get current engine statistics for migration decision.
    """
    from sglang.srt.entrypoints.http_server import _global_state
    
    tokenizer_manager = _global_state.tokenizer_manager
    
    # 1. Get stats from Scheduler (Waiting queue info, free blocks)
    # This call is now async and goes via IPC
    scheduler_stats = await tokenizer_manager.get_engine_stats()
    
    # 2. Get active LoRA models from TokenizerManager (local info)
    active_models = []
    # Check lora_registry or lora_manager depending on implementation
    if hasattr(tokenizer_manager, 'lora_registry') and tokenizer_manager.lora_registry:
        # lora_registry maps name -> id/path. We want active loaded ones.
        # Usually loaded state is in ModelWorker, but TokenizerManager might track it via LoRAManager
        # For now, we can return empty or try to fetch if available.
        # If using dLoRA, the EngineManager tracks active models via 'adjust_lora_adapter'.
        pass
    
    # 3. Construct response
    stats = {
        "num_requests": scheduler_stats.num_requests,
        "req_model_cnt": scheduler_stats.req_model_cnt,
        "num_free_gpu_blocks": scheduler_stats.num_free_gpu_blocks,
        "num_free_cpu_blocks": scheduler_stats.num_free_cpu_blocks,
        "lora_capacity": 8,  # Configure this
        "active_models": active_models, # EngineManager tracks this mostly
        "req_metadata": scheduler_stats.req_metadata, # Only waiting requests
        "model_exec_time": {},
    }
    
    return JSONResponse(stats)


async def fetch_seq_groups(request: Request):
    """
    Fetch sequence groups for migration. 
    Used to extract request state before migration.
    """
    from sglang.srt.entrypoints.http_server import _global_state
    
    data = await request.json()
    request_ids = data.get("request_ids", [])
    
    tokenizer_manager = _global_state.tokenizer_manager
    
    # Fetch from Scheduler via TokenizerManager
    result = await tokenizer_manager.fetch_seq_groups(request_ids)
    
    return JSONResponse({"seq_groups": result.seq_groups})


async def insert_seq_groups(request: Request):
    """
    Insert migrated sequence groups into scheduler.
    Used to receive migrated requests. 
    """
    from sglang.srt.entrypoints.http_server import _global_state
    from sglang.srt.managers.io_struct import GenerateReqInput
    
    data = await request.json()
    seq_groups = data.get("seq_groups", [])
    
    tokenizer_manager = _global_state.tokenizer_manager
    
    inserted = []
    
    for seq_data in seq_groups:
        try:
            # Recreate request using GenerateReqInput
            # This ensures it goes through the standard validation and tokenization pipeline
            req_input = GenerateReqInput(
                rid=seq_data["request_id"],
                text=seq_data["prompt"],
                input_ids=seq_data.get("prompt_token_ids"),
                sampling_params=seq_data["sampling_params"],
                lora_path=seq_data.get("lora_path"),
                image_data=seq_data.get("image_data"),
                modalities=seq_data.get("modalities"),
            )
            
            # Submit to TokenizerManager
            # We use generate_request which handles tokenization and sending to scheduler
            await tokenizer_manager.generate_request(req_input, request)
            
            inserted.append(seq_data["request_id"])
            
        except Exception as e:
            logger.error(f"Failed to insert seq_group {seq_data.get('request_id')}: {e}", exc_info=True)
    
    return JSONResponse({"inserted": inserted, "count": len(inserted)})


async def abort_requests(request: Request):
    """
    Abort multiple requests by IDs.
    Used after successful migration to clean up source engine.
    """
    from sglang.srt.entrypoints.http_server import _global_state
    
    data = await request.json()
    request_ids = data.get("request_ids", [])
    
    tokenizer_manager = _global_state.tokenizer_manager
    
    aborted = []
    for req_id in request_ids:
        try:
            tokenizer_manager.abort_request(req_id)
            aborted.append(req_id)
        except Exception as e:
            logger.warning(f"Failed to abort {req_id}: {e}")
    
    return JSONResponse({"aborted": aborted, "count": len(aborted)})


async def adjust_lora_adapter(request: Request):
    """
    Adjust active LoRA adapters based on migration plan.
    """
    from sglang.srt.entrypoints.http_server import _global_state
    from sglang.srt.managers.io_struct import LoadLoRAAdapterReqInput, UnloadLoRAAdapterReqInput
    
    data = await request.json()
    active_model_ids = set(data.get("active", []))
    
    tokenizer_manager = _global_state.tokenizer_manager
    
    # 1. Parse available LoRAs from server_args to get name->path mapping
    # server_args.lora_paths format is expected to be ["name=path", ...]
    available_loras = {}
    if tokenizer_manager.server_args.lora_paths:
        for item in tokenizer_manager.server_args.lora_paths:
            if "=" in item:
                name, path = item.split("=", 1)
                available_loras[name] = path
            else:
                # Handle case where only path is provided, assuming directory name is lora name or similar
                # For safety in this demo, we skip or log warning
                logger.warning(f"Skipping malformed lora_path entry: {item}")

    # 2. Determine actions
    # We map model_id X to "loraX"
    target_lora_names = {f"lora{mid}" for mid in active_model_ids}
    
    results = []
    
    # Iterate over all known LoRAs to sync state
    for name, path in available_loras.items():
        if name in target_lora_names:
            # Target: Ensure loaded
            # Note: load_lora_adapter in SGLang is typically idempotent or handles re-loading gracefully
            logger.info(f"Requesting load for LoRA: {name}")
            req = LoadLoRAAdapterReqInput(lora_name=name, lora_path=path)
            try:
                res = await tokenizer_manager.load_lora_adapter(req, request)
                results.append({"name": name, "action": "load", "success": res.success, "msg": getattr(res, 'message', '')})
            except Exception as e:
                logger.error(f"Error loading LoRA {name}: {e}")
                results.append({"name": name, "action": "load", "success": False, "error": str(e)})
        else:
            # Not in target: Ensure unloaded
            # We aggressively unload to free up space for migration
            logger.info(f"Requesting unload for LoRA: {name}")
            req = UnloadLoRAAdapterReqInput(lora_name=name)
            try:
                res = await tokenizer_manager.unload_lora_adapter(req, request)
                results.append({"name": name, "action": "unload", "success": res.success, "msg": getattr(res, 'message', '')})
            except Exception as e:
                logger.error(f"Error unloading LoRA {name}: {e}")
                results.append({"name": name, "action": "unload", "success": False, "error": str(e)})
                
    return JSONResponse({"results": results})
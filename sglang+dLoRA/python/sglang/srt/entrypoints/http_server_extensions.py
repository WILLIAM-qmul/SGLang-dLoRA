# File: sglang+dLoRA/python/sglang/srt/entrypoints/http_server_extensions.py
"""
HTTP server extensions for instance manager integration. 
Add these endpoints to the existing http_server. py
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
    scheduler_stats = await tokenizer_manager.get_engine_stats()
    
    # 2. Get active LoRA models from TokenizerManager
    active_models = []
    if hasattr(tokenizer_manager, 'lora_registry') and tokenizer_manager.lora_registry:
        active_models = list(tokenizer_manager.lora_registry.get_all_adapters(). keys())
    
    # 3.  Construct response
    stats = {
        "num_requests": scheduler_stats.num_requests,
        "req_model_cnt": scheduler_stats.req_model_cnt,
        "num_free_gpu_pages": scheduler_stats.num_free_gpu_pages,
        "lora_capacity": scheduler_stats.lora_capacity,
        "active_models": active_models,
        "req_metadata": scheduler_stats.req_metadata,
        "model_exec_time": scheduler_stats.model_exec_time,
        "available_gpu_memory": scheduler_stats.available_gpu_memory,  # In bytes
        "cache_page_size": scheduler_stats.cache_page_size,
        "lora_weight_size": scheduler_stats.lora_weight_size,
    }
    
    return JSONResponse(stats)


async def fetch_seq_groups(request: Request):
    """
    Fetch sequence groups for migration.
    """
    from sglang.srt.entrypoints.http_server import _global_state
    
    data = await request.json()
    request_ids = data.get("request_ids", [])
    
    tokenizer_manager = _global_state.tokenizer_manager
    result = await tokenizer_manager.fetch_seq_groups(request_ids)
    
    return JSONResponse({"seq_groups": result. seq_groups})


async def insert_seq_groups(request: Request):
    """
    Insert migrated sequence groups into scheduler.
    """
    from sglang.srt.entrypoints.http_server import _global_state
    from sglang.srt.managers.io_struct import GenerateReqInput
    
    data = await request. json()
    seq_groups = data.get("seq_groups", [])
    
    tokenizer_manager = _global_state.tokenizer_manager
    
    inserted = []
    for seq_data in seq_groups:
        try:
            req_input = GenerateReqInput(
                rid=seq_data["request_id"],
                text=seq_data["prompt"],
                input_ids=seq_data. get("prompt_token_ids"),
                sampling_params=seq_data["sampling_params"],
                lora_path=seq_data.get("lora_path"),
            )
            
            await tokenizer_manager.generate_request(req_input, request)
            inserted.append(seq_data["request_id"])
            
        except Exception as e:
            logger.error(f"Failed to insert seq_group {seq_data. get('request_id')}: {e}")
    
    return JSONResponse({"inserted": inserted, "count": len(inserted)})


async def abort_requests(request: Request):
    """
    Abort multiple requests by IDs.
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

    # Get available LoRAs from the LoRA registry, which is the source of truth
    available_loras = {}
    if tokenizer_manager.lora_registry:
        for lora_ref in tokenizer_manager.lora_registry.get_all_adapters().values():
            available_loras[lora_ref.lora_name] = lora_ref.lora_path

    # Determine target LoRAs (model_id X -> "loraX")
    target_lora_names = {f"lora{mid}" for mid in active_model_ids}

    results = []
    for name, path in available_loras.items():
        if name in target_lora_names:
            # Load LoRA
            req = LoadLoRAAdapterReqInput(lora_name=name, lora_path=path)
            try:
                res = await tokenizer_manager.load_lora_adapter(req, request)
                results.append({"name": name, "action": "load", "success": res.success})
            except Exception as e:
                results.append({"name": name, "action": "load", "success": False, "error": str(e)})
        else:
            # Unload LoRA
            req = UnloadLoRAAdapterReqInput(lora_name=name)
            try:
                res = await tokenizer_manager.unload_lora_adapter(req, request)
                results.append({"name": name, "action": "unload", "success": res.success})
            except Exception as e:
                results.append({"name": name, "action": "unload", "success": False, "error": str(e)})

    return JSONResponse({"results": results})
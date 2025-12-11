# File: sglang+dLoRA/python/sglang/srt/entrypoints/http_server_extensions.py
"""
HTTP server extensions for instance manager integration. 
Add these endpoints to the existing http_server. py
"""

from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


async def get_instance_stats(request: Request):
    """
    Get current instance statistics for migration decision.
    """
    from sglang.srt.entrypoints.http_server import _global_state
    
    tokenizer_manager = _global_state.tokenizer_manager
    
    # 1. Get stats from Scheduler (Waiting queue info, free blocks)
    scheduler_stats = await tokenizer_manager.get_engine_stats()
    
    # 2.  Construct response
    stats = {
        "lora_capacity": scheduler_stats.lora_capacity,
        "available_gpu_memory": scheduler_stats.available_gpu_memory,
        "num_free_gpu_pages": scheduler_stats.num_free_gpu_pages,
        "cache_page_size": scheduler_stats.cache_page_size,
    }
    
    return JSONResponse(stats)


async def get_engine_stats(request: Request):
    """
    Get current engine statistics for migration decision.
    """
    from sglang.srt.entrypoints.http_server import _global_state
    
    tokenizer_manager = _global_state.tokenizer_manager
    
    # 1. Get stats from Scheduler (Waiting queue info, free blocks)
    scheduler_stats = await tokenizer_manager.get_engine_stats()
    
    # 2. Get active LoRA models from TokenizerManager
    # active_models = []
    # if hasattr(tokenizer_manager, 'lora_registry') and tokenizer_manager.lora_registry:
    #     active_models = list(tokenizer_manager.lora_registry.get_all_adapters(). keys())
    
    # 3.  Construct response
    stats = {
        "num_requests": scheduler_stats.num_requests,
        "req_model_cnt": scheduler_stats.req_model_cnt,
        "num_free_gpu_pages": scheduler_stats.num_free_gpu_pages,
        "lora_capacity": scheduler_stats.lora_capacity,
        "req_metadata": scheduler_stats.req_metadata,
        "model_exec_time": scheduler_stats.model_exec_time,
        "available_gpu_memory": scheduler_stats.available_gpu_memory,
        "cache_page_size": scheduler_stats.cache_page_size,
        # "active_models": active_models,
    }
    
    return JSONResponse(stats)


async def get_loaded_lora_adapters(request: Request):
    """
    Get currently loaded LoRA adapters from the LoRA registry.
    This is the source of truth for loaded adapters.
    
    Returns:
        Dict mapping lora_name -> lora_path for all loaded adapters
    """
    from sglang.srt.entrypoints.http_server import _global_state
    
    tokenizer_manager = _global_state.tokenizer_manager
    
    loaded_adapters = {}
    
    # Get from LoRA registry (source of truth)
    if hasattr(tokenizer_manager, 'lora_registry') and tokenizer_manager.lora_registry:
        all_adapters = tokenizer_manager.lora_registry.get_all_adapters()
        loaded_adapters = {
            lora_ref.lora_name: lora_ref.lora_path 
            for lora_ref in all_adapters. values()
        }
    
    return JSONResponse({
        "loaded_adapters":  loaded_adapters,
        "num_loaded": len(loaded_adapters)
    })


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
# File: sglang+dLoRA/python/sglang/srt/entrypoints/http_server_extensions.py
"""
HTTP server extensions for instance manager integration. 
Add these endpoints to the existing http_server. py
"""

from fastapi import Request
from fastapi.responses import JSONResponse
import logging
import time

logger = logging.getLogger(__name__)


async def get_instance_stats(request: Request):
    """
    Get current instance statistics for migration decision.
    """
    from sglang.srt.entrypoints.http_server import _global_state
    
    tokenizer_manager = _global_state.tokenizer_manager
    
    # 1. Get stats from Scheduler (Waiting queue info, free blocks)
    scheduler_stats = await tokenizer_manager.get_instance_stats()
    
    # 2.  Construct response
    stats = {
        "lora_capacity": scheduler_stats.lora_capacity,
        "available_gpu_memory": scheduler_stats.available_gpu_memory,
        "num_free_gpu_pages": scheduler_stats.num_free_gpu_pages,
        "cache_page_size": scheduler_stats.cache_page_size,
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
    
    
async def get_req_model_cnt(request: Request):
    """
    Get lightweight request model count statistics.
    Much faster than get_engine_stats as it only returns request counts.
    
    Returns:
        {
            "req_model_cnt": {"model1": 10, "model2":  5, ...},
            "total_requests": 15,
            "exec_cost":  2.5
        }
    """
    from sglang.srt.entrypoints.http_server import _global_state
    
    tokenizer_manager = _global_state.tokenizer_manager
    
    try:
        result = await tokenizer_manager.get_req_model_cnt()
        
        return JSONResponse({
            "req_model_cnt": result.req_model_cnt,
            "total_requests":  result.total_requests,
            "exec_cost": result.exec_cost,
            "timestamp": time.time()
        })
    
    except Exception as e:
        logger.error(f"Failed to get req_model_cnt: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )
        

async def get_migration_info(request:  Request):
    """
    Get migration information for instance manager. 
    Returns request metadata and model execution time statistics.
    """
    from sglang.srt.entrypoints.http_server import _global_state
    
    tokenizer_manager = _global_state.tokenizer_manager
    
    try:
        result = await tokenizer_manager.get_migration_info()
        
        return JSONResponse({
            "req_metadata": result.req_metadata,
            "model_exec_time": {
                model_id: {"count": stats[0], "total_time": stats[1]}
                for model_id, stats in result.model_exec_time.items()
            },
            "num_requests":  result.num_requests,
            "timestamp": time.time(),
        })
    
    except Exception as e:
        logger.error(f"Failed to get migration info: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )
# File: sglang+dLoRA/python/sglang/srt/entrypoints/http_server_extensions.py
"""
HTTP server extensions for instance manager integration. 
Add these endpoints to the existing http_server. py
"""

from fastapi import Request
from fastapi.responses import JSONResponse
import logging
import time
import asyncio
import inspect
from sglang.srt.managers.io_struct import GenerateReqInput

from sglang.srt.sampling.sampling_params import SamplingParams

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
        

async def fetch_seq_groups(request: Request):
    """Endpoint to fetch request states for migration."""
    from sglang.srt.entrypoints.http_server import _global_state
    data = await request.json()
    request_ids = data.get("request_ids", [])
    
    if not request_ids:
        return JSONResponse({"seq_groups": []})
        
    seq_groups = await _global_state.tokenizer_manager.fetch_requests(request_ids)
    return JSONResponse({"seq_groups": seq_groups})


async def insert_seq_groups(request: Request):
    """Endpoint to insert migrated requests (text-level resume via input_ids)."""
    from sglang.srt.entrypoints.http_server import _global_state

    data = await request.json()
    seq_groups = data.get("seq_groups", []) or []

    accepted = 0
    errors = []

    for seq_data in seq_groups:
        try:
            rid = seq_data.get("rid")
            input_ids = seq_data.get("input_ids")
            sampling_params = seq_data.get("sampling_params")
            lora_path = seq_data.get("lora_path")

            if not rid:
                raise ValueError("missing rid")
            if not input_ids:
                raise ValueError(f"missing input_ids for rid={rid}")
            if sampling_params is None:
                raise ValueError(f"missing sampling_params for rid={rid}")
            
            # Use inspect to get valid constructor arguments for SamplingParams
            init_params = inspect.signature(SamplingParams.__init__).parameters
            valid_keys = set(init_params) - {'self'}
            
            # The constructor uses 'stop' but the class stores 'stop_strs'.
            # The incoming dictionary might have either. Let's allow both.
            if 'stop_strs' in sampling_params:
                valid_keys.add('stop_strs')

            filtered_sampling_params = {k: v for k, v in sampling_params.items() if k in valid_keys}

            # Rename 'stop_strs' to 'stop' for the constructor if needed
            if 'stop_strs' in filtered_sampling_params and 'stop' not in filtered_sampling_params:
                filtered_sampling_params['stop'] = filtered_sampling_params.pop('stop_strs')

            obj = GenerateReqInput(
                rid=rid,
                input_ids=input_ids,
                sampling_params=filtered_sampling_params,
                lora_path=lora_path,
                stream=True,
            )
            
            async def _consume_generate_request(obj):
                async for _ in _global_state.tokenizer_manager.generate_request(obj, None):
                    break
            
            asyncio.create_task(_consume_generate_request(obj))
            
            # Fire-and-forget without coupling to this HTTP request context
            # asyncio.create_task(_global_state.tokenizer_manager.generate_request(obj, None))
            accepted += 1

        except Exception as e:
            logger.error(f"Error inserting seq_group. Data: {seq_data}, Error: {e}", exc_info=True)
            errors.append({"rid": seq_data.get("rid"), "error": str(e)})

    return JSONResponse({"count": accepted, "errors": errors})


async def abort_requests(request: Request):
    """
    Abort multiple requests. 
    This is used after migration to clean up source engine. 
    
    Request body: 
        {
            "request_ids": ["req1", "req2", ...]
        }
    
    Returns:
        {
            "aborted_count": number of aborted requests
        }
    """
    from sglang.srt.entrypoints.http_server import _global_state
    
    tokenizer_manager = _global_state.tokenizer_manager
    
    try:
        body = await request.json()
        request_ids = body.get("request_ids", [])
        
        if not request_ids:
            return JSONResponse({"aborted_count": 0})
        
        # Abort requests
        for request_id in request_ids: 
            tokenizer_manager.abort_request(rid=request_id)
        
        return JSONResponse({
            "aborted_count":  len(request_ids)
        })
    
    except Exception as e:
        logger.error(f"Failed to abort requests: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )
# File: sglang+dLoRA/python/sglang/srt/instances/instance_manager.py

import asyncio
import aiohttp
import time
import logging
import torch
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from sglang.srt.managers.io_struct import InstanceStats, ReqModelCntStats, RequestMetadata, EngineStats
from sglang.srt.instances.lora_config_paths import NUM_LORAS, LORA_PATH
from sglang.srt.instances.lora_init import get_lora_sizes_for_instance_manager
from sglang.srt.instances.migration_ilp import MigrationILP

logger = logging.getLogger(__name__)

_GB = 1 << 30
PCIE_BANDWIDTH = 32 * _GB


class MigrationType(Enum): # ✅
    DISPATCH_ONLY = 1
    DISPATCH_MIG = 2
    PERIOD_MIG = 3


class InstanceManager:
    def __init__( # ✅
        self,
        num_instances: int,
        num_models: int,
        instance_urls: List[str],
        migration_type: MigrationType = MigrationType.PERIOD_MIG,
        migration_interval: float = 10.0,
        lora_capacity_per_engine: int = 8,
        max_running_requests: int = 16,
        default_exec_time: float = 5.0,
        migration_req_thres: int = 16,
        merge_speed_ratio: float = 0.6,
        # lora_init_params
        max_lora_rank: int = 64,
        dtype: torch.dtype = torch.float16,
        tp_size:  int = 1,
        pcie_bandwidth: float = PCIE_BANDWIDTH,
    ):
        # Basic configuration
        self.num_instances = num_instances
        self.num_models = num_models
        self.instance_urls = instance_urls
        
        # Execution parameters (from dLoRA)
        self.migration_type = migration_type
        self.migration_interval = migration_interval
        self.lora_capacity_per_engine = lora_capacity_per_engine
        self.max_running_requests = max_running_requests
        self.default_exec_time = default_exec_time
        self.migration_req_thres = migration_req_thres
        self.merge_speed_ratio = merge_speed_ratio
        
        # Request tracking
        self.request_to_engine: Dict[str, int] = {}
        self.engine_request_count: Dict[int, int] = {i: 0 for i in range(num_instances)}
        
        # Model distribution
        self.model_engine_mapping: Dict[int, List[int]] = {i: [] for i in range(num_models)}
        self.engine_model_mapping: Dict[int, List[int]] = {i: [] for i in range(num_instances)}
        self.expected_lora_distribution: List[float] = [(1.0 + 1e-7) / num_models] * num_models
        
        # ===== NEW: Calculate LoRA sizes during initialization =====
        print("Calculating LoRA weight sizes...")
        self.lora_weight_info:  Dict[int, Tuple[int, float]] = (
            get_lora_sizes_for_instance_manager(
                max_lora_rank=max_lora_rank,
                dtype=dtype,
                tp_size=tp_size,
                pcie_bandwidth=pcie_bandwidth,
            )
        )

        # Extract sizes and load costs
        self.lora_weight_sizes:  Dict[int, int] = {
            model_id: size for model_id, (size, _) in self.lora_weight_info.items()
        }
        self.lora_load_costs: Dict[int, float] = {
            model_id: load_time for model_id, (_, load_time) in self.lora_weight_info.items()
        }

        print("LoRA weight sizes calculated:")
        for model_id in range(num_models):
            size_bytes = self.lora_weight_sizes.get(model_id, 0)
            load_s = self.lora_load_costs.get(model_id, 0)
            print(f"  Model {model_id}: {size_bytes:.2f} bytes, load time: {load_s:.2f} s", flush=True)
        # ============================================================
        
        # Statistics
        self.reqs_metadata: Dict[int, List[RequestMetadata]] = {}
        self.instance_stats: Dict[int, InstanceStats] = {}
        self.engine_stats: Dict[int, EngineStats] = {}
        self.global_model_request_count: Dict[int, int] = {i: 0 for i in range(num_models)}
        self.engine_exec_cost: Dict[int, float] = {}
        self.engine_req_model_cnt: Dict[int, Dict[int, int]] = {}
        self.model_exec_info: Dict[int, List[float]] = {i: [0, 0.0] for i in range(num_models)}
        self.model_avg_exec_time: List[float] = [default_exec_time] * num_models
        
        # Async locks
        self.select_lock = asyncio.Lock()
        self.migration_lock = asyncio.Lock()
        
        # Background tasks
        self.background_loop = None
        self._running = False
        
        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        
        # GPU memory and blocks
        self.available_gpu_memorys: List[float] = [0.0] * num_instances
        self.num_gpu_pages: List[int] = [0] * num_instances
        self.cache_page_size: int = 4096
        
        # Engine capacity
        self.engine_lora_capacity: List[int] = [lora_capacity_per_engine] * num_instances
        
        # New: Track number of requests per engine
        self.engine_num_requests: Dict[int, int] = {i: 0 for i in range(num_instances)}
        
        logger.info(f"InstanceManager initialized: {num_instances} instances, {num_models} models")

    
    async def initialize(self): # ✅
        """Async initialization:  fetch stats and place LoRAs"""
        logger.info("Fetching initial stats from all instances...")
        await self._fetch_instance_initial_stats()
        
        logger.info("Initializing LoRA placement...")
        self._initialize_lora_placement()
        
        logger.info("Applying initial LoRA placement to instances...")
        await self._apply_lora_placement()
        
        logger.info("InstanceManager initialization complete")

    
    async def _fetch_instance_stats( # ✅
        self, 
        session: aiohttp.ClientSession, 
        engine_id: int, 
        url: str
    ) -> Optional[InstanceStats]:
        """Fetch statistics from a single instance"""
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger. warning(f"Instance {engine_id} returned status {resp.status}")
                    return None
                
                data = await resp.json()
                return InstanceStats.from_dict(data, engine_id)
        except Exception as e: 
            logger.error(f"Error fetching stats from instance {engine_id}: {e}")
            return None


    async def _fetch_instance_initial_stats(self): # ✅
        """Fetch initial stats from all instances"""
        session = await self._get_session()

        tasks = []
        for engine_id in range(self.num_instances):
            url = f"{self.instance_urls[engine_id]}/get_instance_stats"
            tasks.append(self._fetch_instance_stats(session, engine_id, url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for engine_id, result in enumerate(results):
            self.instance_stats[engine_id] = result
            self.engine_lora_capacity[engine_id] = result.lora_capacity
            self.num_gpu_pages[engine_id] = result.num_free_gpu_pages
            self.available_gpu_memorys[engine_id] = result.available_gpu_memory
            self.cache_page_size = result.cache_page_size

        logger.info(f"Fetched stats:  GPU pages={self.num_gpu_pages}, cache_page_size={self.cache_page_size}")

    
    def _initialize_lora_placement(self): # ✅
        """Initialize LoRA placement using dLoRA's greedy algorithm"""
        logger.info("Running LoRA placement optimization...")
        
        current_lora_distribution = [0] * self.num_models
        num_lora_replicas = 0
        best_bt = 0
        update_flag = True
        
        engine_ids = list(range(self.num_instances))
        models_not_allocated = list(range(self.num_models))
        
        # Greedy placement loop
        while update_flag:
            update_flag = False
            
            # Select next LoRA to place
            next_lora_type = 0
            for i in range(self.num_models):
                ratio_i = (current_lora_distribution[i] / (num_lora_replicas + 1e-7)) - self.expected_lora_distribution[i]
                ratio_next = (current_lora_distribution[next_lora_type] / (num_lora_replicas + 1e-7)) - self.expected_lora_distribution[next_lora_type]
                if ratio_i < ratio_next:
                    next_lora_type = i
            
            if next_lora_type not in models_not_allocated and models_not_allocated:
                next_lora_type = models_not_allocated[0]
            
            # Sort engines by available capacity
            engine_ids = sorted(
                engine_ids,
                key=lambda eid: self.engine_lora_capacity[eid] - len(self.engine_model_mapping[eid]),
                reverse=True
            )
            engine_ids = [
                eid for eid in engine_ids
                if self.engine_lora_capacity[eid] > len(self.engine_model_mapping[eid])
            ]
            
            # Try placement
            for engine_id in engine_ids:
                if next_lora_type in self.engine_model_mapping[engine_id]:
                    continue
                
                self.engine_model_mapping[engine_id].append(next_lora_type)
                self. available_gpu_memorys[engine_id] -= self.lora_weight_sizes[next_lora_type]
                
                new_bt = self._calc_min_bottleneck_throughput(self.expected_lora_distribution)
                
                if new_bt >= best_bt:
                    current_lora_distribution[next_lora_type] += 1
                    num_lora_replicas += 1
                    best_bt = new_bt
                    update_flag = True
                    
                    if next_lora_type in models_not_allocated: 
                        models_not_allocated.remove(next_lora_type)
                    break
                else:
                    self.engine_model_mapping[engine_id].remove(next_lora_type)
                    self.available_gpu_memorys[engine_id] += self.lora_weight_sizes[next_lora_type]
                    
                    if models_not_allocated: 
                        update_flag = True
        
        # Ensure minimum replicas for PERIOD_MIG
        if self.migration_type == MigrationType.PERIOD_MIG: 
            min_replicas = self.num_instances + self.num_models - 1
            next_engine_id = 0
            next_lora_type = random.randint(0, self. num_models - 1)
            
            while num_lora_replicas < min_replicas:
                while (len(self.engine_model_mapping[next_engine_id]) >= self.num_models or
                       next_lora_type in self.engine_model_mapping[next_engine_id]):
                    next_engine_id = (next_engine_id + 1) % self.num_instances
                
                self.engine_model_mapping[next_engine_id].append(next_lora_type)
                self.available_gpu_memorys[next_engine_id] -= self.lora_weight_sizes[next_lora_type]
                next_engine_id = (next_engine_id + 1) % self.num_instances
                current_lora_distribution[next_lora_type] += 1
                num_lora_replicas += 1
        
        # Build reverse mapping
        self.model_engine_mapping = {i: [] for i in range(self.num_models)}
        for engine_id, model_ids in self.engine_model_mapping.items():
            for model_id in model_ids: 
                self.model_engine_mapping[model_id].append(engine_id)
        
        logger.info(f"LoRA placement:  {num_lora_replicas} replicas, mapping={self.engine_model_mapping}")

    
    def find_best_lora_weight_schedule( # ✅
        self,
        expected_lora_distribution: List[float],
        current_lora_distribution: List[int] = None,
        engine_lora_capacity: List[int] = None
    ):
        """
        Greedy LoRA placement (from dLoRA's find_best_lora_weight_schedule).
        Main loop only - no cold start补齐.
        """
        if current_lora_distribution is None:
            current_lora_distribution = [0] * self. num_models
        
        num_lora_replicas = sum(current_lora_distribution)
        best_bt = 0
        update_flag = True
        
        engine_ids = list(range(self.num_instances))
        models_not_allocated = [i for i in range(self. num_models) if len(self.model_engine_mapping[i]) == 0]
        
        while update_flag:
            update_flag = False
            next_lora_type = 0
            for i in range(self.num_models):
                ratio_i = (current_lora_distribution[i] / (num_lora_replicas + 1e-7)) - expected_lora_distribution[i]
                ratio_next = (current_lora_distribution[next_lora_type] / (num_lora_replicas + 1e-7)) - expected_lora_distribution[next_lora_type]
                if ratio_i < ratio_next:
                    next_lora_type = i
            
            if next_lora_type not in models_not_allocated and models_not_allocated: 
                next_lora_type = models_not_allocated[0]
            
            # Sort engines
            if engine_lora_capacity is None:
                engine_ids = sorted(engine_ids, key=lambda eid: len(self.engine_model_mapping[eid]))
            else:
                engine_ids = sorted(
                    engine_ids,
                    key=lambda eid:  engine_lora_capacity[eid] - len(self.engine_model_mapping[eid]),
                    reverse=True
                )
                engine_ids = [eid for eid in engine_ids if engine_lora_capacity[eid] > len(self.engine_model_mapping[eid])]
            
            for engine_id in engine_ids: 
                if next_lora_type in self.engine_model_mapping[engine_id]:
                    continue
                
                self. engine_model_mapping[engine_id].append(next_lora_type)
                self.available_gpu_memorys[engine_id] -= self.lora_weight_sizes[next_lora_type]
                new_bt = self._calc_min_bottleneck_throughput(expected_lora_distribution)
                
                if new_bt >= best_bt:
                    current_lora_distribution[next_lora_type] += 1
                    num_lora_replicas += 1
                    best_bt = new_bt
                    update_flag = True
                    if next_lora_type in models_not_allocated:
                        models_not_allocated.remove(next_lora_type)
                    break
                else: 
                    self.engine_model_mapping[engine_id].remove(next_lora_type)
                    self.available_gpu_memorys[engine_id] += self.lora_weight_sizes[next_lora_type]
                    if models_not_allocated: 
                        update_flag = True
        
        # Build reverse mapping
        self.model_engine_mapping = {i: [] for i in range(self.num_models)}
        for engine_id, model_ids in self.engine_model_mapping.items():
            for model_id in model_ids: 
                self.model_engine_mapping[model_id].append(engine_id)

    
    def _calc_min_bottleneck_throughput(self, expected_lora_distribution: List[float]) -> float: # ✅
        """Calculate minimum bottleneck throughput (dLoRA's calc_min_bt)"""
        min_bt = float('inf')
        
        for lora_type in range(self.num_models):
            total_throughput = 0
            for engine_id in range(self.num_instances):
                if lora_type in self. engine_model_mapping[engine_id]:
                    total_throughput += self.available_gpu_memorys[engine_id]
            
            if expected_lora_distribution[lora_type] > 0:
                min_bt = min(min_bt, total_throughput / expected_lora_distribution[lora_type])
        
        return min_bt

    
    async def _apply_lora_placement(self): # ✅
        """
        Apply LoRA placement to all instances using SGLang's load/unload API.
        Uses official /load_lora_adapter and /unload_lora_adapter endpoints.
        """
        logger. info("Applying LoRA placement to instances...")
        session = await self._get_session()
        
        tasks = []
        for engine_id in range(self.num_instances):
            active_model_ids = self.engine_model_mapping[engine_id]
            tasks.append(self._sync_lora_adapters(session, engine_id, active_model_ids))
        
        results = await asyncio. gather(*tasks, return_exceptions=True)
        
        success_count = 0
        for engine_id, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to apply LoRA placement to engine {engine_id}: {result}")
            elif result. get("success", False):
                success_count += 1
                logger.info(f"✓ Engine {engine_id} LoRA placement applied:  {result.get('loaded_adapters', {})}")
            else:
                logger.warning(f"Engine {engine_id} LoRA placement partially failed: {result}")
        
        logger.info(f"LoRA placement applied to {success_count}/{self.num_instances} instances")

    
    async def _sync_lora_adapters( # ✅
        self,
        session: aiohttp.ClientSession,
        engine_id: int,
        target_model_ids: List[int]
    ) -> Dict:
        """
        Synchronize LoRA adapters on a single engine to match target_model_ids.
        Uses SGLang's official /load_lora_adapter and /unload_lora_adapter endpoints.
        
        Args:
            session: HTTP session
            engine_id: Target engine ID
            target_model_ids: List of model IDs that should be loaded
            
        Returns:
            Dict with sync results
        """
        base_url = self. instance_urls[engine_id]
        
        # Get currently loaded adapters
        currently_loaded = await self._get_loaded_adapters(session, base_url)
        
        print(f"Engine {engine_id} currently loaded adapters: {currently_loaded}", flush=True)
        
        target_lora_names = {f"lora{mid}" for mid in target_model_ids}
        currently_loaded_names = set(currently_loaded.keys())
        
        # Determine what to load and unload
        to_load = target_lora_names - currently_loaded_names
        to_unload = currently_loaded_names - target_lora_names
        
        results = {
            "engine_id": engine_id,
            "loaded":  [],
            "unloaded": [],
            "errors": [],
            "success": True,
        }
        
        # # Unload adapters first to free up capacity
        # for lora_name in to_unload:
        #     try:
        #         success = await self._unload_lora_adapter(session, base_url, lora_name)
        #         if success:
        #             results["unloaded"].append(lora_name)
        #             logger.debug(f"Engine {engine_id}:  Unloaded {lora_name}")
        #         else:
        #             results["errors"]. append(f"Failed to unload {lora_name}")
        #             results["success"] = False
        #     except Exception as e:
        #         error_msg = f"Error unloading {lora_name}: {e}"
        #         results["errors"].append(error_msg)
        #         results["success"] = False
        #         logger.error(f"Engine {engine_id}: {error_msg}")
        
        # # Load new adapters
        # for lora_name in to_load:
        #     # Extract model_id from lora_name (e.g., "lora0" -> 0)
        #     try:
        #         model_id = int(lora_name.replace("lora", ""))
        #         lora_path = LORA_PATH.get(lora_name)
                
        #         if not lora_path:
        #             error_msg = f"LoRA path not found for {lora_name}"
        #             results["errors"]. append(error_msg)
        #             results["success"] = False
        #             logger.error(f"Engine {engine_id}: {error_msg}")
        #             continue
                
        #         loaded_adapters = await self._load_lora_adapter(session, base_url, lora_name, lora_path)
                
        #         if loaded_adapters and lora_name in loaded_adapters:
        #             results["loaded"].append(lora_name)
        #             logger.debug(f"Engine {engine_id}: Loaded {lora_name} from {lora_path}")
        #         else:
        #             error_msg = f"Failed to load {lora_name}"
        #             results["errors"].append(error_msg)
        #             results["success"] = False
        #             logger.warning(f"Engine {engine_id}: {error_msg}")
                    
        #     except Exception as e:
        #         error_msg = f"Error loading {lora_name}: {e}"
        #         results["errors"].append(error_msg)
        #         results["success"] = False
        #         logger.error(f"Engine {engine_id}: {error_msg}")
        async def unload_one(lora_name):
            try:
                success = await self._unload_lora_adapter(session, base_url, lora_name)
                if success:
                    results["unloaded"].append(lora_name)
                    logger.debug(f"Engine {engine_id}:  Unloaded {lora_name}")
                else:
                    results["errors"].append(f"Failed to unload {lora_name}")
                    results["success"] = False
            except Exception as e:
                error_msg = f"Error unloading {lora_name}: {e}"
                results["errors"].append(error_msg)
                results["success"] = False
                logger.error(f"Engine {engine_id}: {error_msg}")

        await asyncio.gather(*(unload_one(lora_name) for lora_name in to_unload))

        # Load adapters in parallel
        async def load_one(lora_name):
            try:
                model_id = int(lora_name.replace("lora", ""))
                lora_path = LORA_PATH.get(lora_name)
                if not lora_path:
                    error_msg = f"LoRA path not found for {lora_name}"
                    results["errors"].append(error_msg)
                    results["success"] = False
                    logger.error(f"Engine {engine_id}: {error_msg}")
                    return
                loaded_adapters = await self._load_lora_adapter(session, base_url, lora_name, lora_path)
                if loaded_adapters and lora_name in loaded_adapters:
                    results["loaded"].append(lora_name)
                    logger.debug(f"Engine {engine_id}: Loaded {lora_name} from {lora_path}")
                else:
                    error_msg = f"Failed to load {lora_name}"
                    results["errors"].append(error_msg)
                    results["success"] = False
                    logger.warning(f"Engine {engine_id}: {error_msg}")
            except Exception as e:
                error_msg = f"Error loading {lora_name}: {e}"
                results["errors"].append(error_msg)
                results["success"] = False
                logger.error(f"Engine {engine_id}: {error_msg}")

        await asyncio.gather(*(load_one(lora_name) for lora_name in to_load))
        
        # Get final state
        final_loaded = await self._get_loaded_adapters(session, base_url)
        results["loaded_adapters"] = final_loaded
        
        return results

    
    async def _get_loaded_adapters( # ✅
        self,
        session: aiohttp.ClientSession,
        base_url: str
    ) -> Dict[str, str]:
        """
        Get currently loaded LoRA adapters from an instance using the dedicated endpoint.
        
        Returns:
            Dict mapping lora_name -> lora_path
        """
        try:
            url = f"{base_url}/get_loaded_lora_adapters"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to get loaded adapters from {base_url}: status {resp.status}")
                    return {}
                
                data = await resp.json()
                return data. get("loaded_adapters", {})
        except Exception as e:
            logger.error(f"Error getting loaded adapters from {base_url}: {e}")
            return {}

    
    async def _load_lora_adapter( # ✅
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        lora_name: str,
        lora_path: str
    ) -> Optional[Dict[str, str]]:
        """
        Load a LoRA adapter using SGLang's official /load_lora_adapter endpoint. 
        
        Args:
            session: HTTP session
            base_url: Instance base URL
            lora_name:  LoRA adapter name (e.g., "lora0")
            lora_path: Path to LoRA weights
            
        Returns:
            Dict of loaded adapters if successful, None otherwise
        """
        try:
            url = f"{base_url}/load_lora_adapter"
            payload = {
                "lora_name": lora_name,
                "lora_path": lora_path,
            }
            
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Failed to load LoRA {lora_name}: status {resp.status}, {error_text}")
                    return None
                
                result = await resp.json()
                
                if result.get("success", False):
                    return result. get("loaded_adapters", {})
                else:
                    error_msg = result.get("error_message", "Unknown error")
                    logger.error(f"Failed to load LoRA {lora_name}: {error_msg}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout loading LoRA {lora_name} from {base_url}")
            return None
        except Exception as e:
            logger.error(f"Error loading LoRA {lora_name}:  {e}")
            return None

    
    async def _unload_lora_adapter( # ✅
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        lora_name: str
    ) -> bool:
        """
        Unload a LoRA adapter using SGLang's official /unload_lora_adapter endpoint. 
        
        Args:
            session: HTTP session
            base_url: Instance base URL
            lora_name: LoRA adapter name to unload
            
        Returns: 
            True if successful, False otherwise
        """
        try: 
            url = f"{base_url}/unload_lora_adapter"
            payload = {
                "lora_name": lora_name,
            }
            
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Failed to unload LoRA {lora_name}: status {resp.status}, {error_text}")
                    return False
                
                result = await resp.json()
                
                if result. get("success", False):
                    return True
                else: 
                    error_msg = result. get("error_message", "Unknown error")
                    logger.error(f"Failed to unload LoRA {lora_name}:  {error_msg}")
                    return False
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout unloading LoRA {lora_name} from {base_url}")
            return False
        except Exception as e:
            logger.error(f"Error unloading LoRA {lora_name}: {e}")
            return False

    
    async def _get_session(self) -> aiohttp.ClientSession: # ✅
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300, connect=10, sock_read=300)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    
    async def _update_engine_costs(self): # ✅
        """
        Update engine execution costs based on current request distribution.
        Uses lightweight req_model_cnt endpoint instead of full engine stats.
        """
        session = await self._get_session()
        
        tasks = []
        for instance_url in self.instance_urls:
            url = f"{instance_url}/get_req_model_cnt"
            tasks.append(session.get(url, timeout=aiohttp.ClientTimeout(total=5)))
        
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, response in enumerate(responses):
                try:
                    data = await response.json()
                    
                    # Store the statistics
                    self.engine_req_model_cnt[i] = data.get("req_model_cnt", {})
                    self.engine_num_requests[i] = data.get("total_requests", 0)
                    self.engine_exec_cost[i] = data.get("exec_cost", 0.0)
                    
                    logger.debug(
                        f"Engine {i}:  {self.engine_num_requests[i]} requests, "
                        f"cost={self.engine_exec_cost[i]:.2f}, "
                        f"models={list(self.engine_req_model_cnt[i].keys())}"
                    )
                
                except Exception as e:
                    logger.error(f"Failed to parse req_model_cnt from instance {i}: {e}")
                    self.engine_exec_cost[i] = 0.0
                    self.engine_req_model_cnt[i] = {}
                    self.engine_num_requests[i] = 0
        
        except Exception as e:
            logger. error(f"Error updating engine costs: {e}")
            
    
    async def _apply_lora_placement_single(self, engine_id: int): # ✅
        """Apply LoRA placement to a single instance"""
        session = await self._get_session()
        active_model_ids = self.engine_model_mapping[engine_id]
        result = await self._sync_lora_adapters(session, engine_id, active_model_ids)
        
        if not result.get("success", False):
            logger.error(f"Failed to apply LoRA placement to engine {engine_id}: {result. get('errors', [])}")

    
    async def select_engine(self, request_id: str, model_id: int) -> int: # ✅
        """
        Select engine for a new request using dLoRA's dispatch logic.
        
        Args:
            request_id: Unique request identifier
            model_id: Model ID for this request
            
        Returns: 
            Selected engine ID
        """
        async with self.select_lock:
            self.expected_lora_distribution[model_id] += 1
            await self._update_engine_costs()
            
            # If model has no assigned engines, assign to least loaded
            if len(self.model_engine_mapping[model_id]) == 0:
                engine_id = min(
                    range(self.num_instances),
                    key=lambda eid: self.engine_request_count[eid]
                )
                self.model_engine_mapping[model_id] = [engine_id]
                self.engine_model_mapping[engine_id]. append(model_id)
                logger.info(f"Added model {model_id} to engine {engine_id} (no prior assignment)")
                
                # Apply LoRA placement
                await self._apply_lora_placement_single(engine_id)
            
            # Candidate engines with current model loaded
            candidate_engines = {}
            sub_engine_model_ids = {}
            
            for engine_id in self.model_engine_mapping[model_id]:
                candidate_engines[engine_id] = self.engine_exec_cost. get(engine_id, 0)
            
            # Dispatch migration:  try substituting unused LoRAs
            if self.migration_type == MigrationType.DISPATCH_MIG:
                for engine_id, req_model_cnt in self.engine_req_model_cnt.items():
                    if engine_id in candidate_engines:
                        continue
                    
                    for sub_model_id in self.engine_model_mapping[engine_id]:
                        if req_model_cnt. get(sub_model_id, 0) == 0 and len(self.model_engine_mapping[sub_model_id]) > 1:
                            load_cost = self.lora_load_costs.get(model_id, self.default_exec_time)
                            candidate_engines[engine_id] = self.engine_exec_cost.get(engine_id, 0) + load_cost
                            sub_engine_model_ids[engine_id] = sub_model_id
                            break
            
            # Select engine with minimum cost
            if not candidate_engines:
                # Fallback to least loaded engine
                min_engine_id = min(
                    range(self.num_instances),
                    key=lambda eid: self.engine_request_count[eid]
                )
                logger.warning(f"No candidate engines for model {model_id}, using least loaded: {min_engine_id}")
            else:
                min_engine_id = min(candidate_engines, key=candidate_engines.get)
            
            # Apply substitution if needed
            if min_engine_id in sub_engine_model_ids:
                sub_model_id = sub_engine_model_ids[min_engine_id]
                logger.info(f"Substituting model {sub_model_id} with {model_id} on engine {min_engine_id}")
                
                self.engine_model_mapping[min_engine_id].remove(sub_model_id)
                self.model_engine_mapping[sub_model_id].remove(min_engine_id)
                self.engine_model_mapping[min_engine_id].append(model_id)
                self.model_engine_mapping[model_id].append(min_engine_id)
                
                # Apply to instance
                await self._apply_lora_placement_single(min_engine_id)
            
            # Track request
            self.request_to_engine[request_id] = min_engine_id
            self.engine_request_count[min_engine_id] += 1
            
            logger.debug(f"Request {request_id} (model {model_id}) -> Engine {min_engine_id}")
            
            return min_engine_id

    
    def is_running(self) -> bool: # ✅
        """Check if background migration loop is running"""
        return self._running

    
    async def start_background_loop(self): # ✅
        """Start periodic migration loop"""
        if self._running:
            logger.warning("Background loop already running")
            return
        
        if self.migration_type == MigrationType.DISPATCH_ONLY:
            logger.info("Migration type is DISPATCH_ONLY, background loop not started")
            return
        
        self._running = True
        self. background_loop = asyncio.create_task(self. run_loop())
        logger.info(f"Background migration loop started (interval: {self. migration_interval}s)")

    
    async def run_loop(self): # ✅
        """Main periodic migration loop"""
        logger.info("Migration loop running...")
        
        while self._running:
            await asyncio.sleep(self.migration_interval)
            
            try:
                await self._perform_migration()
            except Exception as e:
                logger.error(f"Migration cycle failed: {e}", exc_info=True)
                
                
    # File: sglang+dLoRA/python/sglang/srt/instances/instance_manager.py

    async def _fetch_engine_stats( # ✅
        self,
        session: aiohttp.ClientSession,
        engine_id: int,
        url: str
    ) -> Optional[EngineStats]:
        """
        Fetch detailed engine statistics for migration decision.
        Equivalent to dLoRA's get_migration_info per engine.
        """
        try:  
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.warning(f"Engine {engine_id} stats endpoint returned status {resp.status}")
                    return None
                
                data = await resp.json()
                return EngineStats.from_response(data, engine_id)
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching engine stats from engine {engine_id}")
            return None
        except Exception as e:  
            logger.debug(f"Error fetching engine stats from engine {engine_id}: {e}")
            return None


    async def _fetch_all_engine_stats(self): # ✅
        """
        Fetch comprehensive stats from all engines for migration decision.
        Equivalent to dLoRA's get_migration_info() method.
        
        This includes:
        - Request metadata (for ILP solver)
        - Model execution time (for cost estimation)
        - Free GPU pages (for capacity planning)
        
        Updates:
        - self.reqs_metadata: {engine_id: [RequestMetadata]}
        - self.model_exec_info: {model_id: [count, total_time]}
        - self.model_avg_exec_time: [avg_time per model]
        """
        session = await self._get_session()
        
        tasks = []
        for engine_id in range(self.num_instances):
            url = f"{self.instance_urls[engine_id]}/get_migration_info"
            tasks.append(self._fetch_engine_stats(session, engine_id, url))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for engine_id, result in enumerate(results):
            if isinstance(result, Exception):
                logger. error(f"Failed to fetch stats from engine {engine_id}: {result}")
                self.reqs_metadata[engine_id] = []
                continue
            
            if result is None:  
                logger.warning(f"Engine {engine_id} returned no stats")
                self.reqs_metadata[engine_id] = []
                continue
            
            # Store request metadata for this engine
            self.engine_stats[engine_id] = result
            self.reqs_metadata[engine_id] = result.req_metadata
            
            # Aggregate model execution info across all engines (like dLoRA)
            for model_id_str, exec_info in result.model_exec_time.items():
                try:
                    # Handle both string and int model IDs
                    model_id = int(model_id_str)

                    if 0 <= model_id < self.num_models:
                        # exec_info is (count, total_time)
                        self.model_exec_info[model_id][0] += exec_info[0]  # count
                        self.model_exec_info[model_id][1] += exec_info[1]  # total time
                    else:
                        # Log base model (-1) or out-of-range models at debug level
                        logger.debug(f"Skipping model exec info for model ID {model_id} (out of range or base model)")
                except (ValueError, TypeError) as e:
                    # Log other invalid model_id strings (like 'null' or UUIDs) at debug level
                    logger.debug(f"Error processing model exec info for '{model_id_str}': {e}")
                    continue
        
        # Calculate average execution times (like dLoRA)
        for model_id, exec_info in self.model_exec_info.items():
            if exec_info[0] > 0:
                self.model_avg_exec_time[model_id] = exec_info[1] / exec_info[0]
            else:
                # Use default if no data available
                self.model_avg_exec_time[model_id] = self.default_exec_time
        
        logger.debug(f"Updated model avg exec times: {self.model_avg_exec_time}")
        logger.info(f"Fetched stats from {len([r for r in results if r is not None])}/{self.num_instances} engines")


    def _check_migration_needed(self) -> Tuple[bool, Optional[List[int]]]:
        """
        Check if migration is needed based on dLoRA's criteria.
        Returns both decision and the engines to participate in ILP. 
        
        Criteria (from dLoRA):
        1. KV cache pressure imbalance: one engine >90% full, another <90% full
        2. Request count imbalance: max_requests - avg_requests >= threshold
        
        Returns:
            (need_migration, ilp_engines):
                - need_migration: bool, whether to perform migration
                - ilp_engines: List[int] or None, engines to include in ILP solver
        """
        # Check 1: KV cache pressure imbalance (PRIORITY)
        engine_page_cnt = {
            i: sum(req["num_blocks"] for req in self.reqs_metadata.get(i, []))
            for i in range(self.num_instances)
        }

        if engine_page_cnt:
            sorted_pages = sorted(engine_page_cnt.items(), key=lambda x: x[1])
            most_page_engine_id, most_page_cnt = sorted_pages[-1]
            least_page_engine_id, least_page_cnt = sorted_pages[0]

            max_pages = self.num_gpu_pages[most_page_engine_id]
            min_pages = self.num_gpu_pages[least_page_engine_id]

            if max_pages > 0 and min_pages > 0:
                most_usage = most_page_cnt / max_pages
                least_usage = least_page_cnt / min_pages if min_pages > 0 else 0
                
                # KV cache imbalance detected (like dLoRA)
                if most_usage >= 0.9 and least_usage < 0.9:
                    ilp_engines = sorted([most_page_engine_id, least_page_engine_id])
                    logger.info(
                        f"Migration needed: KV cache imbalance detected\n"
                        f"  Engine {most_page_engine_id}:  {most_page_cnt}/{max_pages} pages ({most_usage:.1%})\n"
                        f"  Engine {least_page_engine_id}: {least_page_cnt}/{min_pages} pages ({least_usage:.1%})\n"
                        f"  ILP engines: {ilp_engines}"
                    )
                    return True, ilp_engines
        
        # Check 2: Request count imbalance (SECONDARY)
        engine_req_cnt = {i: len(reqs) for i, reqs in self.reqs_metadata.items()}
        num_reqs = sum(engine_req_cnt. values())
        
        if num_reqs == 0:
            logger.debug("No active requests, migration not needed")
            return False, None
        
        # Sort by request count
        sorted_req_cnt = sorted(engine_req_cnt.items(), key=lambda x: x[1])
        
        # Try to find imbalanced engines (like dLoRA's matching algorithm)
        matched_engines = []
        remaining_engines = list(sorted_req_cnt)
        
        while remaining_engines:
            avg_num_reqs = sum(cnt for _, cnt in remaining_engines) / len(remaining_engines)
            most_req_engine_id, most_req_engine_cnt = remaining_engines. pop()
            
            delta = most_req_engine_cnt - avg_num_reqs
            
            # Not imbalanced enough
            if delta < self.migration_req_thres:
                break
            
            # Try to match with underloaded engines
            for engine_id, cnt in remaining_engines: 
                if delta <= 0 or len(matched_engines) > 0:
                    break
                
                to_fill = avg_num_reqs - cnt
                if to_fill <= 0:
                    break
                
                to_fill = min(to_fill, delta)
                matched_engines.append(engine_id)
                delta -= to_fill
            
            # Found matches
            if matched_engines:
                ilp_engines = sorted([most_req_engine_id] + matched_engines)
                logger.info(
                    f"Migration needed: Request count imbalance detected\n"
                    f"  Max requests: {most_req_engine_cnt}, Avg:  {avg_num_reqs:.1f}\n"
                    f"  Imbalance: {delta:.1f} (threshold: {self.migration_req_thres})\n"
                    f"  ILP engines: {ilp_engines}"
                )
                return True, ilp_engines
        
        logger.debug(
            f"No migration needed (system balanced:  "
            f"max={max(engine_req_cnt. values()) if engine_req_cnt else 0}, "
            f"avg={num_reqs/len(engine_req_cnt) if engine_req_cnt else 0:.1f})"
        )
        return False, None


    def _prepare_ilp_problem(self, ilp_engines: List[int]) -> Dict:
        """
        Prepare ILP problem with index remapping.
        Equivalent to dLoRA's remapping logic in migration_schedule.
        
        Args:
            ilp_engines: List of engine IDs to include in ILP
            
        Returns:
            Dict with: 
            - ilp_reqs_metadata: List of RequestMetadata with remapped indices
            - ilp_num_groups: Number of engines in ILP
            - ilp_num_models: Number of models in ILP
            - ilp_num_gpu_blocks: GPU blocks per engine
            - ilp_num_cpu_blocks: CPU blocks per engine
            - ilp_lora_capacity: LoRA capacity per engine
            - ilp_model_avg_exec_time:  Avg exec time per model
            - ilp_model_engine_mapping: Model->engine mapping
            - ilp_engine_mapping: Original->ILP engine ID mapping
            - ilp_model_mapping: Original->ILP model ID mapping
        """
        # Create engine index mapping (original engine_id -> ilp engine_id)
        ilp_engine_mapping = {ilp_engines[i]: i for i in range(len(ilp_engines))}
        
        # Collect all models used by ILP engines
        ilp_models = set()
        for engine_id in ilp_engines: 
            for model_id in self.engine_model_mapping[engine_id]:
                ilp_models.add(model_id)
        ilp_models = sorted(list(ilp_models))
        
        # Create model index mapping (original model_id -> ilp model_id)
        ilp_model_mapping = {ilp_models[i]: i for i in range(len(ilp_models))}
        
        # Collect and remap request metadata
        ilp_reqs_metadata = []
        for engine_id in ilp_engines: 
            for req_metadata in self.reqs_metadata[engine_id]:
                # Remap engine and model IDs
                req_metadata.engine_id = ilp_engine_mapping[engine_id]
                req_metadata.model_id = ilp_model_mapping[req_metadata.model_id]
                ilp_reqs_metadata.append(req_metadata)
        
        # Prepare ILP parameters
        ilp_num_groups = len(ilp_engines)
        ilp_num_models = len(ilp_models)
        ilp_num_gpu_pages = [self.num_gpu_pages[engine_id] for engine_id in ilp_engines]
        ilp_num_cpu_pages = [0] * ilp_num_groups  # SGLang doesn't use CPU blocks
        ilp_lora_capacity = [len(self.engine_model_mapping[engine_id]) for engine_id in ilp_engines]
        ilp_model_avg_exec_time = [self.model_avg_exec_time[model_id] for model_id in ilp_models]
        
        # Remap model->engine mapping
        ilp_model_engine_mapping = {i: [] for i in range(ilp_num_models)}
        for engine_id, model_ids in self.engine_model_mapping.items():
            if engine_id not in ilp_engines:
                continue
            for model_id in model_ids: 
                if model_id in ilp_model_mapping:
                    ilp_model_id = ilp_model_mapping[model_id]
                    ilp_engine_id = ilp_engine_mapping[engine_id]
                    ilp_model_engine_mapping[ilp_model_id].append(ilp_engine_id)
        
        logger.info(
            f"ILP problem prepared:\n"
            f"  Engines: {ilp_num_groups} ({ilp_engines})\n"
            f"  Models: {ilp_num_models} ({ilp_models})\n"
            f"  Requests: {len(ilp_reqs_metadata)}\n"
            f"  GPU pages: {ilp_num_gpu_pages}\n"
            f"  LoRA capacity: {ilp_lora_capacity}"
        )
        
        return {
            "ilp_reqs_metadata": ilp_reqs_metadata,
            "ilp_num_groups": ilp_num_groups,
            "ilp_num_models": ilp_num_models,
            "ilp_num_gpu_pages": ilp_num_gpu_pages,
            "ilp_num_cpu_pages": ilp_num_cpu_pages,
            "ilp_lora_capacity": ilp_lora_capacity,
            "ilp_model_avg_exec_time": ilp_model_avg_exec_time,
            "ilp_model_engine_mapping":  ilp_model_engine_mapping,
            "ilp_engine_mapping": ilp_engine_mapping,
            "ilp_model_mapping": ilp_model_mapping,
            "ilp_engines": ilp_engines,
            "ilp_models": ilp_models,
        }


    async def _perform_migration(self):
        """
        Perform migration decision and execution.
        Follows dLoRA's migration_schedule logic exactly.
        
        Steps:
        1. Fetch current stats from all engines (get_migration_info)
        2. Check if migration is needed and select ILP engines
        3. Prepare ILP problem with remapped indices
        4. Solve ILP for optimal migration plan
        5. Execute the migration plan
        6. Update internal state
        """
        async with self. migration_lock:
            logger.info("=" * 86)
            logger.info("Starting migration cycle...")
            logger.info("=" * 86)
            
            # # Step 1: Fetch current stats (like dLoRA's get_migration_info)
            # try:
            #     await self._fetch_all_engine_stats()
            # except Exception as e:
            #     logger.error(f"Failed to fetch engine stats: {e}", exc_info=True)
            #     logger.info("=" * 86)
            #     return
            
            # # Step 2: Check if migration is needed and get ILP engines
            # try:
            #     need_migration, ilp_engines = self._check_migration_needed()
            # except Exception as e:
            #     logger.error(f"Error checking migration criteria: {e}", exc_info=True)
            #     logger.info("=" * 86)
            #     return
            
            # if not need_migration or ilp_engines is None:
            #     logger.info("✓ No migration needed (system balanced)")
            #     logger.info("=" * 86)
            #     return
            
            # logger.info(f"⚠ Migration needed for engines: {ilp_engines}")
            # logger.info("Preparing ILP problem...")
            
            # # Step 3: Prepare ILP problem (like dLoRA's remapping)
            # try:
            #     ilp_problem = self._prepare_ilp_problem(ilp_engines)
            # except Exception as e:
            #     logger.error(f"Failed to prepare ILP problem:  {e}", exc_info=True)
            #     logger.info("=" * 86)
            #     return
            
            # # Step 4: Solve ILP
            # logger.info("Running ILP solver...")
            # try:
            #     migration_plan = await self._solve_migration_ilp(ilp_problem, ilp_engines)
            # except Exception as e:
            #     logger.error(f"ILP solver error: {e}", exc_info=True)
            #     logger.info("=" * 86)
            #     return
            
            # if migration_plan is None:
            #     logger.warning("✗ Migration ILP solver failed or returned no plan")
            #     logger.info("=" * 86)
            #     return
            
            # # Step 5: Execute migration
            # logger.info("Executing migration plan...")
            # try:
            #     await self._execute_migration(migration_plan, ilp_engines)
            # except Exception as e:
            #     logger. error(f"Migration execution failed: {e}", exc_info=True)
            #     logger.info("=" * 86)
            #     return
            
            logger.info("✓ Migration cycle complete")
            logger.info("=" * 86)


    async def _solve_migration_ilp(
        self, 
        ilp_problem: Dict,
        ilp_engines: List[int]
    ) -> Optional[Dict]:
        """
        Solve migration ILP using dLoRA's ILP solver with remapped indices.
        
        Args:
            ilp_problem:  Prepared ILP problem from _prepare_ilp_problem
            ilp_engines: Original engine IDs participating in ILP
            
        Returns:
            Dict containing:  
            - req_migration: {src_engine:  {dst_engine: [req_ids]}} (original engine IDs)
            - lora_weights: {engine_id: [model_ids]} (original IDs)
            - lora_counts: [count per model]
            Or None if solver fails
        """
        try:
            ilp_reqs_metadata = ilp_problem["ilp_reqs_metadata"]
            
            if len(ilp_reqs_metadata) == 0:
                logger.info("No requests to migrate")
                return None
            
            logger.info(f"Running ILP solver with {len(ilp_reqs_metadata)} requests...")
            
            # Create ILP solver with remapped problem
            from sglang.srt.instances.migration_ilp import MigrationILP
            
            ilp = MigrationILP(
                reqs_metadata=ilp_reqs_metadata,
                num_groups=ilp_problem["ilp_num_groups"],
                num_models=ilp_problem["ilp_num_models"],
                engine_gpu_blocks=ilp_problem["ilp_num_gpu_pages"],
                engine_cpu_blocks=ilp_problem["ilp_num_cpu_pages"],
                engine_lora_capacity=ilp_problem["ilp_lora_capacity"],
                lora_exec_time=ilp_problem["ilp_model_avg_exec_time"],
                alpha=0.05,  # Migration overhead coefficient
                bw=PCIE_BANDWIDTH / self.cache_page_size[0],  # Bandwidth in blocks/sec
                model_engine_mapping=ilp_problem["ilp_model_engine_mapping"],
            )
            
            # Solve (run in thread pool to avoid blocking)
            req_migration_mapping, lora_weight_mapping, lora_weight_cnt = await asyncio.get_event_loop().run_in_executor(
                None, ilp.solve
            )
            
            if req_migration_mapping is None: 
                logger.warning("ILP solver returned no solution")
                return None
            
            # # Reverse map back to original engine and model IDs (like dLoRA)
            # ilp_engines_list = ilp_problem["ilp_engines"]
            # ilp_models_list = ilp_problem["ilp_models"]
            
            # # Reverse map request migrations
            # original_req_migration = {i: {j: [] for j in range(self.num_instances)} for i in range(self.num_instances)}
            # for src_ilp_id, dst_mapping in req_migration_mapping.items():
            #     src_engine_id = ilp_engines_list[src_ilp_id]
            #     for dst_ilp_id, req_ids in dst_mapping.items():
            #         dst_engine_id = ilp_engines_list[dst_ilp_id]
            #         if req_ids:
            #             logger.debug(f"Move {len(req_ids)} requests from engine {src_engine_id} to {dst_engine_id}")
            #             original_req_migration[src_engine_id][dst_engine_id] = req_ids
            
            # # Reverse map LoRA weights
            # original_lora_weights = {}
            # for ilp_engine_id, ilp_model_ids in lora_weight_mapping. items():
            #     engine_id = ilp_engines_list[ilp_engine_id]
            #     original_model_ids = [ilp_models_list[mid] for mid in ilp_model_ids]
            #     original_lora_weights[engine_id] = original_model_ids
            
            # # Update global lora weight counts
            # original_lora_counts = [0] * self.num_models
            # for engine_id, model_ids in original_lora_weights.items():
            #     for model_id in model_ids:
            #         original_lora_counts[model_id] += 1
            
            # Count total migrations
            total_migrations = sum(
                len(req_ids)
                for dst_mapping in req_migration_mapping.values()
                for req_ids in dst_mapping.values()
            )
            
            logger.info(f"ILP solution:  {total_migrations} request migrations planned")
            logger.debug(f"LoRA placement: {lora_weight_mapping}")
            
            return {
                "req_migration":  req_migration_mapping,
                "lora_weights": lora_weight_mapping,
                "lora_counts": lora_weight_cnt,
            }
        
        except Exception as e: 
            logger.error(f"ILP solver failed: {e}", exc_info=True)
            return None


    async def _execute_migration(self, migration_plan: Dict, ilp_engines: List[int]):
        """
        Execute migration plan. 
        Follows dLoRA's execution order exactly.
        
        Steps:
        1. Execute request migrations (fetch -> insert -> abort)
        2. Update engine_model_mapping from ILP solution
        3. Adjust LoRA adapters on all engines
        4. Run find_best_lora_weight_schedule to refine placement
        5. Update internal state
        
        Args:
            migration_plan: Dict from _solve_migration_ilp
            ilp_engines: Original engine IDs that participated in ILP
        """
        req_migration = migration_plan["req_migration"]
        lora_weights = migration_plan["lora_weights"]
        lora_counts = migration_plan["lora_counts"]
        
        logger.info(f"Migration plan:\n  LoRA adjustments: {lora_weights}\n  Request migrations: {sum(len(reqs) for dsts in req_migration.values() for reqs in dsts. values())} requests")
        
        # Step 1: Update engine_model_mapping from ILP solution (like dLoRA)
        logger.info("Step 1: Updating LoRA placement from ILP solution...")
        for engine_id, model_ids in lora_weights.items():
            self.engine_model_mapping[engine_id] = model_ids
        
        # Rebuild model_engine_mapping
        self.model_engine_mapping = {i: [] for i in range(self.num_models)}
        for engine_id, model_ids in self.engine_model_mapping.items():
            for model_id in model_ids: 
                self.model_engine_mapping[model_id].append(engine_id)
        
        logger.info(f"Updated engine_model_mapping: {self. engine_model_mapping}")
        
        # Step 2: Refine placement with find_best_lora_weight_schedule (like dLoRA)
        logger.info("Step 2: Refining LoRA placement...")
        try:
            self.find_best_lora_weight_schedule(
                is_init=False,
                expected_lora_distribution=self.expected_lora_distribution,
                current_lora_distribution=lora_counts,
                engine_lora_capacity=self.engine_lora_capacity
            )
        except Exception as e:
            logger.error(f"Failed to refine LoRA placement: {e}", exc_info=True)
        
        # Step 3: Adjust LoRA adapters on engines (like dLoRA)
        logger.info("Step 3: Syncing LoRA adapters on engines...")
        try:
            await self._execute_lora_adjustment(self.engine_model_mapping)
        except Exception as e: 
            logger.error(f"LoRA adjustment failed: {e}", exc_info=True)
            
        # Step 4: Migrate requests FIRST (like dLoRA)
        logger.info("Step 4: Migrating requests...")
        try:
            await self._execute_request_migration(req_migration)
        except Exception as e: 
            logger.error(f"Request migration failed: {e}", exc_info=True)
            return
        
        logger.info("✓ Migration execution complete")


    async def _execute_lora_adjustment(self, lora_weights: Dict[int, List[int]]):
        """
        Adjust LoRA adapters on all engines based on migration plan.
        
        Args:
            lora_weights: {engine_id: [model_ids]} - target LoRA placement
        """
        session = await self._get_session()
        
        tasks = []
        for engine_id, model_ids in lora_weights.items():
            if engine_id >= self.num_instances:
                logger.warning(f"Invalid engine_id {engine_id}, skipping")
                continue
            tasks.append(self._sync_lora_adapters(session, engine_id, model_ids))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 0
        for engine_id, result in enumerate(results):
            if isinstance(result, Exception):
                logger. error(f"Failed to adjust LoRA on engine {engine_id}: {result}")
            elif result. get("success", False):
                success_count += 1
            else:
                logger.warning(f"LoRA adjustment on engine {engine_id} had errors: {result. get('errors', [])}")
        
        logger.info(f"LoRA adjustment complete: {success_count}/{len(lora_weights)} engines successful")


    async def _execute_request_migration(self, req_migration: Dict[int, Dict[int, List[str]]]):
        """
        Migrate requests between engines.
        Follows dLoRA's flow:  Fetch -> Insert -> Abort
        
        Args:
            req_migration: {src_engine:  {dst_engine: [request_ids]}}
        """
        session = await self._get_session()
        
        # Step 1: Fetch all seq_groups to migrate
        fetch_tasks = []
        fetch_info = []
        
        for src_engine_id, dst_mapping in req_migration.items():
            for dst_engine_id, request_ids in dst_mapping.items():
                if len(request_ids) == 0:
                    continue
                
                if src_engine_id >= self.num_instances or dst_engine_id >= self.num_instances:
                    logger.warning(f"Invalid engine IDs: src={src_engine_id}, dst={dst_engine_id}")
                    continue
                
                url = f"{self.instance_urls[src_engine_id]}/fetch_seq_groups"
                fetch_tasks.append(self._fetch_seq_groups(session, url, request_ids))
                fetch_info.append((src_engine_id, dst_engine_id, request_ids))
        
        if not fetch_tasks:
            logger.info("No requests to migrate")
            return
        
        logger.info(f"Fetching {len(fetch_tasks)} request groups...")
        seq_groups_list = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Step 2: Insert to destination engines
        insert_tasks = []
        insert_info = []
        
        for idx, (src_engine_id, dst_engine_id, request_ids) in enumerate(fetch_info):
            if idx >= len(seq_groups_list):
                continue
            
            seq_groups = seq_groups_list[idx]
            if isinstance(seq_groups, Exception):
                logger.error(f"Failed to fetch seq_groups from engine {src_engine_id}:  {seq_groups}")
                continue
            
            if not seq_groups:
                logger. warning(f"No seq_groups fetched from engine {src_engine_id} for {len(request_ids)} requests")
                continue
            
            url = f"{self.instance_urls[dst_engine_id]}/insert_seq_groups"
            insert_tasks.append(self._insert_seq_groups(session, url, seq_groups))
            insert_info.append((src_engine_id, dst_engine_id, request_ids))
        
        if not insert_tasks:
            logger.warning("No seq_groups to insert")
            return
        
        logger.info(f"Inserting {len(insert_tasks)} request groups...")
        insert_results = await asyncio.gather(*insert_tasks, return_exceptions=True)
        
        # Step 3: Abort from source engines
        abort_tasks = []
        
        for idx, (src_engine_id, dst_engine_id, request_ids) in enumerate(insert_info):
            if idx >= len(insert_results):
                continue
            
            insert_result = insert_results[idx]
            if isinstance(insert_result, Exception):
                logger. error(f"Failed to insert to engine {dst_engine_id}:  {insert_result}")
                continue
            
            inserted_count = insert_result.get("count", 0)
            if inserted_count > 0:
                url = f"{self.instance_urls[src_engine_id]}/abort_requests"
                abort_tasks.append(self._abort_requests(session, url, request_ids))
                logger.info(f"Migrated {inserted_count} requests:  Engine {src_engine_id} -> {dst_engine_id}")
            else:
                logger.warning(f"Failed to insert requests from engine {src_engine_id} to {dst_engine_id}")
        
        if abort_tasks:
            logger.info(f"Aborting {len(abort_tasks)} request groups from source engines...")
            await asyncio. gather(*abort_tasks, return_exceptions=True)


    async def _fetch_seq_groups(
        self,
        session: aiohttp.ClientSession,
        url: str,
        request_ids: List[str]
    ) -> Optional[List[Dict]]:
        """Fetch sequence groups from an engine"""
        try:  
            async with session.post(
                url, 
                json={"request_ids": request_ids},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Failed to fetch seq_groups from {url}: status {resp.status}, {error_text}")
                    return None
                data = await resp.json()
                return data.get("seq_groups", [])
        except asyncio. TimeoutError:
            logger. error(f"Timeout fetching seq_groups from {url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching seq groups from {url}: {e}")
            return None


    async def _insert_seq_groups(
        self,
        session: aiohttp.ClientSession,
        url: str,
        seq_groups:  List[Dict]
    ) -> Dict[str, Any]:
        """Insert sequence groups to an engine"""
        try: 
            async with session.post(
                url, 
                json={"seq_groups": seq_groups},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp. text()
                    logger.error(f"Failed to insert seq_groups to {url}: status {resp. status}, {error_text}")
                    return {"count": 0}
                return await resp.json()
        except asyncio.TimeoutError:
            logger.error(f"Timeout inserting seq_groups to {url}")
            return {"count": 0}
        except Exception as e:
            logger.error(f"Error inserting seq groups to {url}: {e}")
            return {"count": 0}


    async def _abort_requests(
        self,
        session:  aiohttp.ClientSession,
        url: str,
        request_ids: List[str]
    ) -> Optional[Dict]:
        """Abort requests on an engine"""
        try: 
            async with session.post(
                url, 
                json={"request_ids": request_ids},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger. warning(f"Failed to abort requests at {url}: status {resp.status}, {error_text}")
                    return None
                return await resp. json()
        except asyncio. TimeoutError:
            logger. error(f"Timeout aborting requests at {url}")
            return None
        except Exception as e: 
            logger.error(f"Error aborting requests at {url}: {e}")
            return None
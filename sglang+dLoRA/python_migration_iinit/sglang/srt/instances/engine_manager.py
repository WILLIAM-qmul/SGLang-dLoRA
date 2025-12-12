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
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from sglang.srt.instances.lora_config_paths import NUM_LORAS
from sglang.srt.instances.lora_init import get_lora_sizes_for_instance_manager
from sglang.srt.instances.migration_ilp import MigrationILP


logger = logging.getLogger(__name__)


_GB = 1 << 30
PCIE_BANDWIDTH = 32 * _GB


class MigrationType(Enum):
    DISPATCH_ONLY = 1
    DISPATCH_MIG = 2
    PERIOD_MIG = 3


@dataclass
class RequestMetadata:
    request_id: str
    model_id: int
    engine_id: int
    num_blocks: int
    in_gpu: bool
    prompt_length: int = 0
    output_length: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict, engine_id: int) -> "RequestMetadata":
        return cls(
            request_id=data["request_id"],
            model_id=data. get("model_id", 0),
            engine_id=engine_id,
            num_blocks=data["num_blocks"],
            in_gpu=data["in_gpu"],
            prompt_length=data.get("prompt_length", 0),
            output_length=data.get("output_length", 0),
        )


@dataclass
class EngineStats:
    engine_id: int
    num_requests: int
    req_model_cnt: Dict[int, int]
    num_free_gpu_pages: int
    lora_capacity: int
    active_models: List[int]
    req_metadata: List[RequestMetadata]
    model_exec_time: Dict[int, Tuple[int, float]]
    available_gpu_memory: float
    cache_page_size: int
    
    @classmethod
    def from_response(cls, data: Dict, engine_id: int) -> "EngineStats":
        req_metadata = [
            RequestMetadata.from_dict(req_data, engine_id)
            for req_data in data. get("req_metadata", [])
        ]
        
        return cls(
            engine_id=engine_id,
            num_requests=data["num_requests"],
            req_model_cnt=data. get("req_model_cnt", {}),
            num_free_gpu_pages=data["num_free_gpu_pages"],
            lora_capacity=data. get("lora_capacity", 8),
            active_models=data.get("active_models", []),
            req_metadata=req_metadata,
            model_exec_time=data.get("model_exec_time", {}),
            available_gpu_memory=data.get("available_gpu_memory", 10.0 * _GB),
            cache_page_size=data.get("cache_page_size", 4096),
        )


class InstanceManager:
    def __init__(
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
        # logger.info("Calculating LoRA weight sizes...")
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

        # logger.info("LoRA weight sizes calculated:")
        print("LoRA weight sizes calculated:")
        for model_id in range(num_models):
            size_bytes = self.lora_weight_sizes.get(model_id, 0)
            load_s = self.lora_load_costs.get(model_id, 0)
            print(f"  Model {model_id}: {size_bytes:.2f} bytes, load time: {load_s:.2f} s", flush=True)
        # ============================================================
        
        # Statistics
        self.engine_stats: Dict[int, EngineStats] = {}
        self.global_model_request_count: Dict[int, int] = {i: 0 for i in range(num_models)}
        self.engine_exec_cost: Dict[int, float] = {}
        self.engine_req_model_cnt: Dict[int, Dict[int, int]] = {}
        self.model_exec_info: Dict[int, List[float]] = {i: [0, 0.0] for i in range(num_models)}
        self.model_avg_exec_time: List[float] = [default_exec_time] * num_models
        self.reqs_metadata: Dict[int, List[RequestMetadata]] = {}
        
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
        self.lora_weight_sizes: List[int] = [0] * num_instances
        
        # Engine capacity
        self.engine_lora_capacity: List[int] = [lora_capacity_per_engine] * num_instances
        
        logger.info(f"InstanceManager initialized: {num_instances} instances, {num_models} models")
        

    async def initialize(self):
        """Async initialization"""
        logger.info("Fetching initial stats from all instances...")
        await self._fetch_instance_stats()
        
        logger.info("Initializing LoRA placement...")
        self._initialize_lora_placement()
        
        logger.info("Applying initial LoRA placement to instances...")
        await self._apply_lora_placement()
        
        logger.info("InstanceManager initialization complete")
        

    async def _fetch_instance_stats(self):
        """Fetch initial stats from all instances"""
        session = await self._get_session()

        tasks = []
        for engine_id in range(self.num_instances):
            url = f"{self.instance_urls[engine_id]}/get_engine_stats"
            tasks.append(self._fetch_engine_stats(session, engine_id, url))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for engine_id, result in enumerate(results):
            self.engine_stats[engine_id] = result
            self.num_gpu_pages[engine_id] = result.num_free_gpu_pages
            self.available_gpu_memorys[engine_id] = result.available_gpu_memory
            self.cache_page_size = result.cache_page_size
                
        # 打印所有实例的统计信息
        print("Fetched instance stats:")
        for engine_id, stats in self.engine_stats.items():
            print(f"Engine {engine_id}: {stats}")
            

    def _initialize_lora_placement(self):
        """Initialize LoRA placement using dLoRA's algorithm"""
        logger.info("Running LoRA placement optimization...")
        
        current_lora_distribution = [0] * self.num_models
        num_lora_replicas = 0
        best_bt = 0
        update_flag = True
        
        engine_ids = list(range(self.num_instances))
        models_not_allocated = list(range(self.num_models))
        
        while update_flag:
            update_flag = False
            
            # Select next LoRA
            next_lora_type = 0
            for i in range(self.num_models):
                ratio_i = (current_lora_distribution[i] / (num_lora_replicas + 1e-7)) - self.expected_lora_distribution[i]
                ratio_next = (current_lora_distribution[next_lora_type] / (num_lora_replicas + 1e-7)) - self.expected_lora_distribution[next_lora_type]
                if ratio_i < ratio_next:
                    next_lora_type = i
            
            if next_lora_type not in models_not_allocated and models_not_allocated:
                next_lora_type = models_not_allocated[0]
            
            # Sort engines
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
                self. available_gpu_memorys[engine_id] -= self.lora_weight_sizes[engine_id]
                
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
                    self.available_gpu_memorys[engine_id] += self.lora_weight_sizes[engine_id]
                    
                    if models_not_allocated:
                        update_flag = True
        
        # Ensure minimum replicas
        min_replicas = self. num_instances + self.num_models - 1
        next_engine_id = 0
        next_lora_type = random.randint(0, self. num_models - 1)
        
        while num_lora_replicas < min_replicas:
            while (len(self.engine_model_mapping[next_engine_id]) >= self.num_models or
                   next_lora_type in self.engine_model_mapping[next_engine_id]):
                next_engine_id = (next_engine_id + 1) % self.num_instances
            
            self.engine_model_mapping[next_engine_id].append(next_lora_type)
            self.available_gpu_memorys[next_engine_id] -= self.lora_weight_sizes[next_engine_id]
            next_engine_id = (next_engine_id + 1) % self. num_instances
            current_lora_distribution[next_lora_type] += 1
            num_lora_replicas += 1
        
        # Build reverse mapping
        self.model_engine_mapping = {i: [] for i in range(self.num_models)}
        for engine_id, model_ids in self.engine_model_mapping.items():
            for model_id in model_ids:
                self.model_engine_mapping[model_id].append(engine_id)
        
        logger.info(f"LoRA placement: {num_lora_replicas} replicas")
        logger.info(f"Engine model mapping: {self.engine_model_mapping}")
        
        
    def find_best_lora_weight_schedule(
        self,
        expected_lora_distribution: List[float],
        current_lora_distribution: List[int] = None,
        engine_lora_capacity: List[int] = None
    ):
        """
        贪心分配 LoRA 副本，使分布尽量贴合 expected_lora_distribution。
        只负责主循环，不做冷启动补齐。
        """
        if current_lora_distribution is None:
            current_lora_distribution = [0 for _ in range(self.num_models)]
        num_lora_replicas = sum(current_lora_distribution)
        best_bt = 0
        update_flag = True

        engine_ids = list(range(self.num_instances))
        models_not_allocated = [i for i in range(self.num_models) if len(self.model_engine_mapping[i]) == 0]

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

            # 排序引擎
            if engine_lora_capacity is None:
                engine_ids = sorted(engine_ids, key=lambda eid: len(self.engine_model_mapping[eid]))
            else:
                engine_ids = sorted(
                    engine_ids,
                    key=lambda eid: engine_lora_capacity[eid] - len(self.engine_model_mapping[eid]),
                    reverse=True
                )
                engine_ids = [eid for eid in engine_ids if engine_lora_capacity[eid] > len(self.engine_model_mapping[eid])]

            for engine_id in engine_ids:
                if next_lora_type in self.engine_model_mapping[engine_id]:
                    continue
                self.engine_model_mapping[engine_id].append(next_lora_type)
                self.available_gpu_memorys[engine_id] -= self.lora_weight_sizes[engine_id]
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
                    self.available_gpu_memorys[engine_id] += self.lora_weight_sizes[engine_id]
                    if models_not_allocated:
                        update_flag = True
                        
        # Build reverse mapping
        self.model_engine_mapping = {i: [] for i in range(self.num_models)}
        for engine_id, model_ids in self.engine_model_mapping.items():
            for model_id in model_ids:
                self.model_engine_mapping[model_id].append(engine_id)
                

    def _calc_min_bottleneck_throughput(self, expected_lora_distribution: List[float]) -> float:
        """Calculate minimum bottleneck throughput"""
        min_bt = float('inf')
        
        for lora_type in range(self.num_models):
            total_throughput = 0
            for engine_id in range(self. num_instances):
                if lora_type in self.engine_model_mapping[engine_id]:
                    total_throughput += self.available_gpu_memorys[engine_id]
            
            if expected_lora_distribution[lora_type] > 0:
                min_bt = min(min_bt, total_throughput / expected_lora_distribution[lora_type])
        
        return min_bt
    

    async def _apply_lora_placement(self):
        """Apply LoRA placement to all instances"""
        session = await self._get_session()
        
        tasks = []
        for engine_id in range(self.num_instances):
            url = f"{self.instance_urls[engine_id]}/adjust_lora_adapter"
            active_models = self.engine_model_mapping[engine_id]
            tasks.append(self._adjust_lora_adapter(session, url, active_models))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for engine_id, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to apply LoRA placement to instance {engine_id}: {result}")
            else:
                logger.info(f"Instance {engine_id} LoRA placement applied: {result}")

    
    async def _adjust_lora_adapter(self, session: aiohttp.ClientSession, url: str, active_models: List[int]):
        """Adjust LoRA adapters on a single instance"""
        try:
            async with session.post(url, json={"active": active_models}) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to adjust LoRA adapters: status {resp.status}")
                    return None
                return await resp.json()
        except Exception as e:
            logger. error(f"Error adjusting LoRA adapters: {e}")
            return None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300, connect=10, sock_read=300)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _fetch_engine_stats(
        self, 
        session: aiohttp. ClientSession, 
        engine_id: int, 
        url: str
    ) -> Optional[EngineStats]:
        """Fetch statistics from a single engine instance"""
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger. warning(f"Engine {engine_id} returned status {resp.status}")
                    return None
                
                data = await resp.json()
                return EngineStats.from_response(data, engine_id)
        except Exception as e:
            logger.error(f"Error fetching stats from engine {engine_id}: {e}")
            return None

    async def select_engine(self, request_id: str, model_id: int) -> int:
        """
        Select engine for a new request (adapted from dLoRA's select_engine). 
        
        Returns:
            engine_id: Selected engine ID
        """
        async with self.select_lock:
            self.expected_lora_distribution[model_id] += 1
            await self._update_engine_costs()
            
            # If model has no assigned engines, assign to least loaded
            if len(self.model_engine_mapping[model_id]) == 0:
                engine_id = min(
                    range(self. num_instances), 
                    key=lambda eid: self.engine_request_count[eid]
                )
                self.model_engine_mapping[model_id] = [engine_id]
                self.engine_model_mapping[engine_id].append(model_id)
                logger.info(f"Added model {model_id} to engine {engine_id}")
            
            # Candidate engines
            candidate_engines = {}
            sub_engine_model_ids = {}
            
            for engine_id in self.model_engine_mapping[model_id]:
                candidate_engines[engine_id] = self.engine_exec_cost[engine_id]
            
            # Dispatch migration: try substituting unused LoRAs
            if self.migration_type == MigrationType.DISPATCH_MIG:
                for engine_id, req_model_cnt in self.engine_req_model_cnt.items():
                    if engine_id in candidate_engines:
                        continue
                    for sub_model_id in self.engine_model_mapping[engine_id]:
                        if req_model_cnt. get(sub_model_id, 0) == 0 and len(self.model_engine_mapping[sub_model_id]) > 1:
                            candidate_engines[engine_id] = self.engine_exec_cost[engine_id] + self.lora_load_cost
                            sub_engine_model_ids[engine_id] = sub_model_id
                            break
            
            # Select engine with minimum cost
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
            
            return min_engine_id

    async def _apply_lora_placement_single(self, engine_id: int):
        """Apply LoRA placement to a single instance"""
        session = await self._get_session()
        url = f"{self.instance_urls[engine_id]}/adjust_lora_adapter"
        active_models = self.engine_model_mapping[engine_id]
        await self._adjust_lora_adapter(session, url, active_models)

    async def _update_engine_costs(self):
        """Update execution costs for all engines"""
        for engine_id in range(self. num_instances):
            if engine_id not in self.engine_stats:
                continue
            
            stats = self.engine_stats[engine_id]
            req_model_cnt = stats.req_model_cnt
            
            cost = 0.0
            res_cnt = 0
            for _, cnt in req_model_cnt.items():
                res_cnt += cnt % self.max_running_requests
                cost += (cnt // self.max_running_requests) * self.merge_speed_ratio
            cost += res_cnt // self.max_running_requests
            
            self.engine_exec_cost[engine_id] = cost
            self.engine_req_model_cnt[engine_id] = req_model_cnt

    async def start_background_loop(self):
        """Start periodic migration loop"""
        if self._running:
            logger.warning("Background loop already running")
            return
        
        self._running = True
        self.background_loop = asyncio. create_task(self._migration_loop())
        logger.info("Background migration loop started")

    async def _migration_loop(self):
        """Periodic migration loop"""
        while self._running:
            await asyncio.sleep(self.migration_interval)
            
            try:
                await self._perform_migration()
            except Exception as e:
                logger.error(f"Migration failed: {e}", exc_info=True)

    async def _perform_migration(self):
        """Perform migration decision and execution (adapted from dLoRA)"""
        async with self.migration_lock:
            logger.info("Starting migration cycle...")
            
            # Fetch current stats
            await self._fetch_all_engine_stats()
            
            # Check if migration is needed
            need_migration = self._check_migration_needed()
            if not need_migration:
                logger. info("No migration needed")
                return
            
            # Run ILP solver
            migration_plan = await self._solve_migration_ilp()
            if migration_plan is None:
                logger.warning("Migration ILP failed")
                return
            
            # Execute migration
            await self._execute_migration(migration_plan)
            
            logger.info("Migration cycle complete")

    async def _fetch_all_engine_stats(self):
        """Fetch stats from all engines"""
        session = await self._get_session()
        
        tasks = []
        for engine_id in range(self.num_instances):
            url = f"{self.instance_urls[engine_id]}/get_engine_stats"
            tasks.append(self._fetch_engine_stats(session, engine_id, url))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.reqs_metadata = {}
        self.model_exec_info = {i: [0, 0.0] for i in range(self.num_models)}
        
        for engine_id, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch stats from engine {engine_id}: {result}")
                continue
            
            self.engine_stats[engine_id] = result
            self.reqs_metadata[engine_id] = result.req_metadata
            
            # Update model exec info
            for model_id, exec_info in result.model_exec_time.items():
                self.model_exec_info[model_id][0] += exec_info[0]
                self.model_exec_info[model_id][1] += exec_info[1]
        
        # Calculate average exec times
        for model_id, exec_info in self.model_exec_info.items():
            if exec_info[0] > 0:
                self.model_avg_exec_time[model_id] = exec_info[1] / exec_info[0]
            else:
                self.model_avg_exec_time[model_id] = self.default_exec_time

    def _check_migration_needed(self) -> bool:
        """Check if migration is needed"""
        # Check KV cache pressure
        engine_block_cnt = {}
        for engine_id, reqs in self.reqs_metadata.items():
            engine_block_cnt[engine_id] = sum(req.num_blocks for req in reqs)
        
        sorted_blocks = sorted(engine_block_cnt.items(), key=lambda x: x[1])
        if sorted_blocks:
            most_block_engine_id, most_block_cnt = sorted_blocks[-1]
            least_block_engine_id, least_block_cnt = sorted_blocks[0]
            
            if (most_block_cnt >= self.num_gpu_pages[most_block_engine_id] * 0.9 and
                least_block_cnt < self.num_gpu_pages[least_block_engine_id] * 0.9):
                return True
        
        # Check request imbalance
        engine_req_cnt = {i: len(reqs) for i, reqs in self.reqs_metadata.items()}
        num_reqs = sum(engine_req_cnt.values())
        if num_reqs == 0:
            return False
        
        avg_num_reqs = num_reqs / len(engine_req_cnt)
        max_req_cnt = max(engine_req_cnt.values())
        
        return (max_req_cnt - avg_num_reqs) >= self.migration_req_thres

    async def _solve_migration_ilp(self):
        """Solve migration ILP (placeholder - needs MigrationILP implementation)"""
        # TODO: Implement MigrationILP solver
        logger.warning("ILP solver not yet implemented, skipping migration")
        return None

    async def _execute_migration(self, migration_plan):
        """Execute migration plan (placeholder)"""
        # TODO: Implement migration execution
        pass

    async def close(self):
        """Close manager and cleanup"""
        self._running = False
        if self.background_loop:
            self.background_loop.cancel()
            try:
                await self.background_loop
            except asyncio.CancelledError:
                pass
        
        if self._session and not self._session.closed:
            await self._session.close()
        
        logger.info("InstanceManager closed")
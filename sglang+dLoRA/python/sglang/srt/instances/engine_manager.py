# File: sglang+dLoRA/python/sglang/srt/instances/instance_manager.py

import asyncio
import time
import aiohttp
import logging
import random
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

from sglang.srt. instances.migration_ilp import MigrationILP

logger = logging.getLogger(__name__)

_GB = 1 << 30
PCIE_BANDWIDTH = 32 * _GB


class MigrationType(Enum):
    DISPATCH_ONLY = 1
    DISPATCH_MIG = 2
    PERIOD_MIG = 3


@dataclass
class RequestMetadata:
    """Metadata for a single request"""
    request_id:  str
    model_id: Optional[str]  # lora_id or None for base model
    engine_id: int
    num_blocks: int
    in_gpu: bool
    prompt_length: int = 0
    output_length: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict, engine_id: int) -> "RequestMetadata":
        return cls(
            request_id=data["request_id"],
            model_id=data.get("model_id"),
            engine_id=engine_id,
            num_blocks=data["num_blocks"],
            in_gpu=data["in_gpu"],
            prompt_length=data. get("prompt_length", 0),
            output_length=data.get("output_length", 0),
        )


@dataclass
class EngineStats:
    """Statistics for a single SGLang instance"""
    engine_id: int
    num_requests: int
    req_model_cnt: Dict[Optional[str], int]
    num_free_gpu_blocks: int
    num_free_cpu_blocks: int
    lora_capacity: int
    active_models: List[str]
    req_metadata: List[RequestMetadata]
    model_exec_time: Dict[Optional[str], List[float]]
    available_gpu_memory: float
    cache_block_size: int
    lora_weight_size: int
    
    @classmethod
    def from_response(cls, data: Dict, engine_id: int) -> "EngineStats":
        req_metadata = [
            RequestMetadata.from_dict(req_data, engine_id)
            for req_data in data. get("req_metadata", [])
        ]
        
        return cls(
            engine_id=engine_id,
            num_requests=data["num_requests"],
            req_model_cnt=data.get("req_model_cnt", {}),
            num_free_gpu_blocks=data["num_free_gpu_blocks"],
            num_free_cpu_blocks=data. get("num_free_cpu_blocks", 0),
            lora_capacity=data. get("lora_capacity", 8),
            active_models=data.get("active_models", []),
            req_metadata=req_metadata,
            model_exec_time=data.get("model_exec_time", {}),
            available_gpu_memory=data.get("available_gpu_memory", 10.0 * _GB),
            cache_block_size=data.get("cache_block_size", 4096),
            lora_weight_size=data.get("lora_weight_size", 200 * 1024 * 1024),
        )


class InstanceManager:
    """
    Manages multiple SGLang instances with dLoRA-style load balancing.
    Adapted from dLoRA's EngineManager to work with SGLang. 
    """
    
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
    ):
        # Basic configuration
        self.num_instances = num_instances
        self. num_models = num_models
        self.instance_urls = instance_urls
        
        # Execution parameters (from dLoRA)
        self.migration_type = migration_type
        self.migration_interval = migration_interval
        self.lora_capacity_per_engine = lora_capacity_per_engine
        self.max_running_requests = max_running_requests
        self.default_exec_time = default_exec_time
        self. migration_req_thres = migration_req_thres
        self.merge_speed_ratio = merge_speed_ratio
        
        # Request tracking
        self.request_to_engine:  Dict[str, int] = {}
        self.engine_request_count: Dict[int, int] = {i: 0 for i in range(num_instances)}
        
        # Model distribution (model_id:  0, 1, 2, ...)
        self.model_engine_mapping: Dict[int, List[int]] = {i: [] for i in range(num_models)}
        self.engine_model_mapping: Dict[int, List[int]] = {i: [] for i in range(num_instances)}
        self.expected_lora_distribution: List[float] = [(1.0 + 1e-7) / num_models] * num_models
        
        # Statistics
        self.engine_stats: Dict[int, EngineStats] = {}
        self.global_model_request_count: Dict[int, int] = {i: 0 for i in range(num_models)}
        self.engine_exec_cost: Dict[int, float] = {}
        self.engine_req_model_cnt: Dict[int, Dict[Optional[str], int]] = {}
        self.model_exec_info: Dict[int, List[float]] = {i: [0, 0.0] for i in range(num_models)}
        self.model_avg_exec_time: List[float] = [default_exec_time] * num_models
        self.reqs_metadata: Dict[int, List[RequestMetadata]] = {}
        
        # LoRA cost estimation
        self.lora_load_cost = (200 * 1024 * 1024) / PCIE_BANDWIDTH
        
        # Async locks
        self.select_lock = asyncio.Lock()
        self.migration_lock = asyncio.Lock()
        
        # Background tasks
        self.background_loop = None
        self._running = False
        
        # HTTP session
        self._session:  Optional[aiohttp.ClientSession] = None
        
        # GPU memory and blocks (fetched from instances)
        self.available_gpu_memorys: List[float] = [0.0] * num_instances
        self.num_gpu_blocks: List[int] = [0] * num_instances
        self.num_cpu_blocks: List[int] = [0] * num_instances
        self.cache_block_size: int = 0
        self.lora_weight_sizes: List[int] = [0] * num_instances
        
        # Engine capacity
        self.engine_lora_capacity: List[int] = [lora_capacity_per_engine] * num_instances
        
        logger.info(f"InstanceManager initialized:  {num_instances} instances, {num_models} models")

    async def initialize(self):
        """Async initialization:  fetch stats and place LoRAs"""
        logger.info("Fetching initial stats from all instances...")
        await self._fetch_instance_stats()
        
        logger.info("Initializing LoRA placement...")
        self._initialize_lora_placement()
        
        logger.info("Applying initial LoRA placement to instances...")
        await self._apply_lora_placement()
        
        logger.info("InstanceManager initialization complete")

    async def _fetch_instance_stats(self):
        """Fetch initial statistics from all instances (called during initialize)"""
        session = await self._get_session()
        
        tasks = []
        for engine_id in range(self.num_instances):
            url = f"{self.instance_urls[engine_id]}/get_engine_stats"
            tasks.append(self._fetch_engine_stats(session, engine_id, url))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for engine_id, result in enumerate(results):
            if isinstance(result, Exception):
                logger. error(f"Failed to fetch stats from instance {engine_id}: {result}")
                # Use default fallback values
                self.num_gpu_blocks[engine_id] = 1000
                self.num_cpu_blocks[engine_id] = 0
                self.available_gpu_memorys[engine_id] = 10.0 * _GB
                self.lora_weight_sizes[engine_id] = 200 * 1024 * 1024
                self.cache_block_size = 4096
            else:
                self.engine_stats[engine_id] = result
                self.num_gpu_blocks[engine_id] = result. num_free_gpu_blocks
                self. num_cpu_blocks[engine_id] = result.num_free_cpu_blocks
                self. available_gpu_memorys[engine_id] = result.available_gpu_memory
                self.lora_weight_sizes[engine_id] = result.lora_weight_size
                if result.cache_block_size > 0:
                    self. cache_block_size = result. cache_block_size
        
        logger.info(f"Fetched stats:  GPU blocks={self.num_gpu_blocks}, cache_block_size={self.cache_block_size}")

    def _initialize_lora_placement(self):
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
                self.available_gpu_memorys[next_engine_id] -= self.lora_weight_sizes[next_engine_id]
                next_engine_id = (next_engine_id + 1) % self.num_instances
                current_lora_distribution[next_lora_type] += 1
                num_lora_replicas += 1
        
        # Build reverse mapping
        self.model_engine_mapping = {i: [] for i in range(self.num_models)}
        for engine_id, model_ids in self.engine_model_mapping.items():
            for model_id in model_ids: 
                self.model_engine_mapping[model_id].append(engine_id)
        
        logger.info(f"LoRA placement:  {num_lora_replicas} replicas, mapping={self.engine_model_mapping}")

    def find_best_lora_weight_schedule(
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

    async def _apply_lora_placement(self):
        """Apply LoRA placement to all instances via HTTP"""
        session = await self._get_session()
        
        tasks = []
        for engine_id in range(self.num_instances):
            url = f"{self.instance_urls[engine_id]}/adjust_lora_adapter"
            active_model_ids = self.engine_model_mapping[engine_id]
            tasks.append(self._adjust_lora_adapter(session, url, active_model_ids))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for engine_id, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to apply LoRA placement to instance {engine_id}: {result}")
            else:
                logger.info(f"Instance {engine_id} LoRA placement applied:  {result}")

    async def _adjust_lora_adapter(self, session:  aiohttp.ClientSession, url: str, active_model_ids: List[int]):
        """Adjust LoRA adapters on a single instance"""
        try: 
            async with session.post(url, json={"active":  active_model_ids}) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to adjust LoRA:  status {resp.status}")
                    return None
                return await resp.json()
        except Exception as e:
            logger. error(f"Error adjusting LoRA:  {e}")
            return None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300, connect=10, sock_read=300)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _fetch_engine_stats(
        self, 
        session: aiohttp.ClientSession, 
        engine_id: int, 
        url: str
    ) -> Optional[EngineStats]:
        """Fetch statistics from a single instance"""
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

    def get_migration_info(self):
        """
        Get migration information (from dLoRA's get_migration_info).
        Called by external systems to monitor migration state.
        """
        return {
            "model_engine_mapping": self.model_engine_mapping,
            "engine_model_mapping": self.engine_model_mapping,
            "expected_lora_distribution": self.expected_lora_distribution,
            "engine_request_count": self.engine_request_count,
            "num_gpu_blocks": self.num_gpu_blocks,
            "num_cpu_blocks": self. num_cpu_blocks,
            "lora_capacity": self.engine_lora_capacity,
        }

    async def select_engine(self, request_id: str, model_id:  int) -> int:
        """
        Select engine for a new request (dLoRA's select_engine).
        
        Returns:
            engine_id: Selected engine ID
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
                self.engine_model_mapping[engine_id].append(model_id)
                logger.info(f"Added model {model_id} to engine {engine_id}")
            
            # Candidate engines
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
                            candidate_engines[engine_id] = self.engine_exec_cost. get(engine_id, 0) + self.lora_load_cost
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
        active_model_ids = self.engine_model_mapping[engine_id]
        await self._adjust_lora_adapter(session, url, active_model_ids)

    async def _update_engine_costs(self):
        """Update execution costs for all engines (dLoRA's formula)"""
        # Fetch latest stats
        session = await self._get_session()
        tasks = []
        for engine_id in range(self.num_instances):
            url = f"{self.instance_urls[engine_id]}/get_engine_stats"
            tasks.append(self._fetch_engine_stats(session, engine_id, url))
        
        results = await asyncio. gather(*tasks, return_exceptions=True)
        
        for engine_id, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            
            self.engine_stats[engine_id] = result
            req_model_cnt = result.req_model_cnt
            
            cost = 0.0
            res_cnt = 0
            for _, cnt in req_model_cnt.items():
                res_cnt += cnt % self.max_running_requests
                cost += (cnt // self.max_running_requests) * self.merge_speed_ratio
            cost += res_cnt // self.max_running_requests
            
            self.engine_exec_cost[engine_id] = cost
            self.engine_req_model_cnt[engine_id] = req_model_cnt

    def is_running(self) -> bool:
        """Check if background migration loop is running"""
        return self._running and self.background_loop is not None

    def start_background_loop(self) -> None:
        """Start periodic migration loop (from dLoRA)"""
        if self._running:
            logger.warning("Background loop already running")
            return
        
        self._running = True
        # Note: async task needs to be created in an async context
        logger.info("Background migration loop will be started")

    async def run_loop(self):
        """Main migration loop (from dLoRA's run_loop)"""
        self._running = True
        while self._running:
            await asyncio.sleep(self.migration_interval)
            
            try:
                await self._perform_migration()
            except Exception as e:
                logger.error(f"Migration failed: {e}", exc_info=True)

    async def _perform_migration(self):
        """Perform migration decision and execution (dLoRA's migration_schedule)"""
        async with self.migration_lock:
            logger.info("Starting migration cycle...")
            
            # Fetch current stats from all engines
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
        """Fetch stats from all engines (called by migration loop)"""
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
                logger.error(f"Failed to fetch stats from engine {engine_id}:  {result}")
                continue
            
            self.engine_stats[engine_id] = result
            self.reqs_metadata[engine_id] = result.req_metadata
            
            # Update model exec info
            for model_id, exec_info in result.model_exec_time.items():
                if model_id is not None:
                    # Convert lora_id to model_id if needed
                    # Assuming model_id is already int
                    self.model_exec_info[model_id][0] += exec_info[0]
                    self.model_exec_info[model_id][1] += exec_info[1]
        
        # Calculate average exec times
        for model_id, exec_info in self.model_exec_info.items():
            if exec_info[0] > 0:
                self.model_avg_exec_time[model_id] = exec_info[1] / exec_info[0]
            else:
                self.model_avg_exec_time[model_id] = self.default_exec_time

    def _check_migration_needed(self) -> bool:
        """Check if migration is needed (dLoRA's logic)"""
        # Check KV cache pressure
        engine_block_cnt = {}
        for engine_id, reqs in self.reqs_metadata.items():
            engine_block_cnt[engine_id] = sum(req.num_blocks for req in reqs)
        
        sorted_blocks = sorted(engine_block_cnt.items(), key=lambda x: x[1])
        if sorted_blocks: 
            most_block_engine_id, most_block_cnt = sorted_blocks[-1]
            least_block_engine_id, least_block_cnt = sorted_blocks[0]
            
            if (most_block_cnt >= self.num_gpu_blocks[most_block_engine_id] * 0.9 and
                least_block_cnt < self.num_gpu_blocks[least_block_engine_id] * 0.9):
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
        """Solve migration ILP (from dLoRA's migration_ilp. py)"""
        try:
            # Flatten all request metadata
            all_reqs_metadata = []
            for engine_id, reqs in self.reqs_metadata.items():
                all_reqs_metadata.extend(reqs)
            
            if len(all_reqs_metadata) == 0:
                return None
            
            # Create ILP solver
            ilp = MigrationILP(
                reqs_metadata=all_reqs_metadata,
                num_groups=self.num_instances,
                num_models=self.num_models,
                engine_gpu_blocks=self.num_gpu_blocks,
                engine_cpu_blocks=self.num_cpu_blocks,
                engine_lora_capacity=self. engine_lora_capacity,
                lora_exec_time=self.model_avg_exec_time,
                alpha=0.05,  # Default from dLoRA
                bw=PCIE_BANDWIDTH,
                model_engine_mapping=self.model_engine_mapping,
            )
            
            # Solve
            req_migration_mapping, lora_weight_mapping, lora_weight_cnt = ilp.solve()
            
            return {
                "req_migration":  req_migration_mapping,
                "lora_weights": lora_weight_mapping,
                "lora_counts": lora_weight_cnt,
            }
        except Exception as e:
            logger.error(f"ILP solver failed: {e}", exc_info=True)
            return None

    async def _execute_migration(self, migration_plan):
        """Execute migration plan (from dLoRA)"""
        req_migration = migration_plan["req_migration"]
        lora_weights = migration_plan["lora_weights"]
        
        logger.info(f"Executing migration plan: {migration_plan}")
        
        # Step 1: Adjust LoRA adapters
        await self._execute_lora_adjustment(lora_weights)
        
        # Step 2: Migrate requests
        await self._execute_request_migration(req_migration)
        
        # Step 3: Update internal state
        self.engine_model_mapping = lora_weights

    async def _execute_lora_adjustment(self, lora_weights:  Dict[int, List[int]]):
        """Adjust LoRA adapters based on migration plan"""
        session = await self._get_session()
        
        tasks = []
        for engine_id, model_ids in lora_weights. items():
            url = f"{self.instance_urls[engine_id]}/adjust_lora_adapter"
            tasks.append(self._adjust_lora_adapter(session, url, model_ids))
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_request_migration(self, req_migration: Dict[int, Dict[int, List[str]]]):
        """Migrate requests between engines"""
        session = await self._get_session()
        
        # Fetch all requests to migrate
        fetch_tasks = []
        for src_engine_id, dst_mapping in req_migration.items():
            for dst_engine_id, request_ids in dst_mapping.items():
                if len(request_ids) == 0:
                    continue
                url = f"{self.instance_urls[src_engine_id]}/fetch_seq_groups"
                fetch_tasks.append(self._fetch_seq_groups(session, url, request_ids))
        
        seq_groups_list = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Insert to destination engines
        insert_tasks = []
        idx = 0
        for src_engine_id, dst_mapping in req_migration.items():
            for dst_engine_id, request_ids in dst_mapping. items():
                if len(request_ids) == 0:
                    continue
                if idx < len(seq_groups_list) and not isinstance(seq_groups_list[idx], Exception):
                    url = f"{self.instance_urls[dst_engine_id]}/insert_seq_groups"
                    insert_tasks.append(self._insert_seq_groups(session, url, seq_groups_list[idx]))
                idx += 1
        
        await asyncio.gather(*insert_tasks, return_exceptions=True)
        
        # Abort from source engines
        abort_tasks = []
        for src_engine_id, dst_mapping in req_migration.items():
            all_request_ids = []
            for request_ids in dst_mapping.values():
                all_request_ids.extend(request_ids)
            if len(all_request_ids) > 0:
                url = f"{self.instance_urls[src_engine_id]}/abort_requests"
                abort_tasks.append(self._abort_requests(session, url, all_request_ids))
        
        await asyncio.gather(*abort_tasks, return_exceptions=True)

    async def _fetch_seq_groups(self, session:  aiohttp.ClientSession, url: str, request_ids: List[str]):
        """Fetch sequence groups from an engine"""
        try:
            async with session.post(url, json={"request_ids": request_ids}) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data. get("seq_groups", [])
        except Exception as e: 
            logger.error(f"Error fetching seq groups:  {e}")
            return None

    async def _insert_seq_groups(self, session: aiohttp.ClientSession, url: str, seq_groups: List[Dict]):
        """Insert sequence groups to an engine"""
        try:
            async with session.post(url, json={"seq_groups": seq_groups}) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception as e:
            logger. error(f"Error inserting seq groups: {e}")
            return None

    async def _abort_requests(self, session: aiohttp.ClientSession, url: str, request_ids: List[str]):
        """Abort requests on an engine"""
        try: 
            async with session.post(url, json={"request_ids":  request_ids}) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception as e:
            logger.error(f"Error aborting requests: {e}")
            return None

    async def close(self):
        """Close manager and cleanup resources"""
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
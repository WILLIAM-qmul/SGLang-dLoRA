# File: sglang+dLoRA/python/sglang/srt/managers/engine_manager.py
"""
Engine Manager for managing multiple SGLang instances with dynamic load balancing. 
Adapted from dLoRA-artifact's EngineManager to SGLang architecture.
"""

import asyncio
import time
import aiohttp
import json
import multiprocessing
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass
import logging

from sglang.srt. instances.migration_ilp import MigrationILP

logger = logging.getLogger(__name__)

_GB = 1 << 30
PCIE_BANDWIDTH = 32 * _GB  # 32 GB/s


class MigrationType(Enum):
    """Migration strategy types"""
    DISPATCH_ONLY = 1      # Only dispatch, no migration
    DISPATCH_MIG = 2       # Dispatch with migration
    PERIOD_MIG = 3         # Periodic migration


@dataclass
class RequestMetadata:
    """Metadata for a single request in the system"""
    request_id: str
    model_id: int
    engine_id: int
    num_blocks: int
    in_gpu: bool
    prompt_length: int = 0
    output_length: int = 0
    
    def __repr__(self) -> str:
        return (f"ReqMetadata(request_id={self.request_id}, "
                f"model_id={self.model_id}, "
                f"engine_id={self. engine_id}, "
                f"num_blocks={self.num_blocks})")


@dataclass
class EngineStats:
    """Statistics for a single engine instance"""
    engine_id: int
    num_requests: int
    req_model_cnt: Dict[int, int]
    num_free_gpu_blocks: int
    num_free_cpu_blocks: int
    lora_capacity: int
    active_models: List[int]
    req_metadata: List[RequestMetadata]
    model_exec_time: Dict[int, Tuple[int, float]]


class EngineManager:
    """
    Manages multiple SGLang engine instances with dynamic load balancing. 
    
    This manager handles:
    - Request routing to engines
    - Dynamic load balancing
    - Request migration between engines
    - LoRA adapter management across engines
    """

    def __init__(
        self,
        num_instances: int,
        num_models: int,
        instance_urls: List[str],
        migration_type: MigrationType = MigrationType.PERIOD_MIG,
        migration_interval: float = 5.0,
        lora_capacity_per_engine: int = 8,
        max_running_requests: int = 16,
    ):
        """
        Initialize the Engine Manager.
        
        Args:
            num_instances: Number of SGLang instances to manage
            num_models: Total number of LoRA models
            instance_urls: List of instance URLs (e.g., ["http://127.0.0.1:30001", ... ])
            migration_type: Migration strategy
            migration_interval: Interval for periodic migration (seconds)
            lora_capacity_per_engine: Max LoRA adapters per engine
            max_running_requests: Max concurrent requests per engine
        """
        self.num_instances = num_instances
        self.num_models = num_models
        self.instance_urls = instance_urls
        self. migration_type = migration_type
        self.migration_interval = migration_interval
        self.lora_capacity_per_engine = lora_capacity_per_engine
        self.max_running_requests = max_running_requests
        
        # Request tracking
        self.request_to_engine: Dict[str, int] = {}  # request_id -> engine_id
        self.engine_request_count: Dict[int, int] = {i: 0 for i in range(num_instances)}
        
        # Model distribution tracking
        self.engine_active_models: Dict[int, Set[int]] = {i: set() for i in range(num_instances)}
        
        # Statistics
        self.engine_stats: Dict[int, EngineStats] = {}
        self.global_model_request_count: Dict[int, int] = {i: 0 for i in range(num_models)}
        
        # Async locks
        self.select_lock = asyncio.Lock()
        self.migration_lock = asyncio.Lock()
        
        # Background tasks
        self.background_loop = None
        self._running = False
        
        # Aiohttp session for API calls
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"EngineManager initialized with {num_instances} instances, "
                   f"{num_models} models, migration_type={migration_type. name}")

    async def _get_session(self) -> aiohttp. ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300, connect=10, sock_read=300)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the manager and cleanup resources"""
        self._running = False
        if self. background_loop:
            self.background_loop.cancel()
            try:
                await self.background_loop
            except asyncio.CancelledError:
                pass
        
        if self._session and not self._session.closed:
            await self._session.close()

    def is_running(self) -> bool:
        """Check if background migration loop is running"""
        return self._running and self.background_loop is not None

    def start_background_loop(self) -> None:
        """Start the background migration loop"""
        if self.is_running():
            logger.warning("Background loop already running")
            return
        
        self._running = True
        self.background_loop = asyncio.create_task(self. run_loop())
        logger.info("Background migration loop started")

    async def run_loop(self):
        """Background loop for periodic migration"""
        logger.info(f"Migration loop started with interval={self.migration_interval}s")
        
        while self._running:
            try:
                if self.migration_type == MigrationType.PERIOD_MIG:
                    await asyncio.sleep(self.migration_interval)
                    await self._periodic_migration()
                else:
                    await asyncio.sleep(1.0)
            except asyncio. CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in migration loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)
        
        logger.info("Migration loop stopped")

    async def _periodic_migration(self):
        """Execute periodic migration based on current load"""
        async with self.migration_lock:
            try:
                # Get migration plan
                migration_plan, need_migration = await self.migration_schedule()
                
                if not need_migration:
                    logger. debug("No migration needed")
                    return
                
                logger.info(f"Executing migration plan: {migration_plan}")
                
                # Execute migration
                await self._execute_migration(migration_plan)
                
            except Exception as e:
                logger.error(f"Error in periodic migration: {e}", exc_info=True)

    async def get_migration_info(self) -> Tuple[Dict[int, EngineStats], bool]:
        """
        Collect migration info from all engines.
        
        Returns:
            Tuple of (engine_stats_dict, has_data)
        """
        engine_stats = {}
        session = await self._get_session()
        
        tasks = []
        for engine_id in range(self.num_instances):
            url = f"{self.instance_urls[engine_id]}/get_engine_stats"
            tasks.append(self._fetch_engine_stats(session, engine_id, url))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        has_data = False
        for engine_id, result in enumerate(results):
            if isinstance(result, Exception):
                logger. warning(f"Failed to get stats from engine {engine_id}: {result}")
                continue
            
            if result:
                engine_stats[engine_id] = result
                has_data = True
        
        return engine_stats, has_data

    async def _fetch_engine_stats(
        self, 
        session: aiohttp. ClientSession, 
        engine_id: int, 
        url: str
    ) -> Optional[EngineStats]:
        """Fetch statistics from a single engine"""
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger. warning(f"Engine {engine_id} returned status {resp.status}")
                    return None
                
                data = await resp.json()
                
                # Parse request metadata
                req_metadata = []
                for req_data in data.get("req_metadata", []):
                    req_metadata.append(RequestMetadata(
                        request_id=req_data["request_id"],
                        model_id=req_data["model_id"],
                        engine_id=engine_id,
                        num_blocks=req_data["num_blocks"],
                        in_gpu=req_data["in_gpu"],
                        prompt_length=req_data.get("prompt_length", 0),
                        output_length=req_data.get("output_length", 0),
                    ))
                
                return EngineStats(
                    engine_id=engine_id,
                    num_requests=data["num_requests"],
                    req_model_cnt=data["req_model_cnt"],
                    num_free_gpu_blocks=data["num_free_gpu_blocks"],
                    num_free_cpu_blocks=data["num_free_cpu_blocks"],
                    lora_capacity=data["lora_capacity"],
                    active_models=data["active_models"],
                    req_metadata=req_metadata,
                    model_exec_time=data.get("model_exec_time", {}),
                )
        except Exception as e:
            logger.error(f"Error fetching stats from engine {engine_id}: {e}")
            return None

    async def migration_schedule(self) -> Tuple[Dict, bool]:
        """
        Compute migration schedule using ILP solver.
        
        Returns:
            Tuple of (migration_plan, need_migration)
            migration_plan = {
                "req_migration": {src_engine: {dst_engine: [req_ids]}},
                "lora_placement": {engine_id: [model_ids]}
            }
        """
        # Collect current state
        engine_stats, has_data = await self.get_migration_info()
        
        if not has_data or len(engine_stats) < 2:
            return {}, False
        
        # Aggregate all requests
        all_reqs_metadata = []
        for stats in engine_stats.values():
            all_reqs_metadata. extend(stats.req_metadata)
        
        if len(all_reqs_metadata) == 0:
            logger.debug("No requests to migrate")
            return {}, False
        
        # Prepare ILP inputs
        engine_gpu_blocks = [
            engine_stats[i].num_free_gpu_blocks 
            for i in range(self.num_instances)
        ]
        engine_cpu_blocks = [
            engine_stats[i].num_free_cpu_blocks 
            for i in range(self.num_instances)
        ]
        engine_lora_capacity = [
            self.lora_capacity_per_engine 
            for _ in range(self.num_instances)
        ]
        
        # Compute average execution time per model
        lora_exec_time = [0.0] * self.num_models
        for stats in engine_stats.values():
            for model_id, (count, total_time) in stats.model_exec_time.items():
                if count > 0:
                    avg_time = total_time / count
                    lora_exec_time[model_id] = max(lora_exec_time[model_id], avg_time)
        
        # Current model-to-engine mapping
        model_engine_mapping = {i: [] for i in range(self.num_models)}
        for engine_id, stats in engine_stats.items():
            for model_id in stats.active_models:
                model_engine_mapping[model_id].append(engine_id)
        
        # Solve ILP
        try:
            ilp_solver = MigrationILP(
                reqs_metadata=all_reqs_metadata,
                num_groups=self.num_instances,
                num_models=self.num_models,
                engine_gpu_blocks=engine_gpu_blocks,
                engine_cpu_blocks=engine_cpu_blocks,
                engine_lora_capacity=engine_lora_capacity,
                lora_exec_time=lora_exec_time,
                alpha=0.5,
                bw=PCIE_BANDWIDTH,
                model_engine_mapping=model_engine_mapping,
            )
            
            req_migration_mapping, lora_weight_mapping, lora_weight_cnt = ilp_solver.solve()
            
            # Check if migration is needed
            need_migration = False
            for src in req_migration_mapping:
                for dst in req_migration_mapping[src]:
                    if len(req_migration_mapping[src][dst]) > 0:
                        need_migration = True
                        break
            
            if not need_migration:
                logger. info("ILP solver found no beneficial migration")
                return {}, False
            
            migration_plan = {
                "req_migration": req_migration_mapping,
                "lora_placement": lora_weight_mapping,
            }
            
            logger.info(f"Migration plan computed: "
                       f"{sum(len(v) for d in req_migration_mapping.values() for v in d.values())} requests to migrate")
            
            return migration_plan, True
            
        except Exception as e:
            logger.error(f"Error in ILP solver: {e}", exc_info=True)
            return {}, False

    async def _execute_migration(self, migration_plan: Dict):
        """
        Execute the migration plan.
        
        Steps:
        1. Update LoRA adapters on engines
        2. Migrate requests between engines
        3.  Abort old requests on source engines
        """
        req_migration = migration_plan. get("req_migration", {})
        lora_placement = migration_plan.get("lora_placement", {})
        
        session = await self._get_session()
        
        # Step 1: Update LoRA adapters
        logger.info("Step 1: Updating LoRA adapters")
        adapter_tasks = []
        for engine_id, model_ids in lora_placement. items():
            url = f"{self.instance_urls[engine_id]}/adjust_lora_adapter"
            payload = {"active": model_ids}
            adapter_tasks.append(self._post_request(session, url, payload))
        
        await asyncio.gather(*adapter_tasks, return_exceptions=True)
        
        # Step 2: Migrate requests
        logger.info("Step 2: Migrating requests")
        
        # Collect all request IDs to migrate
        requests_to_migrate = {}  # src_engine -> [req_ids]
        migration_targets = {}    # req_id -> dst_engine
        
        for src_engine in req_migration:
            for dst_engine in req_migration[src_engine]:
                req_ids = req_migration[src_engine][dst_engine]
                if len(req_ids) > 0:
                    if src_engine not in requests_to_migrate:
                        requests_to_migrate[src_engine] = []
                    requests_to_migrate[src_engine].extend(req_ids)
                    
                    for req_id in req_ids:
                        migration_targets[req_id] = dst_engine
        
        if not requests_to_migrate:
            logger.info("No requests to migrate")
            return
        
        # Fetch sequence groups from source engines
        fetch_tasks = []
        for src_engine, req_ids in requests_to_migrate.items():
            url = f"{self.instance_urls[src_engine]}/fetch_seq_groups"
            payload = {"request_ids": req_ids}
            fetch_tasks.append(self._post_request(session, url, payload))
        
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Insert sequence groups to destination engines
        insert_tasks = []
        abort_tasks = []
        
        for idx, (src_engine, req_ids) in enumerate(requests_to_migrate. items()):
            result = fetch_results[idx]
            
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch from engine {src_engine}: {result}")
                continue
            
            # Group by destination engine
            dst_groups = {}
            for seq_group_data in result. get("seq_groups", []):
                req_id = seq_group_data["request_id"]
                dst_engine = migration_targets.get(req_id)
                
                if dst_engine is None:
                    continue
                
                if dst_engine not in dst_groups:
                    dst_groups[dst_engine] = []
                dst_groups[dst_engine].append(seq_group_data)
            
            # Insert to destination engines
            for dst_engine, seq_groups in dst_groups.items():
                url = f"{self.instance_urls[dst_engine]}/insert_seq_groups"
                payload = {"seq_groups": seq_groups}
                insert_tasks.append(self._post_request(session, url, payload))
            
            # Abort on source engine after successful migration
            abort_url = f"{self.instance_urls[src_engine]}/abort_requests"
            abort_payload = {"request_ids": req_ids}
            abort_tasks. append(self._post_request(session, abort_url, abort_payload))
        
        # Execute insertions and aborts
        await asyncio. gather(*insert_tasks, return_exceptions=True)
        await asyncio.gather(*abort_tasks, return_exceptions=True)
        
        # Update internal tracking
        for req_id, dst_engine in migration_targets. items():
            if req_id in self.request_to_engine:
                old_engine = self.request_to_engine[req_id]
                self.engine_request_count[old_engine] -= 1
            
            self.request_to_engine[req_id] = dst_engine
            self.engine_request_count[dst_engine] += 1
        
        logger.info(f"Migration completed: {len(migration_targets)} requests migrated")

    async def _post_request(self, session: aiohttp. ClientSession, url: str, payload: Dict):
        """Helper to make POST request"""
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"POST {url} returned status {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Error POST {url}: {e}")
            return None

    async def select_engine(
        self, 
        request_id: str, 
        model_id: int
    ) -> Tuple[int, str]:
        """
        Select an engine for a new request using load balancing.
        
        Strategy: Least-loaded engine with the target LoRA model already loaded.
        
        Returns:
            Tuple of (engine_id, instance_url)
        """
        async with self.select_lock:
            # Update global model request count
            self.global_model_request_count[model_id] = \
                self.global_model_request_count.get(model_id, 0) + 1
            
            # Get current engine stats
            engine_stats, has_data = await self.get_migration_info()
            
            if not has_data:
                # Fallback: round-robin
                engine_id = len(self.request_to_engine) % self.num_instances
                logger.warning(f"No stats available, using round-robin: engine {engine_id}")
            else:
                # Find least-loaded engine with model loaded
                candidates = []
                
                for eid, stats in engine_stats. items():
                    # Check if model is loaded
                    has_model = model_id in stats.active_models
                    
                    # Check capacity
                    has_capacity = stats.num_requests < self.max_running_requests
                    
                    if has_capacity:
                        # Prefer engines with model already loaded
                        priority = stats.num_requests if has_model else (stats.num_requests + 1000)
                        candidates.append((priority, eid))
                
                if candidates:
                    candidates.sort()
                    engine_id = candidates[0][1]
                else:
                    # All engines full, use least loaded
                    engine_id = min(
                        engine_stats.keys(), 
                        key=lambda eid: engine_stats[eid].num_requests
                    )
                
                logger.debug(f"Selected engine {engine_id} for model {model_id} "
                           f"(num_requests={engine_stats[engine_id].num_requests})")
            
            # Track request
            self.request_to_engine[request_id] = engine_id
            self.engine_request_count[engine_id] += 1
            self.engine_active_models[engine_id]. add(model_id)
            
            return engine_id, self.instance_urls[engine_id]

    async def complete_request(self, request_id: str):
        """Mark a request as completed"""
        if request_id in self.request_to_engine:
            engine_id = self.request_to_engine. pop(request_id)
            self.engine_request_count[engine_id] -= 1

    def get_stats(self) -> Dict:
        """Get current manager statistics"""
        return {
            "num_instances": self.num_instances,
            "num_models": self.num_models,
            "total_requests": len(self.request_to_engine),
            "engine_request_count": self.engine_request_count,
            "global_model_request_count": self.global_model_request_count,
            "migration_type": self.migration_type. name,
        }
"""
Engine Manager for dLoRA-style multi-instance LoRA serving.
Adapted from: https://github.com/LLMServe/dLoRA-artifact (vllm/engine/engine_manager.py)
"""

import asyncio
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from sglang.srt.instances.migration_ilp import MigrationILP

# Constants (same as dLoRA)
_GB = 1 << 30
PCIE_BANDWIDTH = 32 * _GB


class MigrationType(Enum):
    """Migration strategies for LoRA adapters."""
    DISPATCH_ONLY = 1      # Only dispatch, no migration
    DISPATCH_MIG = 2       # Dispatch + on-demand migration
    PERIOD_MIG = 3         # Periodic ILP-based migration


class ExecType(Enum):
    """Execution types for LoRA serving (kept for compatibility with dLoRA)."""
    LORA = 1               # LoRA-based execution
    REPLICATED = 2         # Replicated models across instances
    MERGED = 3             # Merged LoRA execution


@dataclass
class RequestMetadata:
    """Metadata for a request in the system (for ILP)."""
    request_id: str
    model_id: int
    engine_id: int
    num_blocks: int
    prompt_len: int
    output_len: int
    arrival_time: float
    exec_time: float = 0.0


class EngineManager:
    """
    Multi-instance Engine Manager for dLoRA-style architecture on SGLang.

    NOTE: 与 vLLM 版本不同，这里不直接管理 AsyncLLMEngine / KV Cache，
    而是管理一组 HTTP SGLang 实例的 URL，并在调度时返回应该访问哪个实例。
    """

    def __init__(
        self,
        exec_type: ExecType,
        migration_type: MigrationType,
        num_instances: int,
        num_loras: int,
        instance_urls: List[str],
        max_loras_per_batch: int = 8,
        max_running_requests: int = 16,
        migration_interval: int = 10,
        migration_req_threshold: int = 16,
        cpu_swap_space_gb: int = 4,
    ):
        """
        Initialize the EngineManager.

        Args:
            exec_type: Execution type (LORA, REPLICATED, MERGED)
            migration_type: Migration strategy
            num_instances: Number of SGLang instances
            num_loras: Total number of LoRA adapters
            instance_urls: HTTP base URLs of instances, e.g. ["http://127.0.0.1:30001", ...]
            max_loras_per_batch: Max LoRA adapters per batch (similar to bs in dLoRA)
            max_running_requests: Approximate capacity per instance (not strictly enforced here)
            migration_interval: Time between migration checks (seconds)
            migration_req_threshold: Request count threshold that triggers migration
            cpu_swap_space_gb: For compatibility with dLoRA (unused in SGLang HTTP version)
        """
        self.exec_type = exec_type
        self.migration_type = migration_type
        self.num_instances = num_instances
        self.num_loras = num_loras
        self.instance_urls = instance_urls
        self.max_loras_per_batch = max_loras_per_batch
        self.max_running_requests = max_running_requests
        self.migration_interval = migration_interval
        self.migration_req_threshold = migration_req_threshold
        self.cpu_swap_space = cpu_swap_space_gb * _GB

        # LoRA placement mappings (model_id <-> engine_id)
        # lora_to_instances[lora_id] = [instance_id...]
        # instance_to_loras[instance_id] = [lora_id...]
        self.lora_to_instances: Dict[int, List[int]] = {i: [] for i in range(num_loras)}
        self.instance_to_loras: Dict[int, List[int]] = {i: [] for i in range(num_instances)}

        # Instance metrics
        self.instance_num_requests: List[int] = [0] * num_instances
        self.instance_num_free_blocks: List[int] = [0] * num_instances
        # instance_req_lora_cnt[instance_id][lora_id] = num_requests
        self.instance_req_lora_cnt: Dict[int, Dict[int, int]] = {}
        # Approximated execution cost per instance
        self.instance_exec_cost: Dict[int, float] = {}

        # Request metadata tracking (for ILP)
        self.reqs_metadata: Dict[int, List[RequestMetadata]] = {}

        # LoRA execution statistics
        # lora_exec_info[lora_id] = (count, total_exec_time)
        self.lora_exec_info: Dict[int, Tuple[int, float]] = {
            i: (0, 0.0) for i in range(num_loras)
        }
        self.lora_avg_exec_time: List[float] = [5.0] * num_loras

        # Migration parameters (same semantics as dLoRA)
        self.default_exec_time = 5.0
        self.lora_load_cost = 0.1  # Estimated LoRA loading time
        self.merge_speed_ratio = 0.6

        # Expected LoRA distribution (maintained by commander)
        self.expected_lora_distribution = [
            (1.0 + 1e-7) / num_loras for _ in range(num_loras)
        ]

        # Capacity tracking (for ILP)
        self.engine_lora_capacity: List[int] = [0] * num_instances
        self.num_gpu_blocks: List[int] = [0] * num_instances
        self.num_cpu_blocks: List[int] = [0] * num_instances

        # Synchronization
        self.select_lock = asyncio.Lock()

        # Background migration loop
        self.background_loop = None
        self.is_running_flag = False

        # Initialize LoRA placement like dLoRA (分布式权重放置)
        if exec_type != ExecType.REPLICATED:
            self._initialize_lora_placement()

        print(f"[EngineManager] Initialized:")
        print(f"  - Instances: {num_instances}")
        print(f"  - LoRAs: {num_loras}")
        print(f"  - Exec Type: {exec_type.name}")
        print(f"  - Migration Type: {migration_type.name}")
        print(f"  - Initial LoRA Placement: {self.instance_to_loras}")

    def _initialize_lora_placement(self):
        """Initialize LoRA adapter placement across instances.

        等价于 dLoRA 中 find_best_lora_weight_schedule(is_init=True) 最后的
        min_replicas = num_groups + num_models - 1 那段逻辑。
        """
        min_replicas = self.num_instances + self.num_loras - 1

        next_instance_id = 0
        next_lora_id = random.randint(0, self.num_loras - 1)
        num_lora_replicas = 0

        while num_lora_replicas < min_replicas:
            # 找下一个有容量的实例
            while len(self.instance_to_loras[next_instance_id]) >= self.num_loras:
                next_instance_id = (next_instance_id + 1) % self.num_instances

            # 找下一个没有放在该实例上的 LoRA
            while next_lora_id in self.instance_to_loras[next_instance_id]:
                next_lora_id = (next_lora_id + 1) % self.num_loras

            self.instance_to_loras[next_instance_id].append(next_lora_id)
            self.lora_to_instances[next_lora_id].append(next_instance_id)

            self.engine_lora_capacity[next_instance_id] = len(
                self.instance_to_loras[next_instance_id]
            )

            next_instance_id = (next_instance_id + 1) % self.num_instances
            next_lora_id = (next_lora_id + 1) % self.num_loras
            num_lora_replicas += 1

    # -------------------------- background loop -------------------------- #

    @property
    def is_running(self) -> bool:
        """Check if background migration loop is running."""
        return self.is_running_flag and self.background_loop is not None

    def start_background_loop(self) -> None:
        """Start the background migration loop (for PERIOD_MIG)."""
        if self.is_running:
            raise RuntimeError("Background loop is already running.")

        print("[EngineManager] Starting background migration loop...")
        self.is_running_flag = True

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        self.background_loop = loop.create_task(self._run_migration_loop())

    def stop_background_loop(self) -> None:
        """Stop the background migration loop."""
        print("[EngineManager] Stopping background migration loop...")
        self.is_running_flag = False
        if self.background_loop:
            self.background_loop.cancel()

    async def _run_migration_loop(self):
        """Background loop for periodic migration."""
        while self.is_running_flag:
            try:
                await asyncio.sleep(self.migration_interval)
                if self.migration_type == MigrationType.PERIOD_MIG:
                    await self._migration_schedule()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[EngineManager] Error in migration loop: {e}")

    # -------------------------- selection logic -------------------------- #

    async def select_instance(
        self,
        request_id: str,
        lora_id: int,
    ) -> Tuple[int, str]:
        """
        Select the best instance for a request with specific LoRA.

        Args:
            request_id: Unique request identifier
            lora_id: LoRA adapter ID

        Returns:
            (instance_id, instance_url)
        """
        async with self.select_lock:
            # 更新预期分布（类似 expected_lora_distribution[model_id] += 1）
            if 0 <= lora_id < self.num_loras:
                self.expected_lora_distribution[lora_id] += 1

            # 更新每个实例的负载指标（当前为 mock，未来可接真实 metrics）
            await self._update_instance_metrics()

            # 如果该 LoRA 目前没有放在任何实例上，把它放到最空闲的实例上
            if len(self.lora_to_instances[lora_id]) == 0:
                instance_id = self.instance_num_requests.index(
                    min(self.instance_num_requests)
                )
                self.lora_to_instances[lora_id] = [instance_id]
                self.instance_to_loras[instance_id].append(lora_id)
                print(
                    f"[EngineManager] Adding LoRA {lora_id} to instance {instance_id}"
                )

            # DISPATCH_ONLY 情况：简单选择 cost 最小的实例
            candidate_instances: Dict[int, float] = {}
            for instance_id in self.lora_to_instances[lora_id]:
                candidate_instances[instance_id] = self.instance_exec_cost.get(
                    instance_id, 0.0
                )

            # TODO: 如果想支持 DISPATCH_MIG，可以在这里实现“替换某个 LoRA”逻辑，
            # 对齐原 dLoRA 的 EngineManager.select_engine 内的 sub_engine_model_ids 分支。

            # 在候选实例中选 execution cost 最小的
            min_cost = math.inf
            min_instance_id = None
            for instance_id, cost in candidate_instances.items():
                if cost < min_cost:
                    min_cost = cost
                    min_instance_id = instance_id

            if min_instance_id is None:
                # 兜底：如果上面没有选出，就选请求数最少的实例
                min_instance_id = self.instance_num_requests.index(
                    min(self.instance_num_requests)
                )

            return min_instance_id, self.instance_urls[min_instance_id]

    async def _update_instance_metrics(self):
        """Update metrics for all instances.

        目前是 mock 数据：
          - 对每个实例中已放置的 LoRA，随机生成该 LoRA 的请求数
          - 根据 dLoRA 论文中的合并代价模型估算 instance_exec_cost
        将来可以改成：
          - 请求每个 SGLang 实例的 /metrics HTTP 接口
          - 使用真实的 req_lora_cnt、num_free_blocks 等指标
        """
        for instance_id in range(self.num_instances):
            # Mock: 每个 LoRA 在该实例上的请求数
            req_lora_cnt: Dict[int, int] = {}
            for lora_id in self.instance_to_loras[instance_id]:
                req_lora_cnt[lora_id] = random.randint(0, 20)

            self.instance_req_lora_cnt[instance_id] = req_lora_cnt

            # 按 dLoRA 的方式计算执行开销：批量合并 + merge_speed_ratio
            cost = 0.0
            res_cnt = 0
            for _, cnt in req_lora_cnt.items():
                res_cnt += cnt % self.max_loras_per_batch
                cost += (cnt // self.max_loras_per_batch) * self.merge_speed_ratio
            cost += res_cnt // self.max_loras_per_batch

            self.instance_exec_cost[instance_id] = cost
            self.instance_num_requests[instance_id] = sum(req_lora_cnt.values())

    # -------------------------- migration logic -------------------------- #

    def _get_migration_info(self) -> bool:
        """
        Collect migration information from all instances.

        Returns:
            True if migration is needed, False otherwise
        """
        need_migration = False
        self.reqs_metadata = {}
        self.lora_exec_info = {i: (0, 0.0) for i in range(self.num_loras)}

        # 这里暂时没有真实 request-level Metadata，只用 instance_num_requests 做触发判断。
        for instance_id in range(self.num_instances):
            num_reqs = self.instance_num_requests[instance_id]
            if num_reqs > self.max_loras_per_batch:
                need_migration = True

            # 占位：可以在后续从 /metrics 中获取真实的 RequestMetadata
            self.reqs_metadata[instance_id] = []

            # 统计 LoRA 执行信息
            for lora_id in self.instance_to_loras[instance_id]:
                exec_cnt = self.instance_req_lora_cnt[instance_id].get(lora_id, 0)
                current_cnt, current_time = self.lora_exec_info[lora_id]
                self.lora_exec_info[lora_id] = (
                    current_cnt + exec_cnt,
                    current_time + exec_cnt * self.default_exec_time,
                )

        # 计算平均执行时间
        self.lora_avg_exec_time = []
        for lora_id in range(self.num_loras):
            cnt, total_time = self.lora_exec_info[lora_id]
            if cnt > 0:
                self.lora_avg_exec_time.append(total_time / cnt)
            else:
                self.lora_avg_exec_time.append(self.default_exec_time)

        return need_migration

    async def _migration_schedule(self):
        """Execute periodic migration scheduling using ILP (for PERIOD_MIG)."""
        async with self.select_lock:
            need_migration = self._get_migration_info()

            if not need_migration:
                print("[EngineManager] No migration needed")
                return

            # 根据 reqs_metadata 看各实例的 request 数（占位：目前为 0）
            instance_req_cnt = {
                i: len(reqs) for i, reqs in self.reqs_metadata.items()
            }
            num_reqs = sum(instance_req_cnt.values())

            if num_reqs == 0:
                # 目前没有真实的 request metadata，不做迁移
                print("[EngineManager] No request metadata, skip migration")
                return

            # 按请求数从大到小排序，找最忙实例
            instance_req_cnt = sorted(
                instance_req_cnt.items(), key=lambda x: x[1], reverse=True
            )

            matched_instances: List[int] = []
            remaining_instances = list(instance_req_cnt)

            while remaining_instances:
                avg_num_reqs = num_reqs / len(remaining_instances)
                most_req_instance_id, most_req_instance_cnt = remaining_instances.pop(0)
                num_reqs -= most_req_instance_cnt
                delta = most_req_instance_cnt - avg_num_reqs

                if delta < self.migration_req_threshold:
                    return

                # 找负载较轻的实例
                for instance_id, cnt in remaining_instances:
                    if delta <= 0 or len(matched_instances) > 0:
                        break
                    to_fill = avg_num_reqs - cnt
                    if to_fill <= 0:
                        break
                    to_fill = min(to_fill, delta)
                    matched_instances.append(instance_id)
                    delta -= to_fill

                if matched_instances:
                    break

            if not matched_instances:
                return

            print(
                f"[EngineManager] Migration: Instance {most_req_instance_id} -> {matched_instances}"
            )

            await self._execute_ilp_migration(most_req_instance_id, matched_instances)

    async def _execute_ilp_migration(
        self,
        src_instance: int,
        dst_instances: List[int],
    ):
        """
        Execute ILP-based migration between instances.

        NOTE: 当前版本只更新 LoRA <-> 实例 映射，不做真正的 request-level 迁移，
        即：只影响后续请求的路由，不会把已经在跑的请求从一个实例迁走。
        """
        ilp_engines = sorted([src_instance] + dst_instances)
        ilp_engine_mapping = {ilp_engines[i]: i for i in range(len(ilp_engines))}

        # 收集参与迁移的 LoRA 集合
        ilp_loras = set()
        for engine_id in ilp_engines:
            for lora_id in self.instance_to_loras[engine_id]:
                ilp_loras.add(lora_id)
        ilp_loras = sorted(list(ilp_loras))
        ilp_lora_mapping = {ilp_loras[i]: i for i in range(len(ilp_loras))}

        # 目前 reqs_metadata 全是空的，占位逻辑
        ilp_reqs_metadata: List[RequestMetadata] = []
        for engine_id in ilp_engines:
            for req_metadata in self.reqs_metadata[engine_id]:
                req_metadata.engine_id = ilp_engine_mapping[engine_id]
                req_metadata.model_id = ilp_lora_mapping[req_metadata.model_id]
            ilp_reqs_metadata.extend(self.reqs_metadata[engine_id])

        # ILP 参数
        ilp_num_instances = len(ilp_engines)
        ilp_num_loras = len(ilp_loras)
        ilp_num_gpu_blocks = [self.num_gpu_blocks[e] for e in ilp_engines]
        ilp_num_cpu_blocks = [self.num_cpu_blocks[e] for e in ilp_engines]
        ilp_lora_capacity = [len(self.instance_to_loras[e]) for e in ilp_engines]
        ilp_lora_avg_exec_time = [
            self.lora_avg_exec_time[lora_id] for lora_id in ilp_loras
        ]

        # 构建 LoRA -> engine 的初始映射
        ilp_lora_to_instances: Dict[int, List[int]] = {i: [] for i in range(ilp_num_loras)}
        for engine_id, lora_ids in self.instance_to_loras.items():
            if engine_id not in ilp_engines:
                continue
            for lora_id in lora_ids:
                if lora_id in ilp_lora_mapping:
                    ilp_lora_to_instances[ilp_lora_mapping[lora_id]].append(
                        ilp_engine_mapping[engine_id]
                    )

        print(
            f"[EngineManager] Solving ILP with {len(ilp_reqs_metadata)} requests, "
            f"{ilp_num_instances} instances, {ilp_num_loras} LoRAs..."
        )
        migration_ilp = MigrationILP(
            reqs_metadata=ilp_reqs_metadata,
            num_groups=ilp_num_instances,
            num_models=ilp_num_loras,
            engine_gpu_blocks=ilp_num_gpu_blocks,
            engine_cpu_blocks=ilp_num_cpu_blocks,
            engine_lora_capacity=ilp_lora_capacity,
            lora_exec_time=ilp_lora_avg_exec_time,
            alpha=0.1,
            bw=PCIE_BANDWIDTH / self.default_exec_time,
            model_engine_mapping=ilp_lora_to_instances,
        )

        req_migration_decision, lora_migration_decision, lora_weight_cnt = (
            migration_ilp.solve()
        )

        if req_migration_decision is None:
            print("[EngineManager] ILP migration failed")
            return

        print("[EngineManager] Applying migration decisions...")
        for src_id, decision in req_migration_decision.items():
            src_eng = ilp_engines[src_id]
            for dst_id, req_ids in decision.items():
                dst_eng = ilp_engines[dst_id]
                if req_ids:
                    print(
                        f"  - (placeholder) move {len(req_ids)} requests from {src_eng} to {dst_eng}"
                    )
                    # TODO: 如果将来 SGLang 暴露 request-level 管理 API，可在这里真正迁移正在执行的请求。

        # 更新 LoRA -> 实例映射
        for engine_id, lora_ids in lora_migration_decision.items():
            real_engine_id = ilp_engines[engine_id]
            self.instance_to_loras[real_engine_id] = [
                ilp_loras[lora_id] for lora_id in lora_ids
            ]

        # 重建 lora_to_instances
        self.lora_to_instances = {i: [] for i in range(self.num_loras)}
        for engine_id, lora_ids in self.instance_to_loras.items():
            for lora_id in lora_ids:
                self.lora_to_instances[lora_id].append(engine_id)

        print(f"[EngineManager] Migration completed")
        print(f"  - New LoRA placement: {self.instance_to_loras}")

    # -------------------------- statistics -------------------------- #

    def get_stats(self) -> Dict:
        """Get current manager statistics."""
        return {
            "num_instances": self.num_instances,
            "num_loras": self.num_loras,
            "instance_requests": self.instance_num_requests,
            "lora_placement": self.instance_to_loras,
            "instance_exec_cost": self.instance_exec_cost,
            "migration_type": self.migration_type.name,
            "exec_type": self.exec_type.name,
        }


def create_engine_manager(
    backend: str,
    num_instances: int,
    num_loras: int,
    base_url: str,
    base_port: int,
    **kwargs,
) -> EngineManager:
    """
    Factory for EngineManager, 模仿 dLoRA 的构造接口，但适配 HTTP 实例。

    Args:
        backend: 'sglang' or 'dlora'（目前只影响日志）
        num_instances: Number of SGLang instances
        num_loras: Number of LoRA adapters
        base_url: Base URL, e.g. "127.0.0.1"
        base_port: Starting port, e.g. 30001
        **kwargs: Extra args for EngineManager (exec_type, migration_type, ...)

    Returns:
        EngineManager instance.
    """
    instance_urls = [f"http://{base_url}:{base_port + i}" for i in range(num_instances)]

    exec_type = kwargs.pop("exec_type", ExecType.LORA)
    migration_type = kwargs.pop("migration_type", MigrationType.PERIOD_MIG)

    return EngineManager(
        exec_type=exec_type,
        migration_type=migration_type,
        num_instances=num_instances,
        num_loras=num_loras,
        instance_urls=instance_urls,
        **kwargs,
    )
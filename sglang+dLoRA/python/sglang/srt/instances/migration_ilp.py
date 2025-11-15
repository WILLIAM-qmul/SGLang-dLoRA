"""
ILP-based migration solver for LoRA adapters.
Adapted from: https://github.com/LLMServe/dLoRA-artifact (vllm/engine/migration_ilp.py)
"""

import multiprocessing
import time
from typing import Dict, List, Optional, Tuple

import pulp


class MigrationILP:
    """
    Integer Linear Programming solver for LoRA migration.

    Minimizes execution time while balancing load across instances.
    """

    def __init__(
        self,
        reqs_metadata: List,  # List[RequestMetadata]
        num_groups: int,
        num_models: int,
        engine_gpu_blocks: List[int],
        engine_cpu_blocks: List[int],
        engine_lora_capacity: List[int],
        lora_exec_time: List[float],
        alpha: float,
        bw: float,
        model_engine_mapping: Dict[int, List[int]],
    ):
        """
        Initialize ILP solver.

        Args:
            reqs_metadata: List of request metadata
            num_groups: Number of engine instances
            num_models: Number of LoRA models
            engine_gpu_blocks: GPU blocks per engine
            engine_cpu_blocks: CPU blocks per engine
            engine_lora_capacity: LoRA capacity per engine
            lora_exec_time: Execution time per LoRA
            alpha: Weight for memory cost
            bw: Bandwidth factor
            model_engine_mapping: Current LoRA-to-engine mapping
        """
        print(
            f"[ILP] Initializing: {len(reqs_metadata)} reqs, "
            f"{num_groups} instances, {num_models} LoRAs"
        )

        self.reqs_metadata_mapping = {
            i: reqs_metadata[i] for i in range(len(reqs_metadata))
        }
        self.num_reqs = len(reqs_metadata)  # i
        self.num_groups = num_groups  # j
        self.num_models = num_models  # k
        self.engine_gpu_blocks = engine_gpu_blocks
        self.engine_cpu_blocks = engine_cpu_blocks
        self.engine_lora_capacity = engine_lora_capacity
        self.lora_exec_time = lora_exec_time
        self.alpha = alpha
        self.B = bw
        self.model_engine_mapping = model_engine_mapping

        # Problem
        self.prob = pulp.LpProblem("LoRA_Migration_ILP", pulp.LpMinimize)

        # Decision variables:
        # x[i, j]: whether request i is assigned to group j
        self.x = pulp.LpVariable.dicts(
            "x",
            ((i, j) for i in range(self.num_reqs) for j in range(self.num_groups)),
            cat="Binary",
        )
        # y[k, j]: whether model k is placed on group j
        self.y = pulp.LpVariable.dicts(
            "y",
            ((k, j) for k in range(self.num_models) for j in range(self.num_groups)),
            cat="Binary",
        )
        # max_time: maximum execution time across all groups
        self.max_time = pulp.LpVariable("max_time", lowBound=0, cat="Continuous")
        # max_mem[j]: memory overflow for group j
        self.max_mem = pulp.LpVariable.dicts(
            "z", (j for j in range(self.num_groups)), lowBound=0
        )

        self._add_constraints()
        self._set_init_value()
        self._set_objective()

    def _add_constraints(self):
        """Add ILP constraints."""
        # Constraint 1: each request assigned to at least one group
        for i in range(self.num_reqs):
            self.prob += (
                pulp.lpSum([self.x[(i, j)] for j in range(self.num_groups)]) >= 1
            )

        # Constraint 2: if request i uses model k on group j, model k must be placed on j
        for k in range(self.num_models):
            num_reqs = len(
                [
                    req
                    for req in self.reqs_metadata_mapping.values()
                    if req.model_id == k
                ]
            )
            for j in range(self.num_groups):
                self.prob += num_reqs * self.y[(k, j)] >= pulp.lpSum(
                    [
                        self.x[(i, j)]
                        for i in range(self.num_reqs)
                        if self.reqs_metadata_mapping[i].model_id == k
                    ]
                )

        # Constraint 3: LoRA capacity per group
        for j in range(self.num_groups):
            self.prob += (
                pulp.lpSum([self.y[(k, j)] for k in range(self.num_models)])
                <= self.engine_lora_capacity[j]
            )

    def _set_init_value(self):
        """Warm start with current assignment."""
        for i in range(self.num_reqs):
            for j in range(self.num_groups):
                self.x[(i, j)].setInitialValue(
                    self.reqs_metadata_mapping[i].engine_id == j
                )
        for k in range(self.num_models):
            for j in range(self.num_groups):
                self.y[(k, j)].setInitialValue(j in self.model_engine_mapping.get(k, []))

    def _set_objective(self):
        """Set objective: minimize max_time (execution) + memory penalty."""
        for j in range(self.num_groups):
            # Memory overflow
            self.prob += self.max_mem[j] >= (
                pulp.lpSum(
                    [
                        self.reqs_metadata_mapping[i].num_blocks * self.x[(i, j)]
                        for i in range(self.num_reqs)
                    ]
                )
                - self.engine_gpu_blocks[j]
            ) / self.B

            # Execution time (per group)
            self.prob += self.max_time >= pulp.lpSum(
                [
                    self.lora_exec_time[self.reqs_metadata_mapping[i].model_id]
                    * self.x[(i, j)]
                    for i in range(self.num_reqs)
                ]
            ) + pulp.lpSum(
                [
                    0.05 * self.x[(i, j)]
                    for i in range(self.num_reqs)
                    if self.reqs_metadata_mapping[i].engine_id != j
                ]
            )

        # Objective: minimize max_time (memory penalty implicitly included via max_mem)
        self.prob += self.max_time

    def solve(
        self,
        time_limit: int = 600,
        verbose: bool = False,
    ) -> Tuple[
        Optional[Dict[int, Dict[int, List[str]]]],
        Optional[Dict[int, List[int]]],
        Optional[List[int]],
    ]:
        """
        Solve the ILP problem.

        Returns:
            req_migration_mapping: {src_group: {dst_group: [req_id...]}}
            lora_weight_mapping: {group: [model_id...]}
            lora_weight_cnt: [replica_count_per_model...]
        """
        req_migration_mapping: Dict[int, Dict[int, List[str]]] = {
            i: {j: [] for j in range(self.num_groups)} for i in range(self.num_groups)
        }
        lora_weight_mapping: Dict[int, List[int]] = {
            i: [] for i in range(self.num_groups)
        }
        lora_weight_cnt: List[int] = [0 for _ in range(self.num_models)]

        start = time.time()
        solver = pulp.PULP_CBC_CMD(
            mip=True,
            msg=verbose,
            timeLimit=time_limit,
            threads=multiprocessing.cpu_count(),
        )
        self.prob.solve(solver)
        print(
            f"[ILP] Solve time: {time.time() - start:.2f}s, "
            f"variables: {len(self.prob.variables())}, "
            f"constraints: {len(self.prob.constraints)}"
        )

        # If not optimal, return None
        if pulp.LpStatus[self.prob.status] != "Optimal":
            print(f"[ILP] Solve failed with status: {pulp.LpStatus[self.prob.status]}")
            return None, None, None

        # Extract request migration decision
        for i in range(self.num_reqs):
            src_group = self.reqs_metadata_mapping[i].engine_id
            req_id = self.reqs_metadata_mapping[i].request_id
            model_id = self.reqs_metadata_mapping[i].model_id

            max_j = 0
            max_val = 0.0
            for j in range(self.num_groups):
                val = self.x[(i, j)].value()
                if val > max_val:
                    max_val = val
                    max_j = j

            # model must be placed on max_j
            assert self.y[(model_id, max_j)].value() == 1, (
                f"Model {model_id} not assigned to group {max_j}, "
                f"x={max_val}, y={self.y[(model_id, max_j)].value()}"
            )

            if src_group != max_j:
                if verbose:
                    print(
                        f"[ILP] Request {req_id} (model {model_id}) "
                        f"{src_group} -> {max_j} (x={max_val})"
                    )
                req_migration_mapping[src_group][max_j].append(req_id)

        # Extract LoRA placement decision
        for k in range(self.num_models):
            for j in range(self.num_groups):
                if self.y[(k, j)].value() == 1:
                    if verbose:
                        print(f"[ILP] Model {k} -> group {j}")
                    lora_weight_mapping[j].append(k)
                    lora_weight_cnt[k] += 1

        return req_migration_mapping, lora_weight_mapping, lora_weight_cnt
"""
ILP-based migration solver for LoRA adapters.
Based on dLoRA's migration scheduling algorithm.
"""

import pulp
import time
import multiprocessing
from typing import Dict, List, Tuple, Optional


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
            bw: Bandwidth for data transfer
            model_engine_mapping: Current LoRA-to-engine mapping
        """
        print(f"[ILP] Initializing: {len(reqs_metadata)} reqs, "
              f"{num_groups} instances, {num_models} LoRAs")
        
        self.reqs_metadata_mapping = {i: reqs_metadata[i] for i in range(len(reqs_metadata))}
        self.num_reqs = len(reqs_metadata)
        self.num_groups = num_groups
        self.num_models = num_models
        self.engine_gpu_blocks = engine_gpu_blocks
        self.engine_cpu_blocks = engine_cpu_blocks
        self.engine_lora_capacity = engine_lora_capacity
        self.lora_exec_time = lora_exec_time
        self.alpha = alpha
        self.B = bw
        self.model_engine_mapping = model_engine_mapping
        
        # Create ILP problem
        self.prob = pulp.LpProblem('LoRA_Migration_ILP', pulp.LpMinimize)
        
        # Decision variables:
        # x[i, j]: whether request i is assigned to engine j
        self.x = pulp.LpVariable.dicts(
            'x',
            ((i, j) for i in range(self.num_reqs) for j in range(self.num_groups)),
            cat='Binary'
        )
        
        # y[k, j]: whether LoRA k is loaded on engine j
        self.y = pulp.LpVariable.dicts(
            'y',
            ((k, j) for k in range(self.num_models) for j in range(self.num_groups)),
            cat='Binary'
        )
        
        # max_time: maximum execution time across all engines
        self.max_time = pulp.LpVariable('max_time', lowBound=0, cat='Continuous')
        
        # max_mem[j]: memory overflow for engine j
        self.max_mem = pulp.LpVariable.dicts(
            'max_mem',
            (j for j in range(self.num_groups)),
            lowBound=0
        )
        
        self._add_constraints()
        self._set_init_value()
        self._set_objective()

    def _add_constraints(self):
        """Add ILP constraints."""
        # Constraint 1: Each request must be assigned to at least one engine
        for i in range(self.num_reqs):
            self.prob += pulp.lpSum([
                self.x[(i, j)] for j in range(self.num_groups)
            ]) >= 1
        
        # Constraint 2: If a request uses LoRA k on engine j, LoRA k must be loaded
        for k in range(self.num_models):
            num_reqs_with_lora = len([
                req for req in self.reqs_metadata_mapping.values()
                if req.model_id == k
            ])
            for j in range(self.num_groups):
                self.prob += num_reqs_with_lora * self.y[(k, j)] >= pulp.lpSum([
                    self.x[(i, j)] for i in range(self.num_reqs)
                    if self.reqs_metadata_mapping[i].model_id == k
                ])
        
        # Constraint 3: Each engine has limited LoRA capacity
        for j in range(self.num_groups):
            self.prob += pulp.lpSum([
                self.y[(k, j)] for k in range(self.num_models)
            ]) <= self.engine_lora_capacity[j]

    def _set_init_value(self):
        """Set initial values for warm start."""
        # Initialize request assignments
        for i in range(self.num_reqs):
            for j in range(self.num_groups):
                self.x[(i, j)].setInitialValue(
                    self.reqs_metadata_mapping[i].engine_id == j
                )
        
        # Initialize LoRA placements
        for k in range(self.num_models):
            for j in range(self.num_groups):
                self.y[(k, j)].setInitialValue(
                    j in self.model_engine_mapping[k]
                )

    def _set_objective(self):
        """Set ILP objective function."""
        # Objective: minimize maximum execution time + memory penalty
        
        # Memory overflow constraint for each engine
        for j in range(self.num_groups):
            self.prob += self.max_mem[j] >= (
                pulp.lpSum([
                    self.reqs_metadata_mapping[i].num_blocks * self.x[(i, j)]
                    for i in range(self.num_reqs)
                ]) - self.engine_gpu_blocks[j]
            ) / self.B
        
        # Execution time constraint for each engine
        for j in range(self.num_groups):
            self.prob += self.max_time >= pulp.lpSum([
                self.lora_exec_time[self.reqs_metadata_mapping[i].model_id] * self.x[(i, j)]
                for i in range(self.num_reqs)
            ]) + pulp.lpSum([
                0.05 * self.x[(i, j)]  # Migration overhead
                for i in range(self.num_reqs)
                if self.reqs_metadata_mapping[i].engine_id != j
            ])
        
        # Objective: minimize max execution time
        self.prob += self.max_time

    def solve(
        self,
        time_limit: int = 600,
        verbose: bool = False
    ) -> Tuple[
        Optional[Dict[int, Dict[int, List[str]]]],
        Optional[Dict[int, List[int]]],
        Optional[List[int]]
    ]:
        """
        Solve the ILP problem.
        
        Args:
            time_limit: Maximum solving time in seconds
            verbose: Whether to print solver output
            
        Returns:
            Tuple of (req_migration_mapping, lora_weight_mapping, lora_weight_cnt)
            Returns (None, None, None) if solve fails
        """
        req_migration_mapping = {
            i: {j: [] for j in range(self.num_groups)}
            for i in range(self.num_groups)
        }
        lora_weight_mapping = {i: [] for i in range(self.num_groups)}
        lora_weight_cnt = [0] * self.num_models
        
        start = time.time()
        
        # Configure solver
        solver = pulp.PULP_CBC_CMD(
            mip=True,
            msg=verbose,
            timeLimit=time_limit,
            threads=multiprocessing.cpu_count()
        )
        
        # Solve
        self.prob.solve(solver)
        
        solve_time = time.time() - start
        print(f"[ILP] Solve time: {solve_time:.2f}s, "
              f"Variables: {len(self.prob.variables())}, "
              f"Constraints: {len(self.prob.constraints)}")
        
        # Check solution status
        if self.prob.status != pulp.LpStatusOptimal:
            print(f"[ILP] Solve failed with status: {pulp.LpStatus[self.prob.status]}")
            return None, None, None
        
        # Extract solution
        for i in range(self.num_reqs):
            src_group = self.reqs_metadata_mapping[i].engine_id
            req_id = self.reqs_metadata_mapping[i].request_id
            model_id = self.reqs_metadata_mapping[i].model_id
            
            # Find assigned engine
            max_j = 0
            max_val = 0.0
            for j in range(self.num_groups):
                if self.x[(i, j)].value() > max_val:
                    max_val = self.x[(i, j)].value()
                    max_j = j
            
            # Verify LoRA is available
            assert self.y[(model_id, max_j)].value() == 1, (
                f"LoRA {model_id} not available on engine {max_j}"
            )
            
            # Record migration if needed
            if src_group != max_j:
                if verbose:
                    print(f"[ILP] Request {req_id} (LoRA {model_id}): "
                          f"{src_group} -> {max_j}")
                req_migration_mapping[src_group][max_j].append(req_id)
        
        # Extract LoRA placements
        for k in range(self.num_models):
            for j in range(self.num_groups):
                if self.y[(k, j)].value() == 1:
                    if verbose:
                        print(f"[ILP] LoRA {k} -> Engine {j}")
                    lora_weight_mapping[j].append(k)
                    lora_weight_cnt[k] += 1
        
        return req_migration_mapping, lora_weight_mapping, lora_weight_cnt
# File: sglang+dLoRA/python/sglang/srt/instances/migration_ilp.py
"""
Migration ILP solver adapted from dLoRA-artifact for SGLang. 

This module solves the request migration and LoRA placement problem using
Integer Linear Programming (ILP) to minimize maximum completion time across engines. 

Key differences from dLoRA:
1. Uses RequestMetadata dataclass instead of vllm's SequenceGroupMetadata
2. Adapted for SGLang's page-based KV cache management
3. Compatible with SGLang's LoRA adapter system
"""

import pulp
from typing import Dict, List, Tuple
import time
import multiprocessing
import logging

logger = logging.getLogger(__name__)


class MigrationILP: 
    """
    Integer Linear Programming solver for request migration optimization.
    
    Objective: Minimize maximum completion time across all engines. 
    
    Decision Variables:
    - x[i,j]:  Binary, whether request i is assigned to engine j
    - y[k,j]: Binary, whether model k is loaded on engine j
    - max_time:  Continuous, maximum completion time across all engines
    - max_mem[j]: Continuous, memory swap time for engine j
    
    Constraints:
    1. Each request must be assigned to at least one engine
    2. If a request is assigned to an engine, its model must be loaded there
    3. Each engine has a maximum LoRA capacity
    4. Memory and time constraints
    """

    def __init__(
        self,
        reqs_metadata: List,  # List[RequestMetadata]
        num_groups: int,
        num_models: int,
        engine_gpu_blocks: List[int],
        engine_cpu_blocks:  List[int],
        engine_lora_capacity: List[int],
        lora_exec_time: List[float],
        alpha: float,
        bw: float,  # bandwidth in blocks/sec
        model_engine_mapping: Dict[int, List[int]],
    ):
        """
        Initialize the ILP solver.
        
        Args:
            reqs_metadata: List of RequestMetadata objects
            num_groups: Number of engines
            num_models: Number of models (LoRA adapters)
            engine_gpu_blocks: GPU page capacity for each engine
            engine_cpu_blocks: CPU page capacity for each engine (not used in SGLang)
            engine_lora_capacity: Max number of LoRA adapters per engine
            lora_exec_time: Average execution time per model
            alpha: Migration overhead coefficient
            bw: Bandwidth for page transfer (pages per second)
            model_engine_mapping: Current model->engine mapping
        """
        logger.info(
            f"Initializing MigrationILP:  "
            f"num_reqs={len(reqs_metadata)}, "
            f"num_groups={num_groups}, "
            f"num_models={num_models}"
        )
        
        # Store parameters
        self.reqs_metadata_mapping = {i: reqs_metadata[i] for i in range(len(reqs_metadata))}
        self.num_reqs = len(reqs_metadata)
        self.num_groups = num_groups
        self. num_models = num_models
        self.engine_gpu_blocks = engine_gpu_blocks
        self.engine_cpu_blocks = engine_cpu_blocks
        self.engine_lora_capacity = engine_lora_capacity
        self.lora_exec_time = lora_exec_time
        self.alpha = alpha
        self.B = bw
        self.model_engine_mapping = model_engine_mapping
        
        # Create ILP problem
        self.prob = pulp.LpProblem('MigrationILP', pulp.LpMinimize)
        
        # Decision variables
        self.x = pulp.LpVariable.dicts(
            'x',
            ((i, j) for i in range(self.num_reqs) for j in range(self.num_groups)),
            cat='Binary'
        )
        self.y = pulp.LpVariable.dicts(
            'y',
            ((k, j) for k in range(self. num_models) for j in range(self.num_groups)),
            cat='Binary'
        )
        self.max_time = pulp. LpVariable('max_time', lowBound=0, cat='Continuous')
        self.max_mem = pulp.LpVariable. dicts(
            'z',
            (j for j in range(self. num_groups)),
            lowBound=0
        )
        
        # Build the problem
        self._add_constraints()
        self._set_init_value()
        self._set_objective()

    def _add_constraints(self):
        """Add constraints to the ILP problem."""
        
        # Constraint 1: Each request must be assigned to at least one engine
        for i in range(self.num_reqs):
            self.prob += pulp.lpSum([self.x[(i, j)] for j in range(self.num_groups)]) >= 1

        # Constraint 2: If request i is assigned to engine j, its model must be loaded on j
        for k in range(self.num_models):
            # Count requests for this model
            num_reqs_model_k = len([
                req for req in self.reqs_metadata_mapping.values()
                if self._get_model_id(req) == k
            ])
            
            if num_reqs_model_k == 0:
                continue
            
            for j in range(self.num_groups):
                # If any request of model k is on engine j, then y[k,j] must be 1
                self.prob += num_reqs_model_k * self.y[(k, j)] >= pulp.lpSum([
                    self.x[(i, j)]
                    for i in range(self.num_reqs)
                    if self._get_model_id(self.reqs_metadata_mapping[i]) == k
                ])

        # Constraint 3: LoRA capacity constraint
        for j in range(self.num_groups):
            self.prob += pulp.lpSum([
                self.y[(k, j)] for k in range(self.num_models)
            ]) <= self.engine_lora_capacity[j]

    def _set_init_value(self):
        """Set initial values for decision variables based on current state."""
        
        # Initialize x based on current request placement
        for i in range(self.num_reqs):
            for j in range(self.num_groups):
                self.x[(i, j)].setInitialValue(
                    1 if self.reqs_metadata_mapping[i].engine_id == j else 0
                )
        
        # Initialize y based on current model placement
        for k in range(self.num_models):
            for j in range(self.num_groups):
                self.y[(k, j)].setInitialValue(
                    1 if j in self.model_engine_mapping. get(k, []) else 0
                )

    def _set_objective(self):
        """Set the objective function to minimize maximum completion time."""
        
        for j in range(self.num_groups):
            # Memory swap time: pages that exceed GPU capacity must be swapped
            self.prob += self.max_mem[j] >= (
                pulp.lpSum([
                    self.reqs_metadata_mapping[i]. num_pages * self.x[(i, j)]
                    for i in range(self.num_reqs)
                ]) - self.engine_gpu_blocks[j]
            ) / self.B
            
            # Total time = execution time + migration overhead + memory swap time
            # Execution time: sum of model execution times
            # Migration overhead:  alpha * number of migrated requests
            # Memory swap time: max_mem[j]
            self.prob += self. max_time >= pulp.lpSum([
                self.lora_exec_time[self._get_model_id(self.reqs_metadata_mapping[i])] * self.x[(i, j)]
                for i in range(self.num_reqs)
            ]) + pulp.lpSum([
                self.alpha * self.x[(i, j)]
                for i in range(self.num_reqs)
                if self. reqs_metadata_mapping[i].engine_id != j
            ]) + self.max_mem[j]

        # Minimize maximum completion time
        self.prob += self.max_time

    def _get_model_id(self, req_metadata) -> int:
        """
        Extract model ID from request metadata.
        
        Args:
            req_metadata: RequestMetadata object
            
        Returns:
            Model ID (integer), defaults to 0 if not specified
        """
        if hasattr(req_metadata, 'model_id'):
            model_id = req_metadata.model_id
            if model_id is None:
                return 0
            if isinstance(model_id, str):
                try:
                    # Try to extract numeric ID from string like "lora0"
                    if model_id.startswith("lora"):
                        return int(model_id[4:])
                    return int(model_id)
                except ValueError:
                    return 0
            return int(model_id)
        return 0

    def solve(self) -> Tuple[Dict, Dict, List]: 
        """
        Solve the ILP problem.
        
        Returns:
            Tuple of: 
            - req_migration_mapping: {src_engine: {dst_engine: [req_ids]}}
            - lora_weight_mapping: {engine_id:  [model_ids]}
            - lora_weight_cnt: [count per model]
        """
        verbose = False
        
        # Initialize result structures
        req_migration_mapping = {
            i: {j: [] for j in range(self.num_groups)}
            for i in range(self.num_groups)
        }
        lora_weight_mapping = {i: [] for i in range(self.num_groups)}
        lora_weight_cnt = [0 for _ in range(self.num_models)]
        
        start = time.time()
        time_limit = 600  # 10 minutes max
        
        # Create solver
        solver = pulp. PULP_CBC_CMD(
            mip=True,
            msg=verbose,
            timeLimit=time_limit,
            threads=multiprocessing.cpu_count()
        )
        
        # Solve the problem
        try:
            self.prob.solve(solver)
        except Exception as e:
            logger.error(f"ILP solver failed: {e}")
            return None, None, None
        
        solve_time = time.time() - start
        
        # Check if solution is optimal or feasible
        status = pulp.LpStatus[self.prob.status]
        if status not in ['Optimal', 'Feasible']:
            logger.error(f"ILP solver failed with status: {status}")
            return None, None, None
        
        logger.info(
            f"ILP solved in {solve_time:.2f}s, "
            f"status={status}, "
            f"num_vars={len(self.prob.variables())}, "
            f"num_constraints={len(self.prob. constraints)}"
        )
        
        # Extract solution for requests
        for i in range(self.num_reqs):
            src_engine = self.reqs_metadata_mapping[i].engine_id
            req_id = self.reqs_metadata_mapping[i].request_id
            model_id = self._get_model_id(self.reqs_metadata_mapping[i])
            
            # Find the engine with highest assignment value
            max_j = 0
            max_val = 0.0
            for j in range(self.num_groups):
                val = self.x[(i, j)].value()
                if val is not None and val > max_val:
                    max_val = val
                    max_j = j
            
            # Verify model is loaded on target engine
            y_val = self.y[(model_id, max_j)].value()
            if y_val is None or y_val < 0.5:
                logger.warning(
                    f"Model {model_id} not loaded on engine {max_j} "
                    f"but request {req_id} assigned there"
                )
                continue
            
            # Record migration if engine changed
            if src_engine != max_j:
                if verbose:
                    logger.debug(
                        f"Request {req_id} (model {model_id}): "
                        f"engine {src_engine} -> {max_j}"
                    )
                req_migration_mapping[src_engine][max_j].append(req_id)
        
        # Extract solution for LoRA placement
        for k in range(self.num_models):
            for j in range(self.num_groups):
                val = self.y[(k, j)].value()
                if val is not None and val > 0.5:
                    if verbose:
                        logger.debug(f"Model {k} loaded on engine {j}")
                    lora_weight_mapping[j].append(k)
                    lora_weight_cnt[k] += 1
        
        # Log summary
        total_migrations = sum(
            len(req_ids)
            for dst_mapping in req_migration_mapping.values()
            for req_ids in dst_mapping.values()
        )
        logger.info(
            f"ILP solution:  {total_migrations} request migrations, "
            f"objective value: {pulp.value(self.prob. objective):.2f}"
        )
        
        return req_migration_mapping, lora_weight_mapping, lora_weight_cnt
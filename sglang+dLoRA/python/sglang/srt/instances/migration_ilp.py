# File: sglang/srt/instances/migration_ilp.py
"""
Migration ILP solver from dLoRA-artifact.
Uses PuLP to solve the optimal migration problem. 

Direct copy from dLoRA with minimal changes:
- Updated imports for SGLang
- Uses RequestMetadata from engine_manager
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
    
    Variables:
    - x[i,j]: Binary, whether request i is assigned to engine j
    - y[k,j]: Binary, whether model k is loaded on engine j
    - max_time: Continuous, maximum completion time
    """

    def __init__(
        self,
        reqs_metadata: List,
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
        logger.info(f"Initializing MigrationILP: num_reqs={len(reqs_metadata)}, "
                   f"num_groups={num_groups}, num_models={num_models}")
        
        self. reqs_metadata_mapping = {i: reqs_metadata[i] for i in range(len(reqs_metadata))}
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
        
        self.prob = pulp.LpProblem('MigrationILP', pulp.LpMinimize)
        
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
        self.max_time = pulp.LpVariable('max_time', lowBound=0, cat='Continuous')
        self.max_mem = pulp.LpVariable. dicts(
            'z',
            (j for j in range(self.num_groups)),
            lowBound=0
        )
        
        self._add_constraints()
        self._set_init_value()
        self._set_objective()

    def _add_constraints(self):
        # Constraint 1: Each request must be assigned to at least one engine
        for i in range(self.num_reqs):
            self.prob += pulp.lpSum([self.x[(i, j)] for j in range(self.num_groups)]) >= 1

        # Constraint 2: If request assigned to j, its model must be loaded on j
        for k in range(self.num_models):
            num_reqs_model_k = len([
                req for req in self.reqs_metadata_mapping.values() 
                if req.model_id == k
            ])
            
            for j in range(self. num_groups):
                self. prob += num_reqs_model_k * self.y[(k, j)] >= pulp.lpSum([
                    self.x[(i, j)] 
                    for i in range(self.num_reqs) 
                    if self.reqs_metadata_mapping[i].model_id == k
                ])

        # Constraint 3: LoRA capacity constraint
        for j in range(self.num_groups):
            self.prob += pulp.lpSum([
                self.y[(k, j)] for k in range(self.num_models)
            ]) <= self.engine_lora_capacity[j]

    def _set_init_value(self):
        for i in range(self.num_reqs):
            for j in range(self.num_groups):
                self.x[(i, j)].setInitialValue(
                    1 if self.reqs_metadata_mapping[i].engine_id == j else 0
                )
        
        for k in range(self. num_models):
            for j in range(self.num_groups):
                self.y[(k, j)].setInitialValue(
                    1 if j in self.model_engine_mapping. get(k, []) else 0
                )

    def _set_objective(self):
        for j in range(self.num_groups):
            # Memory swap time
            self.prob += self.max_mem[j] >= (
                pulp.lpSum([
                    self.reqs_metadata_mapping[i]. num_blocks * self.x[(i, j)]
                    for i in range(self.num_reqs)
                ]) - self.engine_gpu_blocks[j]
            ) / self.B
            
            # Execution time + migration overhead
            self.prob += self.max_time >= pulp.lpSum([
                self.lora_exec_time[self.reqs_metadata_mapping[i].model_id] * self.x[(i, j)]
                for i in range(self.num_reqs)
            ]) + pulp.lpSum([
                0.05 * self.x[(i, j)]
                for i in range(self.num_reqs)
                if self.reqs_metadata_mapping[i].engine_id != j
            ]) + self.max_mem[j]

        self.prob += self.max_time

    def solve(self) -> Tuple[Dict, Dict, List]:
        verbose = False
        req_migration_mapping = {
            i: {j: [] for j in range(self.num_groups)} 
            for i in range(self.num_groups)
        }
        lora_weight_mapping = {i: [] for i in range(self.num_groups)}
        lora_weight_cnt = [0 for _ in range(self.num_models)]
        
        start = time.time()
        time_limit = 600
        
        solver = pulp. PULP_CBC_CMD(
            mip=True,
            msg=verbose,
            timeLimit=time_limit,
            threads=multiprocessing.cpu_count()
        )
        
        self.prob.solve(solver)
        
        logger.info(f"ILP solved in {time.time() - start:.2f}s, "
                   f"num_vars={len(self.prob.variables())}, "
                   f"num_constraints={len(self.prob.constraints)}")
        
        # Extract solution
        for i in range(self.num_reqs):
            src_engine = self.reqs_metadata_mapping[i].engine_id
            req_id = self.reqs_metadata_mapping[i].request_id
            model_id = self.reqs_metadata_mapping[i].model_id
            
            max_j = 0
            max_val = 0.0
            for j in range(self.num_groups):
                val = self. x[(i, j)].value()
                if val is not None and val > max_val:
                    max_val = val
                    max_j = j
            
            # Verify model is loaded
            y_val = self.y[(model_id, max_j)]. value()
            if y_val is None or y_val < 0.5:
                logger.warning(f"Model {model_id} not loaded on engine {max_j}")
                continue
            
            if src_engine != max_j:
                if verbose:
                    logger.debug(f"Request {req_id} (model {model_id}): "
                               f"engine {src_engine} -> {max_j}")
                req_migration_mapping[src_engine][max_j].append(req_id)
        
        # Extract LoRA placement
        for k in range(self.num_models):
            for j in range(self.num_groups):
                val = self.y[(k, j)].value()
                if val is not None and val > 0.5:
                    if verbose:
                        logger.debug(f"Model {k} loaded on engine {j}")
                    lora_weight_mapping[j].append(k)
                    lora_weight_cnt[k] += 1
        
        return req_migration_mapping, lora_weight_mapping, lora_weight_cnt
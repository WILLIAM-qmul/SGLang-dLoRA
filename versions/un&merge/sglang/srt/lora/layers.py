import torch
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.lora.backend.base_backend import BaseLoRABackend

"""
Llama-2 with LoRA support layers.
@ each block has 4 layers that need to be replaced with LoRA versions:—— which is controllby target_modules
1. QKVParallelLinear
2. RowParallelLinear
3. MergedColumnParallelLinear
4. RowParallelLinear
"""


class BaseLayerWithLoRA(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        self.set_lora: bool = False
        self.lora_backend: BaseLoRABackend = lora_backend
        self.merge_lora: bool = True  # 新增参数，True 表示合并权重，False 表示不合并权重推理
        self.original_weight: torch.Tensor | None = None # 缓存原始权重到 CPU，避免重复叠加；以及当前是否已合并的标记
        self._is_merged: bool = False # 标记当前权重是否已合并

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args, merge: bool | None = None):
        if merge is not None:
            # When switching from merge=True to merge=False, restore the original weight.
            if self.merge_lora and not bool(merge):
                self._restore_weight_if_needed()
            self.merge_lora = bool(merge)
        pass
    
    def _ensure_original_weight(self):
        # 只缓存一次原始权重
        if self.original_weight is None:
            with torch.no_grad():
                self.original_weight = self.base_layer.weight.data.detach().clone().cpu()

    def _restore_weight_if_needed(self):
        # 仅当已合并时才从缓存恢复，且不要覆盖缓存
        if self._is_merged and self.original_weight is not None:
            with torch.no_grad():
                self.base_layer.weight.data.copy_(
                    self.original_weight.to(
                        device=self.base_layer.weight.device,
                        dtype=self.base_layer.weight.dtype,
                    )
                )
            self._is_merged = False

    def _prepare_merge(self):
        # 每次合并前都从 original_weight 还原，保证幂等
        self._ensure_original_weight()
        with torch.no_grad():
            self.base_layer.weight.data.copy_(
                self.original_weight.to(
                    device=self.base_layer.weight.device,
                    dtype=self.base_layer.weight.dtype,
                )
            )
        self._is_merged = False

    def _finalize_merge_add_(self):
        # Mark the weight as merged. The addition is done by _add_mm_blockwise_.
        # 添加权重验证：检查是否包含 NaN 或无穷大
        if torch.isnan(self.base_layer.weight).any() or torch.isinf(self.base_layer.weight).any():
            # 如果无效，回滚到原始权重并切换到非合并模式
            self._restore_weight_if_needed()
            self.merge_lora = False
            raise ValueError("Merged weight contains NaN or inf, switching to unmerged mode.")
        self._is_merged = True

    def _add_mm_blockwise_(self, w_slice: torch.Tensor, B: torch.Tensor, A: torch.Tensor, init_block_cols: int = 1024):
        # Block-wise B @ A and in-place add to w_slice to reduce peak memory usage.
        # Computation is done in float32 for numerical stability.
        with torch.no_grad():
            dev = w_slice.device
            out_dtype = w_slice.dtype
            Bf = B.to(dev, dtype=torch.float32)
            Af = A.to(dev, dtype=torch.float32)
            M, r = Bf.shape
            r2, N = Af.shape
            assert r == r2, f"LoRA shapes mismatch: B {tuple(B.shape)}, A {tuple(A.shape)}"
            assert w_slice.shape == (M, N), f"Weight slice shape {tuple(w_slice.shape)} != {(M, N)}"

            block = min(init_block_cols, N) if N > 0 else 0
            if block == 0:
                return
            
            for c in range(0, N, block):
                c2 = min(c + block, N)
                # Compute a block of the delta
                delta_block = Bf @ Af[:, c:c2]
                # 添加数值稳定处理：限制范围并处理 NaN
                delta_block = torch.clamp(delta_block, min=-1e6, max=1e6)
                delta_block = torch.nan_to_num(delta_block, nan=0.0, posinf=1e6, neginf=-1e6)
                # Add the block to the corresponding slice of the weight
                w_slice[:, c:c2].add_(delta_block.to(out_dtype))

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        pass

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        pass


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    """
    Vocab parallel embedding layer with support for LoRA (Low-Rank Adaptation).

    Note: The current version does not yet implement the LoRA functionality.
    This class behaves exactly the same as the base VocabParallelEmbedding.
    Future versions will integrate LoRA functionality to support efficient parameter fine-tuning.
    """

    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        self.weight = base_layer.weight


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: ColumnParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        shard_size = self.base_layer.output_partition_sizes[0]
        self.output_offset = torch.tensor(
            [
                0,
                shard_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )
        self.original_weight = None

    def set_lora_info(
        self,
        A_buffer: torch.Tensor | None,
        B_buffer: torch.Tensor | None,
        merge: bool | None = None,
    ):
        super().set_lora_info(merge=merge)
        
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer
        
        if self.merge_lora == True and self.A_buffer is not None and self.B_buffer is not None:
            self._prepare_merge()
            lora_A = A_buffer[0]
            lora_B = B_buffer[0]
            w = self.base_layer.weight.data
            self._add_mm_blockwise_(w, lora_B, lora_A)
            self._finalize_merge_add_()
            

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.B_buffer,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output

    def forward(self, input_: torch.Tensor):
        if self.merge_lora == False: # False 表示非合并 WX+ABX；
            # duplicate the logic in ColumnParallelLinear
            bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
            print("base_layer weight shape:", self.base_layer.weight.shape)
            output_parallel = self.base_layer.quant_method.apply(
                self.base_layer, input_, bias
            )

            if self.set_lora:
                output_parallel = self.apply_lora(output_parallel, input_)

            if self.base_layer.gather_output:
                output = tensor_model_parallel_all_gather(output_parallel)
            else:
                output = output_parallel
            output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        else: # True 表示合并 (W+AB)X；
            bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
            output_parallel = self.base_layer.quant_method.apply(
                self.base_layer, input_, bias
            )
            if self.base_layer.gather_output:
                output = tensor_model_parallel_all_gather(output_parallel)
            else:
                output = output_parallel
            output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
            
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        B = B[start_idx:end_idx, :]
        return B


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: MergedColumnParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

    def set_lora_info(
        self,
        A_buffer: torch.Tensor | None,
        B_buffer: torch.Tensor | None,
        merge: bool | None = None,
    ):
        self.set_lora = True
        self.A_buffer_gate_up = A_buffer
        self.B_buffer_gate_up = B_buffer
        if merge is not None:
            self.merge_lora = bool(merge)
            
        if self.merge_lora == True and self.A_buffer_gate_up is not None and self.B_buffer_gate_up is not None:
            self._prepare_merge()
            lora_A = self.A_buffer_gate_up[0]
            lora_B = self.B_buffer_gate_up[0]
            r = lora_B.shape[1]
            out_shard_size = self.base_layer.output_partition_sizes[0]

            lora_A_gate, lora_A_up = lora_A.split(r, dim=0)
            lora_B_gate, lora_B_up = lora_B.split(out_shard_size, dim=0)

            w = self.base_layer.weight.data
            w_gate = w[:out_shard_size]
            w_up = w[out_shard_size:]

            self._add_mm_blockwise_(w_gate, lora_B_gate, lora_A_gate)
            self._add_mm_blockwise_(w_up, lora_B_up, lora_A_up)
            self._finalize_merge_add_()
        
        shard_size = self.base_layer.output_partition_sizes[0]
        self.output_offset = torch.tensor(
            [
                0,
                shard_size,
                2 * shard_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_output = self.lora_backend.run_gate_up_lora(
            x=x,
            gate_up_lora_a=self.A_buffer_gate_up,
            gate_up_lora_b=self.B_buffer_gate_up,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output
    
    def forward(self, input_: torch.Tensor):
        if self.merge_lora == False: # False 表示非合并 WX+ABX；
            # duplicate the logic in ColumnParallelLinear
            bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
            print("base_layer weight shape:", self.base_layer.weight.shape)
            output_parallel = self.base_layer.quant_method.apply(
                self.base_layer, input_, bias
            )

            if self.set_lora:
                output_parallel = self.apply_lora(output_parallel, input_)

            if self.base_layer.gather_output:
                output = tensor_model_parallel_all_gather(output_parallel)
            else:
                output = output_parallel
            output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        else: # True 表示合并 (W+AB)X；
            bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
            output_parallel = self.base_layer.quant_method.apply(
                self.base_layer, input_, bias
            )
            if self.base_layer.gather_output:
                output = tensor_model_parallel_all_gather(output_parallel)
            else:
                output = output_parallel
            output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
            
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # Since the outputs for both gate and up are identical, we use a random one.
        shard_size = self.base_layer.output_partition_sizes[0]
        gate_size = self.base_layer.output_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return torch.concat(
            (
                B[start_idx:end_idx, :],
                B[gate_size + start_idx : gate_size + end_idx],
            ),
            dim=0,
        )


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: QKVParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        q_proj_shard_size = self.base_layer.q_proj_shard_size
        kv_proj_shard_size = self.base_layer.kv_proj_shard_size
        self.output_offset = torch.tensor(
            [
                0,
                q_proj_shard_size,
                q_proj_shard_size + kv_proj_shard_size,
                q_proj_shard_size + 2 * kv_proj_shard_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )

        # For computing number of launched blocks
        self.max_qkv_out_dim = max(q_proj_shard_size, kv_proj_shard_size)

    def set_lora_info(
        self,
        A_buffer_qkv: torch.Tensor | None,
        B_buffer_qkv: torch.Tensor | None,
        merge: bool | None = None,
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv
        self.B_buffer_qkv = B_buffer_qkv
        if merge is not None:
            self.merge_lora = bool(merge)
            
        if self.merge_lora == True and self.A_buffer_qkv is not None and self.B_buffer_qkv is not None:
            self._prepare_merge()
            lora_A = A_buffer_qkv[0]
            lora_B = B_buffer_qkv[0]
            r = lora_A.shape[0] // 3
            q_shard = self.base_layer.q_proj_shard_size
            kv_shard = self.base_layer.kv_proj_shard_size

            lora_A_q, lora_A_k, lora_A_v = lora_A.split(r, dim=0)
            lora_B_q, lora_B_k, lora_B_v = lora_B.split([q_shard, kv_shard, kv_shard], dim=0)

            w = self.base_layer.weight.data
            w_q = w[:q_shard]
            w_k = w[q_shard : q_shard + kv_shard]
            w_v = w[q_shard + kv_shard :]

            self._add_mm_blockwise_(w_q, lora_B_q, lora_A_q)
            self._add_mm_blockwise_(w_k, lora_B_k, lora_A_k)
            self._add_mm_blockwise_(w_v, lora_B_v, lora_A_v)
            self._finalize_merge_add_()

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_output = self.lora_backend.run_qkv_lora(
            x=x,
            qkv_lora_a=self.A_buffer_qkv,
            qkv_lora_b=self.B_buffer_qkv,
            base_output=base_output,
            output_offset=self.output_offset,
            max_qkv_out_dim=self.max_qkv_out_dim,
        )
        return lora_output
    
    def forward(self, input_: torch.Tensor):
        if self.merge_lora == False: # False 表示非合并 WX+ABX；
            # duplicate the logic in ColumnParallelLinear
            bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
            print("base_layer weight shape:", self.base_layer.weight.shape)
            output_parallel = self.base_layer.quant_method.apply(
                self.base_layer, input_, bias
            )

            if self.set_lora:
                output_parallel = self.apply_lora(output_parallel, input_)

            if self.base_layer.gather_output:
                output = tensor_model_parallel_all_gather(output_parallel)
            else:
                output = output_parallel
            output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        else: # True 表示合并 (W+AB)X；
            bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
            output_parallel = self.base_layer.quant_method.apply(
                self.base_layer, input_, bias
            )
            if self.base_layer.gather_output:
                output = tensor_model_parallel_all_gather(output_parallel)
            else:
                output = output_parallel
            output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
            
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int) -> torch.Tensor:
        base_layer = self.base_layer
        q_proj_shard_size = base_layer.q_proj_shard_size
        kv_proj_shard_size = base_layer.kv_proj_shard_size
        num_kv_head_replicas = base_layer.num_kv_head_replicas

        q_start_idx = q_proj_shard_size * tp_rank
        q_end_idx = q_start_idx + q_proj_shard_size

        kv_shard_id = tp_rank // num_kv_head_replicas
        kv_start_idx = kv_proj_shard_size * kv_shard_id
        kv_end_idx = kv_start_idx + kv_proj_shard_size

        q_size, k_size, _ = base_layer.output_sizes
        B_q_shard = B[q_start_idx:q_end_idx, :]
        B_k_shard = B[q_size + kv_start_idx : q_size + kv_end_idx, :]
        B_v_shard = B[q_size + k_size + kv_start_idx : q_size + k_size + kv_end_idx, :]

        return torch.concat(
            (
                B_q_shard,
                B_k_shard,
                B_v_shard,
            ),
            dim=0,
        )


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: RowParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        self.original_weight = None

    def set_lora_info(self, A_buffer: torch.Tensor | None, B_buffer: torch.Tensor | None, merge: bool | None = None):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer
        output_size = self.base_layer.output_size
        self.output_offset = torch.tensor(
            [
                0,
                output_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )
        if merge is not None:
            self.merge_lora = bool(merge)
            
        if self.merge_lora == True and self.A_buffer is not None and self.B_buffer is not None:
            self._prepare_merge()
            lora_A = A_buffer[0]
            lora_B = B_buffer[0]
            w = self.base_layer.weight.data
            self._add_mm_blockwise_(w, lora_B, lora_A)
            self._finalize_merge_add_()

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.B_buffer,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output

    # def forward(self, input_: torch.Tensor, skip_all_reduce=False):
    #     # duplicate the logic in RowParallelLinear
    #     if self.base_layer.input_is_parallel:
    #         input_parallel = input_
    #     else:
    #         tp_rank = get_tensor_model_parallel_rank()
    #         splitted_input = split_tensor_along_last_dim(
    #             input_, num_partitions=self.base_layer.tp_size
    #         )
    #         input_parallel = splitted_input[tp_rank].contiguous()
    #     output_parallel = self.base_layer.quant_method.apply(
    #         self.base_layer, input_parallel
    #     )

    #     if self.set_lora:
    #         output_parallel = self.apply_lora(output_parallel, input_parallel)

    #     if (
    #         self.base_layer.reduce_results
    #         and self.base_layer.tp_size > 1
    #         and not skip_all_reduce
    #     ):
    #         output_ = tensor_model_parallel_all_reduce(output_parallel)
    #     else:
    #         output_ = output_parallel

    #     if not self.base_layer.skip_bias_add:
    #         output = (
    #             output_ + self.base_layer.bias
    #             if self.base_layer.bias is not None
    #             else output_
    #         )
    #         output_bias = None
    #     else:
    #         output = output_
    #         output_bias = self.base_layer.bias
    #     return output, output_bias
    def forward(self, input_: torch.Tensor, skip_all_reduce=False):
        # 逻辑与 ColumnParallelLinearWithLoRA 的 forward 类似
        if not self.merge_lora: # False 表示非合并 WX+ABX
            # --- Unmerge Path ---
            if self.base_layer.input_is_parallel:
                input_parallel = input_
            else:
                tp_rank = get_tensor_model_parallel_rank()
                splitted_input = split_tensor_along_last_dim(
                    input_, num_partitions=self.base_layer.tp_size
                )
                input_parallel = splitted_input[tp_rank].contiguous()
            
            output_parallel = self.base_layer.quant_method.apply(
                self.base_layer, input_parallel
            )

            if self.set_lora:
                output_parallel = self.apply_lora(output_parallel, input_parallel)
        else: # True 表示合并 (W+AB)X
            # --- Merge Path ---
            if self.base_layer.input_is_parallel:
                input_parallel = input_
            else:
                tp_rank = get_tensor_model_parallel_rank()
                splitted_input = split_tensor_along_last_dim(
                    input_, num_partitions=self.base_layer.tp_size
                )
                input_parallel = splitted_input[tp_rank].contiguous()

            output_parallel = self.base_layer.quant_method.apply(
                self.base_layer, input_parallel
            )

        # --- Common All-Reduce and Bias Logic ---
        if (
            self.base_layer.reduce_results
            and self.base_layer.tp_size > 1
            and not skip_all_reduce
        ):
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        shard_size = self.base_layer.input_size_per_partition
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        A = A[:, start_idx:end_idx].contiguous()
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        return B

'''
get_lora_layer 把这些层替换为带 LoRA 功能的包装层
'''
def get_lora_layer(
    layer: nn.Module, lora_backend: BaseLoRABackend
) -> BaseLayerWithLoRA:
    print(f"Replacing layer {type(layer)} with LoRA layer.")
    supported_layer_types = {
        # the order matters
        VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        QKVParallelLinear: QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(layer, lora_backend)
            return ret
    raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")

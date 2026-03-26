# File: sglang+dLoRA/python/sglang/srt/instances/lora_init.py

"""
LoRA weight size calculation for InstanceManager initialization.
This module provides standalone LoRA size computation without requiring
active scheduler communication.
"""

import logging
import os
from typing import Dict, Tuple

import torch

from sglang.srt.instances.lora_config_paths import LORA_PATH, NUM_LORAS
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.utils.hf_transformers_utils import AutoConfig

logger = logging.getLogger(__name__)


class LoRAWeightCalculator:
    """
    Standalone calculator for LoRA adapter weight sizes.
    Used by InstanceManager during initialization.
    """

    def __init__(
        self,
        base_model_path: str,
        max_lora_rank: int = 64,
        dtype: torch.dtype = torch.float16,
        tp_size: int = 1,
    ):
        """
        Initialize LoRA weight calculator. 

        Args:
            base_model_path: Path to base model (for config)
            max_lora_rank: Maximum LoRA rank supported
            dtype: Data type for weights (default: float16)——'''[2025-12-10 02:59:11] Model config dtype: torch.float16'''
            tp_size: Tensor parallelism size
        """
        self.base_model_path = base_model_path
        self.max_lora_rank = max_lora_rank
        self.dtype = dtype
        self.tp_size = tp_size

        # Load base model config
        self.base_config = AutoConfig.from_pretrained(
            base_model_path, trust_remote_code=True
        )
        self.dtype_size = torch.tensor([], dtype=dtype).element_size()

        logger.info(
            f"LoRAWeightCalculator initialized:  "
            f"base={base_model_path}, max_rank={max_lora_rank}, "
            f"dtype={dtype}, tp_size={tp_size}"
        )

    def _get_hidden_dims(self, module_name: str) -> Tuple[int, int]:
        """
        Get input and output dimensions for a module. 

        Args:
            module_name: Name of the module (e.g., "qkv_proj", "gate_up_proj")

        Returns:
            Tuple of (input_dim, output_dim)
        """
        config = self.base_config
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        if module_name == "qkv_proj":
            input_dim = config.hidden_size
            output_dim = head_dim * (
                config.num_attention_heads + config.num_key_value_heads * 2
            )
        elif module_name == "o_proj":
            input_dim = head_dim * config.num_attention_heads
            output_dim = config.hidden_size
        elif module_name == "gate_up_proj":
            input_dim = config.hidden_size
            output_dim = config.intermediate_size * 2
        elif module_name == "down_proj":
            input_dim = config.intermediate_size
            output_dim = config.hidden_size
        else:
            raise NotImplementedError(f"Module {module_name} not supported")

        # Adjust for tensor parallelism
        if self.tp_size > 1:
            if module_name in ["o_proj", "down_proj"]:
                input_dim = input_dim // self.tp_size
            else:
                output_dim = output_dim // self.tp_size

        return input_dim, output_dim

    def _get_stacked_multiply(self, module_name: str) -> int:
        """Get stacking multiplier for module."""
        stacked_rank = {
            "qkv_proj": 3,
            "gate_up_proj": 2,
        }
        return stacked_rank.get(module_name, 1)

    def calculate_lora_size(self, lora_path: str) -> int:
        """
        Calculate the memory size required for a single LoRA adapter.

        Args:
            lora_path:  Path to the LoRA adapter

        Returns:
            Size in bytes
        """
        try:
            # Load LoRA config
            lora_config = LoRAConfig(path=lora_path)
            lora_rank = lora_config.r
            target_modules = lora_config.target_modules

            # Normalize target modules
            from sglang.srt.lora.utils import get_normalized_target_modules

            normalized_modules = get_normalized_target_modules(target_modules)

            num_layers = self.base_config.num_hidden_layers
            total_size = 0

            logger.debug(
                f"Calculating size for {lora_path}:  "
                f"rank={lora_rank}, modules={normalized_modules}, layers={num_layers}"
            )

            # Calculate size for each module and layer
            for module_name in normalized_modules:
                input_dim, output_dim = self._get_hidden_dims(module_name)
                stacked_mult = self._get_stacked_multiply(module_name)

                # A matrix:  (rank * stacked_mult, input_dim)
                A_size = lora_rank * stacked_mult * input_dim * self.dtype_size

                # B matrix: (output_dim, rank)
                B_size = output_dim * lora_rank * self.dtype_size

                # Total for this module across all layers
                module_size = (A_size + B_size) * num_layers
                total_size += module_size

                logger.debug(
                    f"  {module_name}: A={A_size}B, B={B_size}B, "
                    f"layers={num_layers}, total={module_size}B"
                )

            logger.info(
                f"LoRA size for {os.path.basename(lora_path)}: "
                f"{total_size / (1024**2):.2f} MB"
            )
            return total_size

        except Exception as e:
            logger.error(f"Failed to calculate LoRA size for {lora_path}: {e}")
            # Return a conservative estimate (e.g., 100MB)
            return 100 * 1024 * 1024

    def calculate_all_lora_sizes(
        self, pcie_bandwidth: float = 32 * (1 << 30)
    ) -> Dict[int, Tuple[int, float]]:
        """
        Calculate sizes and load costs for all configured LoRAs.

        Args:
            pcie_bandwidth: PCIe bandwidth in bytes/sec (default: 32 GB/s)

        Returns:
            Dictionary mapping model_id -> (size_bytes, load_time_seconds)
        """
        lora_sizes = {}

        for i in range(NUM_LORAS):
            lora_name = f"lora{i}"
            if lora_name not in LORA_PATH:
                logger. warning(f"LoRA {lora_name} not found in LORA_PATH, skipping")
                continue

            lora_path = LORA_PATH[lora_name]
            size_bytes = self.calculate_lora_size(lora_path)
            load_time = size_bytes / pcie_bandwidth

            lora_sizes[i] = (size_bytes, load_time)

            logger.info(
                f"Model {i} ({lora_name}): "
                f"size={size_bytes / (1024**2):.2f} MB, "
                f"load_time={load_time * 1000:.2f} ms"
            )

        return lora_sizes


def get_lora_sizes_for_instance_manager(
    max_lora_rank: int = 64,
    dtype: torch.dtype = torch.float16,
    tp_size: int = 1,
    pcie_bandwidth: float = 32 * (1 << 30),
) -> Dict[int, Tuple[int, float]]:
    """
    Convenience function to get LoRA sizes for InstanceManager.

    Args:
        max_lora_rank: Maximum LoRA rank
        dtype: Weight data type
        tp_size:  Tensor parallelism size
        pcie_bandwidth: PCIe bandwidth in bytes/sec

    Returns: 
        Dictionary:  model_id -> (size_bytes, load_time_seconds)
    """
    base_path = LORA_PATH["base"]
    calculator = LoRAWeightCalculator(
        base_model_path=base_path,
        max_lora_rank=max_lora_rank,
        dtype=dtype,
        tp_size=tp_size,
    )
    return calculator.calculate_all_lora_sizes(pcie_bandwidth=pcie_bandwidth)
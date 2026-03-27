# File: sglang+dLoRA/python/sglang/srt/instances/lora_config_paths.py

"""
Centralized LoRA configuration for multi-instance serving.
This file should be the single source of truth for LoRA paths across all components.
"""

from typing import Dict

# Number of LoRA adapters
NUM_LORAS = 16

# LoRA model paths configuration
LORA_PATH:  Dict[str, str] = {
    "base": "/workspace/models/Llama-2-7b-hf",
    "lora0": "/workspace/models/adapters/8/Llama2-7B-LoRA-Adapter",
    "lora1": "/workspace/models/adapters/8/llama2-7b-lora-genwiki",
    "lora2": "/workspace/models/adapters/8/llama2-7b-lora-rebel",
    "lora3": "/workspace/models/adapters/8/MUFFIN-Llama2-lora-7B",
    "lora4": "/workspace/models/adapters/16/llama-2-7b-hf-lora-alpaca-json",
    "lora5": "/workspace/models/adapters/16/llama-2-7b-LORA-data-analyst",
    "lora6": "/workspace/models/adapters/16/llama-2-7b-lora-v1",
    "lora7": "/workspace/models/adapters/16/llama2-7B-init-dolly-lora",
    "lora8": "/workspace/models/adapters/32/Final_llama2-7B-lora_r_32",
    "lora9": "/workspace/models/adapters/32/llama-2-7b-sft-lora",
    "lora10": "/workspace/models/adapters/32/llama2-7b-recipe-lora",
    "lora11": "/workspace/models/adapters/32/ola_llama2_7B_lora1",
    "lora12": "/workspace/models/adapters/64/azma-llama2-7b-hf-lora-adapter",
    "lora13": "/workspace/models/adapters/64/llama-2-7b-chat-lora-adaptor",
    "lora14": "/workspace/models/adapters/64/llama2-7b-airos-lora",
    "lora15": "/workspace/models/adapters/64/llama2-stable-7b-lora",
}


def get_lora_path(lora_name: str) -> str:
    """
    Get the file path for a specific LoRA adapter.
    
    Args:
        lora_name: Name of the LoRA adapter (e.g., "lora0", "base")
    
    Returns:
        Path to the LoRA adapter
    
    Raises:
        KeyError: If lora_name is not found
    """
    if lora_name not in LORA_PATH:
        raise KeyError(
            f"LoRA '{lora_name}' not found. Available: {list(LORA_PATH.keys())}"
        )
    return LORA_PATH[lora_name]


def get_all_lora_names() -> list[str]:
    """Get all configured LoRA adapter names (excluding base)."""
    return [name for name in LORA_PATH.keys() if name != "base"]


def get_lora_id(lora_name: str) -> int:
    """
    Convert LoRA name to numeric ID.
    
    Args:
        lora_name: Name like "lora0", "lora1", etc.
    
    Returns:
        Numeric ID (0, 1, 2, .. .) or 0 for base/unknown
    """
    try:
        return int(lora_name.replace("lora", "")) if lora_name. startswith("lora") else 0
    except (ValueError, AttributeError):
        return 0
    

_LORA_RANK_CACHE: Dict[str, int] = {}


def get_lora_rank(lora_name: str) -> int:
    """
    Return the LoRA rank (r) for a named adapter.

    On the first call for a given lora_name, reads adapter_config.json
    from the adapter directory (same file / key that LoRAConfig uses).
    Results are cached in module-level _LORA_RANK_CACHE for all subsequent calls.

    Args:
        lora_name: Adapter name as in LORA_PATH, e.g. "lora0", "lora1".

    Returns:
        int: rank r (e.g. 8, 16, 32, 64).  Returns 0 on any error.

    Example:
        >>> get_lora_rank("lora0")
        64
    """
    if lora_name in _LORA_RANK_CACHE:
        return _LORA_RANK_CACHE[lora_name]

    if lora_name not in LORA_PATH:
        return 0

    lora_path = LORA_PATH[lora_name]
    try:
        # Import here to avoid circular imports at module level
        from sglang.srt.lora.lora_config import LoRAConfig
        cfg = LoRAConfig(path=lora_path)
        rank = int(cfg.r)
    except Exception:
        rank = 0

    _LORA_RANK_CACHE[lora_name] = rank
    return rank


def get_model_lora_ranks() -> Dict[str, int]:
    """
    Return a mapping of model_id (int) → rank (int) for all adapters.

    Reads adapter_config.json for each lora0…lora{NUM_LORAS-1} in LORA_PATH.
    This is the canonical way to obtain ranks without any CLI argument.

    Returns:
        Dict[int, int]: {0: 64, 1: 16, 2: 64, 3: 64}  (example)
    """
    result: Dict[int, int] = {}
    for i in range(NUM_LORAS):
        lora_name = f"lora{i}"
        result[i] = get_lora_rank(lora_name)
    return result
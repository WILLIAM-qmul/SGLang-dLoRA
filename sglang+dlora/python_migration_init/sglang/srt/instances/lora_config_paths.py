# File: sglang+dLoRA/python/sglang/srt/instances/lora_config_paths.py

"""
Centralized LoRA configuration for multi-instance serving.
This file should be the single source of truth for LoRA paths across all components.
"""

from typing import Dict

# Number of LoRA adapters
NUM_LORAS = 4

# LoRA model paths configuration
LORA_PATH:  Dict[str, str] = {
    "base": "/workspace/models/Llama-2-7b-hf",
    "lora0": "/workspace/models/llama-2-7b-chat-lora-adaptor",
    "lora1": "/workspace/models/llama-2-7b-LORA-data-analyst",
    "lora2": "/workspace/models/llama2-stable-7b-lora",
    "lora3": "/workspace/models/llava-llama-2-7b-chat-lightning-lora-preview",
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
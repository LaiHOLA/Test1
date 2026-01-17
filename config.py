"""SlideFSDP Configuration Module.

Defines configuration dataclass for SlideFSDP with configurable
parameter transfer and gradient synchronization strategies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class SlideFSDPConfig:
    """Configuration for SlideFSDP multi-GPU training.
    
    Attributes:
        dtype: Parameter dtype for GPU computation (bf16/fp16).
        device: Target CUDA device (auto-detected from LOCAL_RANK if None).
        
        param_transfer_mode: Strategy for CPU->GPU parameter transfer.
            - "full": Each GPU loads full layer params via H2D (simpler, higher bandwidth usage)
            - "shard": H2D shard only + GPU AllGather (saves CPU->GPU bandwidth for large models)
        
        grad_sync_mode: Strategy for gradient synchronization.
            - "gpu_reduce": GPU ReduceScatter -> D2H shard (saves D2H bandwidth)
            - "cpu_reduce": D2H full -> CPU AllReduce via Gloo (saves GPU memory)
        
        lr: Learning rate for CPU Adam optimizer.
        weight_decay: Weight decay coefficient.
        adam_betas: Adam beta1 and beta2 coefficients.
        adam_eps: Adam epsilon for numerical stability.
        
        prefetch_layers: Number of layers to prefetch ahead.
        max_workers: Thread pool size for async operations.
        enable_activation_checkpoint: Whether to use activation checkpointing.
        
        model_path: Path to HuggingFace model.
        max_seq_length: Maximum sequence length for training.
        train_batch_size: Per-GPU batch size.
        num_epochs: Number of training epochs.
        log_every_n_steps: Logging frequency.
        seed: Random seed.
        use_liger_kernel: Whether to use Liger kernel optimizations.
        attention_implementation: Attention implementation (flash_attention_2, sdpa, eager).
    """
    
    # Basic configuration
    dtype: torch.dtype = torch.bfloat16
    device: Optional[torch.device] = None  # Auto-detect from LOCAL_RANK
    
    # Parameter transfer strategy
    param_transfer_mode: str = "full"  # "full" | "shard"
    
    # Gradient synchronization strategy
    grad_sync_mode: str = "gpu_reduce"  # "gpu_reduce" | "cpu_reduce"
    
    # Optimizer configuration
    lr: float = 1e-5
    weight_decay: float = 0.01
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    bias_correction: bool = True
    
    # Runtime configuration
    prefetch_layers: int = 1
    max_workers: int = 1
    enable_activation_checkpoint: bool = True
    gpu_buffer_pool_size: int = 2  # Number of GPU buffer units (sliding window)
    
    # Model configuration
    model_path: str = ""
    max_seq_length: int = 1024
    train_batch_size: int = 4
    num_epochs: int = 1
    log_every_n_steps: int = 1
    seed: int = 42
    use_liger_kernel: bool = True
    attention_implementation: str = "flash_attention_2"
    
    # Activation checkpoint offload
    ac_offload_mode: str = "cpu"  # "cpu" | "nvme" | "none"
    ac_offload_dir: str = "/tmp/ac_offload"
    
    # NVMe offload for optimizer states (optional)
    optimizer_nvme_offload_fraction: float = 0.0
    optimizer_offload_dir: str = "/tmp/opt_offload"
    
    # Gradient clipping
    clip_grad_norm: float = 1.0
    
    def __post_init__(self):
        """Validate configuration and auto-detect device."""
        # Validate param_transfer_mode
        if self.param_transfer_mode not in ("full", "shard"):
            raise ValueError(
                f"param_transfer_mode must be 'full' or 'shard', got {self.param_transfer_mode}"
            )
        
        # Validate grad_sync_mode
        if self.grad_sync_mode not in ("gpu_reduce", "cpu_reduce"):
            raise ValueError(
                f"grad_sync_mode must be 'gpu_reduce' or 'cpu_reduce', got {self.grad_sync_mode}"
            )
        
        # Validate dtype
        if self.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"dtype must be torch.bfloat16 or torch.float16, got {self.dtype}"
            )
        
        # Auto-detect device from LOCAL_RANK
        if self.device is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device("cuda", local_rank)
        
        # Validate activation checkpoint mode
        if self.ac_offload_mode not in ("cpu", "nvme", "none"):
            raise ValueError(
                f"ac_offload_mode must be 'cpu', 'nvme', or 'none', got {self.ac_offload_mode}"
            )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "SlideFSDPConfig":
        """Create config from dictionary."""
        # Handle dtype conversion from string
        if "dtype" in config_dict and isinstance(config_dict["dtype"], str):
            dtype_map = {
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
            }
            config_dict["dtype"] = dtype_map.get(
                config_dict["dtype"].lower(), torch.bfloat16
            )
        
        # Handle device conversion from string
        if "device" in config_dict and isinstance(config_dict["device"], str):
            config_dict["device"] = torch.device(config_dict["device"])
        
        # Handle adam_betas conversion from list
        if "adam_betas" in config_dict and isinstance(config_dict["adam_betas"], list):
            config_dict["adam_betas"] = tuple(config_dict["adam_betas"])
        
        # Filter only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, torch.dtype):
                value = str(value).replace("torch.", "")
            elif isinstance(value, torch.device):
                value = str(value)
            result[field_name] = value
        return result


"""SlideFSDP Multi-GPU Training Script.

Usage:
    # Single node, multiple GPUs
    torchrun --nproc_per_node=4 train.py --model-path /path/to/model
    
    # With config file
    torchrun --nproc_per_node=4 train.py --config train_config.json
    
    # Override specific options
    torchrun --nproc_per_node=4 train.py \\
        --model-path /path/to/model \\
        --param-transfer-mode shard \\
        --grad-sync-mode cpu_reduce
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from config import SlideFSDPConfig
from slide_fsdp import SlideFSDP
from utils.datasets import DummyDataset
from utils.metric import calculate_flops_per_batch

# Try to import HuggingFace transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Try to import Liger kernel for efficient model loading
try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False


def init_distributed() -> tuple[int, int, torch.device]:
    """Initialize distributed training.
    
    Returns:
        Tuple of (rank, world_size, device).
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    
    # Enforce Main Thread Affinity to Control Cores
    # Ensures Data Transfer loops / GPU kernel launches stay on reserved cores
    # OMP threads (CPU Adam) will be pinned to Update Cores via OMP_PLACES (env)
    control_cores_str = os.environ.get("NUMA_CONTROL_CORES", "")
    if control_cores_str:
        try:
            control_cores = [int(x) for x in control_cores_str.split(",") if x.strip()]
            if control_cores:
                # os.sched_setaffinity sets affinity for the calling thread (in Python on Linux)
                os.sched_setaffinity(0, control_cores)
                if rank % 8 == 0 or rank == 0:  # Avoid excessive logging
                    print(f"[Rank {rank}] Main Thread pinned to Control Cores: {control_cores}")
        except Exception as e:
            print(f"[Rank {rank}] Failed to pin main thread: {e}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    return rank, world_size, device


def set_seed(seed: int, rank: int = 0) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Base random seed.
        rank: Process rank for unique seeding.
    """
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def load_model(config: SlideFSDPConfig, device: torch.device):
    """Load HuggingFace model.
    
    Args:
        config: Training configuration.
        device: Target device.
        
    Returns:
        Loaded model.
    """
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers library is required")
    
    load_kwargs = {
        "attn_implementation": config.attention_implementation,
        "torch_dtype": config.dtype,
        "device_map": "cpu",  # Load to CPU first
    }
    
    if config.use_liger_kernel and HAS_LIGER:
        model = AutoLigerKernelForCausalLM.from_pretrained(
            config.model_path, **load_kwargs
        )
    else:
        print("Loading model without liger kernel")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path, **load_kwargs
        )
    
    return model


def create_dataloader(
    config: SlideFSDPConfig,
    tokenizer,
    shuffle: bool = True,
) -> DataLoader:
    """Create distributed data loader.
    
    Args:
        config: Training configuration.
        tokenizer: Tokenizer for dataset.
        shuffle: Whether to shuffle data.
        
    Returns:
        DataLoader instance.
    """
    dataset = DummyDataset(
        size=1024 * config.num_epochs,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )
    
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    
    return DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
    )


def train_epoch(
    model: SlideFSDP,
    dataloader: DataLoader,
    config: SlideFSDPConfig,
    device: torch.device,
    epoch: int,
    total_flops: float,
    world_size: int,
) -> float:
    """Train for one epoch.
    
    Args:
        model: SlideFSDP model.
        dataloader: Training data loader.
        config: Training configuration.
        device: Target device.
        epoch: Current epoch number.
        total_flops: FLOPs per batch.
        world_size: World size.
    Returns:
        Average loss for the epoch.
    """
    model.train()
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    log_every = max(1, config.log_every_n_steps)
    tokens_per_batch = config.train_batch_size * config.max_seq_length * world_size
    
    start_time = time.perf_counter()
    running_loss = 0.0
    total_loss = 0.0
    num_steps = 0
    
    for step, batch in enumerate(dataloader, start=1):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward + backward (handled by SlideFSDP)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        
        loss = outputs.loss
        running_loss += loss.item()
        total_loss += loss.item()
        num_steps += 1
        
        # Logging
        if step % log_every == 0 and rank == 0:
            elapsed = time.perf_counter() - start_time
            avg_loss = running_loss / log_every
            tokens_per_sec = tokens_per_batch * log_every / elapsed
            tflops = total_flops * log_every / (elapsed * 1e12)
            
            print(
                f"Epoch {epoch} | Step {step}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | "
                f"{tokens_per_sec:.1f} tokens/s | "
                f"{tflops:.2f} TFLOPS"
            )
            
            running_loss = 0.0
            start_time = time.perf_counter()
    
    return total_loss / max(num_steps, 1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SlideFSDP Multi-GPU Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file",
    )
    
    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to HuggingFace model",
    )
    
    # Transfer strategies
    parser.add_argument(
        "--param-transfer-mode",
        type=str,
        choices=["full", "shard"],
        default=None,
        help="Parameter transfer strategy: full H2D or shard + AllGather",
    )
    parser.add_argument(
        "--grad-sync-mode",
        type=str,
        choices=["gpu_reduce", "cpu_reduce"],
        default=None,
        help="Gradient sync strategy: GPU reduce or CPU reduce",
    )
    
    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-GPU batch size",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save model after training",
    )
    
    # Debug
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> SlideFSDPConfig:
    """Load configuration from file and/or command line.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        SlideFSDPConfig instance.
    """
    # Start with default config
    config_dict = {}
    
    # Load from JSON file if provided
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
    
    # Override with command line arguments
    if args.model_path is not None:
        config_dict["model_path"] = args.model_path
    if args.param_transfer_mode is not None:
        config_dict["param_transfer_mode"] = args.param_transfer_mode
    if args.grad_sync_mode is not None:
        config_dict["grad_sync_mode"] = args.grad_sync_mode
    if args.batch_size is not None:
        config_dict["train_batch_size"] = args.batch_size
    if args.seq_length is not None:
        config_dict["max_seq_length"] = args.seq_length
    if args.num_epochs is not None:
        config_dict["num_epochs"] = args.num_epochs
    if args.lr is not None:
        config_dict["lr"] = args.lr
    if args.seed is not None:
        config_dict["seed"] = args.seed
    
    return SlideFSDPConfig.from_dict(config_dict)


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    # Initialize distributed
    rank, world_size, device = init_distributed()
    
    # Load config
    config = load_config(args)
    config.device = device  # Override with actual device
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("SlideFSDP Training")
        print(f"{'='*60}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print(f"Model: {config.model_path}")
        print(f"Param transfer mode: {config.param_transfer_mode}")
        print(f"Grad sync mode: {config.grad_sync_mode}")
        print(f"Batch size (per GPU): {config.train_batch_size}")
        print(f"Sequence length: {config.max_seq_length}")
        print(f"{'='*60}\n")
    
    # Set random seed
    set_seed(config.seed, rank)
    
    # Load tokenizer
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers library required")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    if rank == 0:
        print("Loading model...")
    base_model = load_model(config, device)
    
    # Calculate FLOPs
    total_flops = calculate_flops_per_batch(
        base_model,
        config.train_batch_size,
        config.max_seq_length,
    ) * world_size
    
    # Wrap with SlideFSDP
    if rank == 0:
        print("Wrapping model with SlideFSDP...")
    model = SlideFSDP(base_model, config)
    
    # Create data loader
    train_loader = create_dataloader(config, tokenizer, shuffle=True)
    
    # Training loop
    if rank == 0:
        print(f"\nStarting training for {config.num_epochs} epochs...")
        print(f"Total steps: {len(train_loader) * config.num_epochs}")
    
    dist.barrier()
    
    for epoch in range(1, config.num_epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        
        epoch_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            config=config,
            device=device,
            epoch=epoch,
            total_flops=total_flops,
            world_size=world_size,
        )
        
        if rank == 0:
            print(f"\nEpoch {epoch} completed. Average loss: {epoch_loss:.4f}\n")
        
        dist.barrier()
    
    # Save model
    if args.save_model and rank == 0:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(output_path))
    
    # Cleanup
    dist.barrier()
    if rank == 0:
        print("Training completed!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()


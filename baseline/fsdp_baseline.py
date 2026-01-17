"""FSDP baseline fine-tuning script.

Usage (multi-GPU example):
    torchrun --nproc_per_node=8 fsdp_baseline.py --config-path ./fsdp_config.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from functools import partial
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from utils.metric import calculate_flops_per_batch
# from utils.log_mem import log_memory_stats
from utils.datasets import DummyDataset

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - transformers not installed
    raise RuntimeError("This script requires Hugging Face transformers.") from exc


@dataclass
class TrainConfig:
    model_path: str = "/home/scc/models/Llama-3.1-8B-Instruct/"
    output_dir: str = "./fsdp-output"
    max_seq_length: int = 1024
    train_batch_size: int = 4
    eval_batch_size: int = 4
    num_epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    log_every_n_steps: int = 1
    eval_steps: int = 0
    save_final: bool = False
    seed: int = 42
    use_liger_kernel: bool = True
    dtype: str = "bf16"  # choices: bf16 / fp16 / fp32
    attention_implementation: str = "flash_attention_2"

    # FSDP specific
    sharding_strategy: str = "full_shard"  # full_shard, shard_grad_op, no_shard, hybrid_shard
    cpu_offload: bool = True
    backward_prefetch: str = "backward_pre"  # backward_pre, backward_post, none
    forward_prefetch: bool = True
    wrap_policy: str = "transformer"  # transformer, size, none
    size_min_params: int = 30_000_000
    mixed_precision_reduce_dtype: Optional[str] = None
    activation_checkpointing: bool = False
    clip_grad_norm: float = 1.0
    limit_all_gathers: bool = True

    # evaluation
    do_eval: bool = False

def init_distributed() -> Tuple[int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, world_size, device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str.lower() == "bf16":
        return torch.bfloat16
    if dtype_str.lower() == "fp16":
        return torch.float16
    if dtype_str.lower() == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def resolve_backward_prefetch(value: str) -> Optional[BackwardPrefetch]:
    mapping = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
        "none": None,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"Invalid backward_prefetch: {value}") from exc


def resolve_sharding_strategy(value: str) -> ShardingStrategy:
    mapping = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"Unknown sharding strategy: {value}") from exc


def build_auto_wrap_policy(model: nn.Module, config: TrainConfig):
    if config.wrap_policy == "none":
        return None

    if config.wrap_policy == "transformer":
        layer_cls = []
        model_type = getattr(model.config, "model_type", "")
        try:
            if model_type in {"llama", "mistral"}:
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer

                layer_cls.append(LlamaDecoderLayer)
            elif model_type in {"qwen"}:
                from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

                layer_cls.append(Qwen2DecoderLayer)  # type: ignore[name-defined]
        except ImportError:
            layer_cls = []

        if layer_cls:
            return partial(transformer_auto_wrap_policy, transformer_layer_cls=tuple(layer_cls))

    return size_based_auto_wrap_policy(min_num_params=config.size_min_params)


def get_mixed_precision(config: TrainConfig) -> Optional[MixedPrecision]:
    param_dtype = resolve_dtype(config.dtype)
    reduce_dtype = (
        resolve_dtype(config.mixed_precision_reduce_dtype)
        if config.mixed_precision_reduce_dtype
        else param_dtype
    )

    if param_dtype == torch.float32:
        return None

    return MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=torch.float32)


def get_model(config: TrainConfig, device: torch.device) -> nn.Module:
    load_kwargs = {
        "attn_implementation": config.attention_implementation,
        "torch_dtype": resolve_dtype(config.dtype),
        "device_map": "cpu",
    }

    if config.use_liger_kernel:
        try:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM

            base_model = AutoLigerKernelForCausalLM.from_pretrained(config.model_path, **load_kwargs)
        except ImportError:
            base_model = AutoModelForCausalLM.from_pretrained(config.model_path, **load_kwargs)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(config.model_path, **load_kwargs)

    return base_model.to("cpu")


def setup_fsdp(model: nn.Module, config: TrainConfig, device: torch.device) -> FSDP:
    auto_wrap_policy = build_auto_wrap_policy(model, config)
    sharding_strategy = resolve_sharding_strategy(config.sharding_strategy)
    backward_prefetch = resolve_backward_prefetch(config.backward_prefetch)
    mixed_precision = get_mixed_precision(config)
    cpu_offload = CPUOffload(offload_params=config.cpu_offload)

    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        cpu_offload=cpu_offload,
        device_id=device,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        forward_prefetch=config.forward_prefetch,
        backward_prefetch=backward_prefetch,
        limit_all_gathers=config.limit_all_gathers,
        use_orig_params=True,
    )

    return fsdp_model


def prepare_optimizer(model: FSDP, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train_one_epoch(
    model: FSDP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
    epoch: int,
    total_flops: float,
) -> None:
    model.train()
    rank = dist.get_rank()
    log_every = max(1, config.log_every_n_steps)
    tokens_per_batch = config.train_batch_size * config.max_seq_length

    start = time.perf_counter()
    running_loss = 0.0

    for step, batch in enumerate(dataloader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        if config.clip_grad_norm > 0:
            model.clip_grad_norm_(config.clip_grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item()

        if step % log_every == 0 and rank == 0:
            elapsed = time.perf_counter() - start
            avg_loss = running_loss / log_every
            tps = tokens_per_batch * log_every / elapsed
            tflops = total_flops * log_every / (elapsed * 1e12)
            print(
                f"Epoch {epoch} Step {step}/{len(dataloader)} | "
                f"Loss {avg_loss:.4f} | {tps:.1f} tokens/s | {tflops:.2f} TFLOPS"
            )
            running_loss = 0.0
            start = time.perf_counter()


def evaluate(model: FSDP, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1
    if steps == 0:
        return 0.0
    return total_loss / steps


def save_model(model: FSDP, tokenizer, output_dir: str) -> None:
    if dist.get_rank() != 0:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        state_dict = model.state_dict()
    torch.save(state_dict, output_path / "pytorch_model.bin")
    tokenizer.save_pretrained(output_path)


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool = True) -> DataLoader:
    sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FSDP baseline training script")
    parser.add_argument("--config-path", type=str, default=None, help="Optional JSON config file")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--save-config", action="store_true", help="Dump resolved config to output dir")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig()
    if args.config_path is not None:
        with open(args.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    return config


def dump_config(config: TrainConfig) -> None:
    rank = dist.get_rank()
    if rank != 0:
        return
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


def main() -> None:
    args = parse_args()
    config = load_config(args)

    rank, world_size, device = init_distributed()
    if rank == 0:
        print(f"Starting FSDP training with world_size={world_size}")
    set_seed(config.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = DummyDataset(size=128, tokenizer=tokenizer, max_length=config.max_seq_length)
    train_loader = create_dataloader(train_dataset, config.train_batch_size, shuffle=True)

    eval_loader = None
    if config.do_eval:
        eval_dataset = DummyDataset(size=120, tokenizer=tokenizer, max_length=config.max_seq_length)
        eval_loader = create_dataloader(eval_dataset, config.eval_batch_size, shuffle=False, drop_last=False)

    base_model = get_model(config, device)
    total_flops = calculate_flops_per_batch(base_model, config.train_batch_size, config.max_seq_length)

    fsdp_model = setup_fsdp(base_model, config, device)
    optimizer = prepare_optimizer(fsdp_model, config)

    if args.save_config:
        dump_config(config)

    for epoch in range(1, config.num_epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(fsdp_model, train_loader, optimizer, config, device, epoch, total_flops)

        if config.do_eval and eval_loader is not None:
            eval_loss = evaluate(fsdp_model, eval_loader, device)
            if dist.get_rank() == 0:
                print(f"Eval loss after epoch {epoch}: {eval_loss:.4f}")

    if config.save_final:
        save_model(fsdp_model, tokenizer, config.output_dir)

    dist.barrier()
    if rank == 0:
        print("Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

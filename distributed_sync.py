"""Distributed Synchronization Strategies for SlideFSDP.

Provides configurable strategies for:
1. Parameter transfer (CPU->GPU): full H2D vs shard H2D + AllGather
2. Gradient synchronization: GPU ReduceScatter + D2H vs D2H + CPU AllReduce
3. Single-optimizer mode: Reduce grads to rank 0, update, broadcast params
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from config import SlideFSDPConfig
    from layer_state import LayerRuntimeState


class DistributedSync:
    """Configurable distributed synchronization for parameters and gradients.
    
    Supports two parameter transfer modes:
    - "full": Each GPU loads full layer params via H2D
    - "shard": H2D shard only + GPU AllGather
    
    Supports two gradient sync modes:
    - "gpu_reduce": GPU ReduceScatter -> D2H shard (saves D2H bandwidth)
    - "cpu_reduce": D2H full -> CPU AllReduce via Gloo (saves GPU memory)
    """
    
    def __init__(self, config: "SlideFSDPConfig"):
        """Initialize distributed synchronization.
        
        Args:
            config: SlideFSDP configuration.
        """
        self.config = config
        self.param_mode = config.param_transfer_mode
        self.grad_mode = config.grad_sync_mode
        self.device = config.device
        self.dtype = config.dtype
        
        # Get distributed info
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.is_distributed = self.world_size > 1
        else:
            self.world_size = 1
            self.rank = 0
            self.is_distributed = False
        
        # Initialize CPU communication group (Gloo backend) for cpu_reduce mode
        self.cpu_group = None
        if self.is_distributed and self.grad_mode == "cpu_reduce":
            # Create a new process group with Gloo backend for CPU tensor communication
            self.cpu_group = dist.new_group(backend="gloo")
        
        # GPU group for NCCL operations
        self.gpu_group = None  # Use default group
        
        # Pre-allocated shard buffers for shard mode
        self._gpu_shard_buffer: Optional[torch.Tensor] = None
        self._shard_size: int = 0
    
    def _ensure_shard_buffer(self, total_size: int) -> None:
        """Ensure shard buffer is allocated with correct size.
        
        Args:
            total_size: Total parameter/gradient size.
        """
        shard_size = (total_size + self.world_size - 1) // self.world_size
        
        if self._gpu_shard_buffer is None or self._shard_size != shard_size:
            self._shard_size = shard_size
            self._gpu_shard_buffer = torch.empty(
                shard_size,
                dtype=self.dtype,
                device=self.device,
            )
    
    def _get_shard_range(self, total_size: int) -> tuple[int, int]:
        """Get start and end indices for this rank's shard.
        
        Args:
            total_size: Total tensor size.
            
        Returns:
            Tuple of (start_idx, end_idx) for this rank's shard.
        """
        shard_size = (total_size + self.world_size - 1) // self.world_size
        start = self.rank * shard_size
        end = min(start + shard_size, total_size)
        return start, end
    
    def gather_params_to_gpu(
        self,
        cpu_params: torch.Tensor,
        gpu_buffer: torch.Tensor,
        layer_state: "LayerRuntimeState",
    ) -> None:
        """Transfer parameters from CPU to GPU with configurable strategy.
        
        Args:
            cpu_params: CPU parameter tensor (bf16, pinned memory).
            gpu_buffer: Target GPU buffer for parameters.
            layer_state: Layer runtime state for context.
        """
        total_size = cpu_params.numel()
        
        if not self.is_distributed or self.param_mode == "full":
            # Strategy A: Direct full H2D transfer
            # Each GPU loads complete layer parameters
            gpu_buffer[:total_size].copy_(cpu_params[:total_size], non_blocking=True)
        
        else:
            # Strategy B: H2D shard + AllGather
            # Each GPU only transfers its shard, then AllGather on GPU
            self._ensure_shard_buffer(total_size)
            
            start, end = self._get_shard_range(total_size)
            shard_numel = end - start
            
            # H2D transfer only this rank's shard
            self._gpu_shard_buffer[:shard_numel].copy_(
                cpu_params[start:end],
                non_blocking=True,
            )
            
            # Synchronize before AllGather
            torch.cuda.current_stream().synchronize()
            
            # AllGather to reconstruct full tensor on GPU
            # Create list of tensors for all_gather
            gather_list = [
                torch.empty(
                    self._shard_size if i < self.world_size - 1 
                    else total_size - i * self._shard_size,
                    dtype=self.dtype,
                    device=self.device,
                )
                for i in range(self.world_size)
            ]
            
            dist.all_gather(
                gather_list,
                self._gpu_shard_buffer[:shard_numel],
                group=self.gpu_group,
            )
            
            # Copy gathered shards to output buffer
            offset = 0
            for i, shard in enumerate(gather_list):
                shard_len = shard.numel()
                gpu_buffer[offset:offset + shard_len].copy_(shard)
                offset += shard_len
    
    def sync_gradients_to_cpu(
        self,
        gpu_grads: torch.Tensor,
        cpu_buffer: torch.Tensor,
        layer_state: "LayerRuntimeState",
    ) -> None:
        """Synchronize gradients from GPU to CPU with configurable strategy.
        
        Args:
            gpu_grads: GPU gradient tensor.
            cpu_buffer: Target CPU buffer for gradients (pinned memory).
            layer_state: Layer runtime state for context.
        """
        total_size = gpu_grads.numel()
        
        if not self.is_distributed:
            # Single GPU: direct D2H copy
            cpu_buffer[:total_size].copy_(gpu_grads[:total_size], non_blocking=True)
            torch.cuda.current_stream().synchronize()
            return
        
        if self.grad_mode == "gpu_reduce":
            # Strategy A: GPU ReduceScatter -> D2H shard
            # Reduces GPU memory and D2H bandwidth
            self._ensure_shard_buffer(total_size)
            
            start, end = self._get_shard_range(total_size)
            shard_numel = end - start
            
            # ReduceScatter on GPU (sum gradients and scatter shards)
            # Prepare input tensor list for reduce_scatter
            input_list = [
                gpu_grads[i * self._shard_size:min((i + 1) * self._shard_size, total_size)]
                for i in range(self.world_size)
            ]
            
            dist.reduce_scatter(
                self._gpu_shard_buffer[:shard_numel],
                input_list,
                op=dist.ReduceOp.SUM,
                group=self.gpu_group,
            )
            
            # Average the gradients
            self._gpu_shard_buffer[:shard_numel].div_(self.world_size)
            
            # D2H transfer only this rank's shard
            cpu_buffer[start:end].copy_(
                self._gpu_shard_buffer[:shard_numel],
                non_blocking=True,
            )
            
            torch.cuda.current_stream().synchronize()
            
            # AllGather CPU shards to reconstruct full gradient
            # Use CPU group (Gloo backend)
            full_cpu_buffer = cpu_buffer[:total_size].clone()
            dist.all_gather_into_tensor(
                full_cpu_buffer,
                cpu_buffer[start:end].contiguous(),
                group=self.cpu_group,
            )
            cpu_buffer[:total_size].copy_(full_cpu_buffer)
        
        else:
            # Strategy B: D2H full -> CPU AllReduce
            # Each GPU transfers full gradients to CPU, then AllReduce on CPU
            cpu_buffer[:total_size].copy_(gpu_grads[:total_size], non_blocking=True)
            
            # Synchronize GPU before CPU AllReduce
            torch.cuda.current_stream().synchronize()
            
            # AllReduce on CPU using Gloo backend
            dist.all_reduce(
                cpu_buffer[:total_size],
                op=dist.ReduceOp.SUM,
                group=self.cpu_group,
            )
            
            # Average the gradients
            cpu_buffer[:total_size].div_(self.world_size)
    
    def reduce_gradients_to_master(
        self,
        gpu_grads: torch.Tensor,
        cpu_buffer: torch.Tensor,
        layer_state: "LayerRuntimeState",
    ) -> None:
        """Reduce gradients from all GPUs to rank 0's CPU buffer.
        
        This is used for single-optimizer mode where only rank 0 maintains
        the optimizer state and performs parameter updates.
        
        Args:
            gpu_grads: GPU gradient tensor.
            cpu_buffer: Target CPU buffer for gradients (pinned memory).
            layer_state: Layer runtime state for context.
        """
        total_size = gpu_grads.numel()
        
        if not self.is_distributed:
            # Single GPU: direct D2H copy
            cpu_buffer[:total_size].copy_(gpu_grads[:total_size], non_blocking=True)
            torch.cuda.current_stream().synchronize()
            return
        
        if self.grad_mode == "gpu_reduce":
            # Strategy: GPU Reduce -> D2H on rank 0 only
            # Create output buffer (only used on rank 0)
            reduced_gpu = torch.empty_like(gpu_grads[:total_size])
            
            # Reduce to rank 0 on GPU
            dist.reduce(
                gpu_grads[:total_size],
                dst=0,
                op=dist.ReduceOp.SUM,
                group=self.gpu_group,
            )
            
            if self.rank == 0:
                # Average and D2H transfer only on rank 0
                gpu_grads[:total_size].div_(self.world_size)
                cpu_buffer[:total_size].copy_(gpu_grads[:total_size], non_blocking=True)
                torch.cuda.current_stream().synchronize()
        else:
            # Strategy: D2H -> CPU Reduce to rank 0
            # Each GPU transfers full gradients to CPU
            cpu_buffer[:total_size].copy_(gpu_grads[:total_size], non_blocking=True)
            torch.cuda.current_stream().synchronize()
            
            # Reduce on CPU to rank 0 using Gloo backend
            dist.reduce(
                cpu_buffer[:total_size],
                dst=0,
                op=dist.ReduceOp.SUM,
                group=self.cpu_group,
            )
            
            if self.rank == 0:
                # Average the gradients
                cpu_buffer[:total_size].div_(self.world_size)
    
    def broadcast_params_from_master(
        self,
        cpu_params: torch.Tensor,
        total_size: int,
    ) -> None:
        """Broadcast updated parameters from rank 0 to all other ranks.
        
        This is called after rank 0 completes the optimizer update.
        Uses CPU (Gloo) communication to broadcast parameters.
        
        Args:
            cpu_params: CPU parameter tensor (FP32, pinned memory).
            total_size: Number of elements to broadcast.
        """
        if not self.is_distributed:
            return
        
        # Broadcast parameters from rank 0 to all ranks using Gloo
        dist.broadcast(
            cpu_params[:total_size],
            src=0,
            group=self.cpu_group,
        )
    
    def barrier(self) -> None:
        """Synchronize all processes using Gloo backend."""
        if self.is_distributed:
            dist.barrier(group=self.cpu_group)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Process groups are automatically cleaned up by PyTorch
        self._gpu_shard_buffer = None


class NoOpSync:
    """No-op synchronization for single GPU or testing."""
    
    def __init__(self, config: "SlideFSDPConfig"):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.world_size = 1
        self.rank = 0
        self.is_distributed = False
    
    def gather_params_to_gpu(
        self,
        cpu_params: torch.Tensor,
        gpu_buffer: torch.Tensor,
        layer_state: "LayerRuntimeState",
    ) -> None:
        """Direct H2D copy."""
        total_size = cpu_params.numel()
        gpu_buffer[:total_size].copy_(cpu_params[:total_size], non_blocking=True)
    
    def sync_gradients_to_cpu(
        self,
        gpu_grads: torch.Tensor,
        cpu_buffer: torch.Tensor,
        layer_state: "LayerRuntimeState",
    ) -> None:
        """Direct D2H copy."""
        total_size = gpu_grads.numel()
        cpu_buffer[:total_size].copy_(gpu_grads[:total_size], non_blocking=True)
        torch.cuda.current_stream().synchronize()
    
    def reduce_gradients_to_master(
        self,
        gpu_grads: torch.Tensor,
        cpu_buffer: torch.Tensor,
        layer_state: "LayerRuntimeState",
    ) -> None:
        """Direct D2H copy (single GPU, no reduce needed)."""
        total_size = gpu_grads.numel()
        cpu_buffer[:total_size].copy_(gpu_grads[:total_size], non_blocking=True)
        torch.cuda.current_stream().synchronize()
    
    def broadcast_params_from_master(
        self,
        cpu_params: torch.Tensor,
        total_size: int,
    ) -> None:
        """No-op (single GPU, no broadcast needed)."""
        pass
    
    def barrier(self) -> None:
        """No-op barrier."""
        pass
    
    def cleanup(self) -> None:
        """No-op cleanup."""
        pass


def create_distributed_sync(config: "SlideFSDPConfig") -> DistributedSync:
    """Factory function to create appropriate sync handler.
    
    Args:
        config: SlideFSDP configuration.
        
    Returns:
        DistributedSync or NoOpSync instance.
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        return DistributedSync(config)
    else:
        return NoOpSync(config)


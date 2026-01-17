"""Layer State Management for SlideFSDP.

This module provides GPU buffer pool management and per-layer runtime state,
enabling sliding window parameter management similar to SlideFormer.

Key feature: Single-optimizer mode
- Only rank 0 maintains the CPU Adam optimizer and optimizer states
- Gradients are reduced to rank 0, which performs the update
- With shared memory, other ranks see updates after a barrier (no broadcast)
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

if TYPE_CHECKING:
    from config import SlideFSDPConfig


class LayerBufferPool:
    """GPU Buffer Pool for sliding window parameter management.
    
    Pre-allocates a fixed number of GPU buffer units to avoid frequent
    memory allocation/deallocation and reduce fragmentation.
    
    With sliding window mechanism, at most 2 layers are active on GPU
    at any time (current layer computing + next layer prefetching).
    """
    
    def __init__(
        self,
        max_param_size: int,
        dtype: torch.dtype,
        device: torch.device,
        pool_size: int = 2,
    ):
        """Initialize GPU buffer pool.
        
        Args:
            max_param_size: Maximum number of parameters in any layer.
            dtype: Data type for GPU buffers (bf16/fp16).
            device: Target CUDA device.
            pool_size: Number of buffer units to pre-allocate.
        """
        self.max_param_size = max_param_size
        self.dtype = dtype
        self.device = device
        self.pool_size = pool_size
        
        # Pre-allocate buffer units
        self.pool: deque = deque(maxlen=pool_size)
        for _ in range(pool_size):
            unit = {
                "param": torch.empty(max_param_size, dtype=dtype, device=device),
                "grad": torch.empty(max_param_size, dtype=dtype, device=device),
            }
            self.pool.append(unit)
        
        # self._lock = threading.Lock()
    
    def acquire(self) -> Dict[str, torch.Tensor]:
        """Acquire a buffer unit from the pool.
        
        Returns:
            Dictionary with 'param' and 'grad' tensors.
        
        Raises:
            RuntimeError: If pool is exhausted.
        """
        # with self._lock:
            # if not self.pool:
            #     # Fallback: allocate new buffer if pool exhausted
            #     # This shouldn't happen with proper sliding window control
            #     print("Warning: Buffer pool exhausted, allocating new buffer")
            #     return {
            #         "param": torch.empty(
            #             self.max_param_size, dtype=self.dtype, device=self.device
            #         ),
            #         "grad": torch.empty(
            #             self.max_param_size, dtype=self.dtype, device=self.device
            #         ),
            #     }
        return self.pool.popleft()
    
    def release(self, unit: Dict[str, torch.Tensor]) -> None:
        """Release a buffer unit back to the pool.
        
        Args:
            unit: Buffer unit to release.
        """
        # with self._lock:
        #     if len(self.pool) < self.pool_size:
        self.pool.append(unit)
        # If pool is full, let the unit be garbage collected


class LayerRuntimeState(nn.Module):
    """Runtime state management for a single layer.
    
    Manages CPU<->GPU parameter transfer, gradient offloading,
    and asynchronous parameter updates for one transformer layer.
    """
    
    def __init__(
        self,
        layer: nn.Module,
        layer_idx: int,
        config: "SlideFSDPConfig",
        buffer_pool: LayerBufferPool,
        layer_optimizer: Any,
        dist_sync: Any,
        shared_param_manager: Any,
        h2d_executor: ThreadPoolExecutor,
        d2h_executor: ThreadPoolExecutor,
        update_executor: ThreadPoolExecutor,
        cpu_grad_buffer: torch.Tensor,
        bf16_convert_buffer: torch.Tensor,
        is_last_layer: bool = False,
    ):
        """Initialize layer runtime state.
        
        Args:
            layer: The transformer layer module.
            layer_idx: Index of this layer.
            config: SlideFSDP configuration.
            buffer_pool: Shared GPU buffer pool.
            layer_optimizer: Shared LayerAdam optimizer (only on rank 0).
            dist_sync: Distributed synchronization handler.
            shared_param_manager: Shared memory manager for FP32 params (all ranks share).
            h2d_executor: Thread pool for H2D transfers.
            d2h_executor: Thread pool for D2H transfers.
            update_executor: Thread pool for parameter updates.
            cpu_grad_buffer: Shared CPU gradient buffer (pinned memory).
            bf16_convert_buffer: Shared buffer for fp32->bf16 conversion.
            is_last_layer: Whether this is the last decoder layer.
        """
        super().__init__()
        
        self.layer = layer
        self.layer_idx = layer_idx
        self.config = config
        self.buffer_pool = buffer_pool
        self.layer_optimizer = layer_optimizer
        self.dist_sync = dist_sync
        self.shared_param_manager = shared_param_manager
        self.is_last_layer = is_last_layer
        
        # Distributed info
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_distributed = self.world_size > 1
        
        # Thread pools (shared across layers)
        self.h2d_executor = h2d_executor
        self.d2h_executor = d2h_executor
        self.update_executor = update_executor
        
        # Shared buffers
        self._cpu_grad_buffer = cpu_grad_buffer
        self._bf16_convert_buffer = bf16_convert_buffer
        
        # CUDA streams for async transfers
        self.h2d_stream = torch.cuda.Stream(device=config.device)
        self.d2h_stream = torch.cuda.Stream(device=config.device)
        self.compute_stream = torch.cuda.default_stream(device=config.device)
        
        # Events for synchronization
        self.compute_ready = torch.cuda.Event()
        self.compute_ready_bw = torch.cuda.Event()
        
        # Futures for async operations
        self._h2d_future = None
        self._d2h_future = None
        
        # Threading synchronization
        self.update_finished = threading.Event()
        self.update_finished.set()
        self.update_lock = threading.Lock()
        
        # Calculate layer parameter size
        self.total_size = sum(p.numel() for p in self.layer.parameters())
        
        # CPU-side FP32 parameter storage
        # Use shared memory if available (all ranks share the same memory)
        if self.shared_param_manager is not None:
            # Create shared tensor backed by memory-mapped file
            self._cpu_params_flat = self.shared_param_manager.create_shared_tensor(
                layer_idx=layer_idx,
                size=self.total_size,
                dtype=torch.float32,
            )
            self._use_shared_params = True
        else:
            # Single GPU or shared memory not available: use local tensor
            self._cpu_params_flat = torch.empty(
                self.total_size,
                dtype=torch.float32,
                device=torch.device("cpu"),
                pin_memory=True,
            )
            self._use_shared_params = False

        # Multi-GPU now requires shared params (broadcast path removed)
        if self.is_distributed and not self._use_shared_params:
            raise RuntimeError(
                "Multi-GPU requires shared parameters; SharedParamManager was not provided."
            )
        
        # Parameter and gradient views
        self._param_maps: OrderedDict[str, Tuple[int, torch.Size, int]] = OrderedDict()
        self._param_views: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._grad_views: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._param_to_grad_views: OrderedDict[nn.Parameter, torch.Tensor] = OrderedDict()
        
        # Initialize parameter views and copy to CPU
        self._init_param_views()
        
        # Current GPU cache unit (when layer is on GPU)
        self._current_gpu_cache: Optional[Dict[str, torch.Tensor]] = None
        
        # Register layer parameters with optimizer (only on rank 0)
        if self.layer_optimizer is not None:
            self.layer_optimizer.add_layer_params(layer_idx, self.layer.parameters())
    
    def _init_param_views(self) -> None:
        """Initialize parameter views and copy initial values to CPU.
        
        When using shared memory:
        - Only rank 0 copies initial parameter values to shared memory
        - Other ranks wait and then use the same shared memory
        """
        with torch.no_grad():
            offset = 0
            for name, param in self.layer.named_parameters():
                shape = param.shape
                size = param.numel()
                
                self._param_maps[name] = (offset, shape, size)
                
                # Create views into flat CPU tensors
                self._param_views[name] = self._cpu_params_flat[offset:offset + size].view(shape)
                self._grad_views[name] = self._cpu_grad_buffer[offset:offset + size].view(shape)
                
                # Copy initial parameter values to CPU
                # When using shared memory, only rank 0 initializes
                if not self._use_shared_params or self.rank == 0:
                    self._param_views[name].copy_(param.data)
                
                # Point parameter data to CPU view
                param.data = self._param_views[name]
                param.grad = None
                
                # Map parameter to gradient view
                self._param_to_grad_views[param] = self._grad_views[name]
                
                offset += size
        
        # Synchronize after initialization when using shared memory
        # to ensure rank 0's writes are visible to all ranks
        if self._use_shared_params and self.is_distributed:
            # Use Gloo group for CPU synchronization
            cpu_group = self.dist_sync.cpu_group if self.dist_sync else None
            dist.barrier(group=cpu_group)
    
    def to_device_async(self, is_backward: bool = False) -> None:
        """Asynchronously load layer parameters to GPU.
        
        Args:
            is_backward: Whether this is for backward pass (zeros gradients).
        """
        def _h2d_task():
            # Wait for any pending D2H to complete first
            # This prevents race condition where D2H is in progress but
            # _current_gpu_cache is not yet None
            self.wait_for_d2h()
            
            # Skip if layer is already on GPU
            if self._current_gpu_cache is not None:
                return
            
            # Wait for any pending parameter update
            self.wait_for_update()
            
            with torch.cuda.stream(self.h2d_stream):
                # Acquire buffer from pool
                self._current_gpu_cache = self.buffer_pool.acquire()
                
                if is_backward:
                    self._current_gpu_cache["grad"].zero_()
                
                with torch.no_grad():
                    # Convert FP32 CPU params to bf16 via shared buffer
                    self._bf16_convert_buffer[:self.total_size].copy_(self._cpu_params_flat)
                    
                    # Use dist_sync for H2D transfer (handles shard/full modes)
                    if self.dist_sync is not None:
                        self.dist_sync.gather_params_to_gpu(
                            self._bf16_convert_buffer[:self.total_size],
                            self._current_gpu_cache["param"][:self.total_size],
                            self,
                        )
                    else:
                        # Direct copy if no distributed sync
                        self._current_gpu_cache["param"][:self.total_size].copy_(
                            self._bf16_convert_buffer[:self.total_size],
                            non_blocking=True,
                        )
                    
                    # Update parameter data pointers to GPU cache
                    for name, param in self.layer.named_parameters():
                        offset, shape, size = self._param_maps[name]
                        param.data = self._current_gpu_cache["param"][offset:offset + size].view(shape)
                        if is_backward:
                            param.grad = self._current_gpu_cache["grad"][offset:offset + size].view(shape)
                        else:
                            param.grad = None
                
                self.h2d_stream.synchronize()
        
        self._h2d_future = self.h2d_executor.submit(_h2d_task)
    
    def wait_for_h2d(self, timeout: Optional[float] = None) -> None:
        """Wait for H2D transfer to complete.
        
        Args:
            timeout: Maximum time to wait in seconds.
        """
        if self._h2d_future is not None and not self._h2d_future.done():
            self._h2d_future.result(timeout=timeout)
    
    def to_offload_async(
        self,
        is_backward: bool = False,
        prev_update: Optional[threading.Event] = None,
    ) -> None:
        """Asynchronously offload layer from GPU to CPU.
        
        Args:
            is_backward: Whether this is after backward pass (copies gradients).
            prev_update: Event to wait for before starting offload.
        """
        def _d2h_task():
            # Skip if layer is already offloaded (handles checkpoint recomputation)
            if self._current_gpu_cache is None:
                return
            
            # Wait for compute to finish
            if is_backward:
                self.d2h_stream.wait_event(self.compute_ready_bw)
            else:
                self.d2h_stream.wait_event(self.compute_ready)
            
            # Wait for previous layer's update if specified
            if prev_update is not None and not prev_update.is_set():
                prev_update.wait()
            
            with torch.cuda.stream(self.d2h_stream):
                with torch.no_grad():
                    if is_backward:
                        # Use dist_sync for gradient synchronization
                        # In single-optimizer mode: reduce gradients to rank 0
                        if self.dist_sync is not None:
                            self.dist_sync.reduce_gradients_to_master(
                                self._current_gpu_cache["grad"][:self.total_size],
                                self._cpu_grad_buffer[:self.total_size],
                                self,
                            )
                        else:
                            # Direct copy if no distributed sync
                            self._cpu_grad_buffer[:self.total_size].copy_(
                                self._current_gpu_cache["grad"][:self.total_size],
                                non_blocking=True,
                            )
                        
                        # Restore parameter pointers to CPU views
                        for name, param in self.layer.named_parameters():
                            param.data = self._param_views[name]
                            param.grad = None
                        
                        self.d2h_stream.synchronize()
                    else:
                        # Forward pass: just restore CPU pointers
                        for name, param in self.layer.named_parameters():
                            param.data = self._param_views[name]
                            param.grad = None
                
                # Release buffer back to pool
                self.buffer_pool.release(self._current_gpu_cache)
                self._current_gpu_cache = None
        
        self._d2h_future = self.d2h_executor.submit(_d2h_task)
    
    def wait_for_d2h(self, timeout: Optional[float] = None) -> None:
        """Wait for D2H transfer to complete.
        
        Args:
            timeout: Maximum time to wait in seconds.
        """
        if self._d2h_future is not None and not self._d2h_future.done():
            self._d2h_future.result(timeout=timeout)
    
    def _do_update(self) -> bool:
        """Execute parameter update on CPU.
        
        In single-optimizer mode with shared memory:
        - Only rank 0 executes the Adam update
        - Parameters are in shared memory, so no broadcast needed
        - A barrier ensures all ranks see the updated values
        """
        # Wait for D2H to complete
        self.wait_for_d2h()
        
        with self.update_lock:
            # Only rank 0 performs the optimizer step
            if self.layer_optimizer is not None and self.rank == 0:
                # Use gradient views for update
                self.layer_optimizer.step_with_grad_views(
                    self.layer_idx,
                    self._param_to_grad_views,
                )
            
            # Synchronize to ensure rank 0's update is visible to all ranks
            # Shared memory + barrier; multi-GPU always uses shared params
            if self.is_distributed and self._use_shared_params:
                cpu_group = self.dist_sync.cpu_group if self.dist_sync else None
                dist.barrier(group=cpu_group)
            
            self.update_finished.set()
            return True
    
    def update_params(self) -> None:
        """Trigger asynchronous parameter update."""
        self.update_finished.clear()
        self.update_executor.submit(self._do_update)
    
    def wait_for_update(self, timeout: Optional[float] = None) -> None:
        """Wait for parameter update to complete.
        
        Args:
            timeout: Maximum time to wait in seconds.
        
        Raises:
            TimeoutError: If update doesn't complete within timeout.
        """
        if not self.update_finished.is_set():
            if not self.update_finished.wait(timeout=timeout):
                raise TimeoutError(
                    f"Layer {self.layer_idx} parameter update timed out"
                )
    
    def forward(self, *args, **kwargs):
        """Forward pass through the layer.
        
        This is a placeholder - actual forward is called via self.layer().
        """
        raise NotImplementedError(
            "Use self.layer() directly for forward pass, "
            "LayerRuntimeState manages state, not computation"
        )


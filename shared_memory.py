"""Shared Memory Manager for SlideFSDP.

Provides cross-process shared memory for FP32 parameters using memory-mapped files.
All ranks share the same parameter memory, so rank 0's updates are immediately
visible to all other ranks without explicit broadcast.
"""

from __future__ import annotations

import os
import mmap
import uuid
import atexit
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.distributed as dist


class SharedParamManager:
    """Manages shared FP32 parameters across all ranks using memory-mapped files.
    
    Key features:
    - Uses /dev/shm (tmpfs) for fast shared memory access
    - All ranks share the same parameter memory
    - Rank 0 creates the shared memory, other ranks attach to it
    - No broadcast needed after parameter update
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize shared parameter manager.
        
        Args:
            session_id: Unique identifier for this training session.
                       If None, generates a new UUID.
        """
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_distributed = self.world_size > 1
        
        # Create Gloo process group for CPU communication
        self._cpu_group = None
        if self.is_distributed:
            self._cpu_group = dist.new_group(backend="gloo")
        
        # Generate or use provided session ID
        if session_id is None:
            if self.rank == 0:
                self._session_id = str(uuid.uuid4())[:8]
            else:
                self._session_id = None
            
            # Broadcast session ID from rank 0 using Gloo backend
            if self.is_distributed:
                if self.rank == 0:
                    session_tensor = torch.tensor(
                        [ord(c) for c in self._session_id],
                        dtype=torch.int64,
                    )
                else:
                    session_tensor = torch.zeros(8, dtype=torch.int64)
                
                # Use Gloo group for CPU tensor broadcast
                dist.broadcast(session_tensor, src=0, group=self._cpu_group)
                
                if self.rank != 0:
                    self._session_id = ''.join(chr(c) for c in session_tensor.tolist())
        else:
            self._session_id = session_id
        
        # Base directory for shared memory files
        self._shm_dir = "/dev/shm"
        if not os.path.exists(self._shm_dir):
            # Fallback to /tmp if /dev/shm not available
            self._shm_dir = "/tmp"
        
        # Storage for memory maps
        self._mmaps: Dict[int, mmap.mmap] = {}
        self._files: Dict[int, object] = {}
        self._tensors: Dict[int, torch.Tensor] = {}
        self._file_paths: List[str] = []
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def _get_shm_path(self, layer_idx: int) -> str:
        """Get shared memory file path for a layer.
        
        Args:
            layer_idx: Layer index.
            
        Returns:
            Path to the shared memory file.
        """
        return os.path.join(
            self._shm_dir,
            f"ssdp_{self._session_id}_layer_{layer_idx}.bin"
        )
    
    def create_shared_tensor(
        self,
        layer_idx: int,
        size: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Create or attach to a shared tensor for a layer.
        
        Rank 0 creates the shared memory file, other ranks attach to it.
        
        Args:
            layer_idx: Layer index.
            size: Number of elements in the tensor.
            dtype: Data type (default: float32).
            
        Returns:
            Shared tensor backed by memory-mapped file.
        """
        if layer_idx in self._tensors:
            return self._tensors[layer_idx]
        
        shm_path = self._get_shm_path(layer_idx)
        bytes_size = size * dtype.itemsize if hasattr(dtype, 'itemsize') else size * 4
        
        # Rank 0 creates the file
        if self.rank == 0:
            with open(shm_path, 'wb') as f:
                f.write(b'\x00' * bytes_size)
            self._file_paths.append(shm_path)
        
        # Synchronize to ensure file is created (use Gloo group)
        if self.is_distributed:
            dist.barrier(group=self._cpu_group)
        
        # All ranks open and map the file
        f = open(shm_path, 'r+b')
        mm = mmap.mmap(f.fileno(), bytes_size)
        
        # Store references
        self._files[layer_idx] = f
        self._mmaps[layer_idx] = mm
        
        # Create numpy array from mmap (zero-copy)
        np_dtype = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.bfloat16: np.float32,  # bfloat16 stored as float32
        }.get(dtype, np.float32)
        
        np_array = np.frombuffer(mm, dtype=np_dtype)
        
        # Create torch tensor (zero-copy)
        tensor = torch.from_numpy(np_array)
        self._tensors[layer_idx] = tensor
        
        return tensor
    
    def get_shared_tensor(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get existing shared tensor for a layer.
        
        Args:
            layer_idx: Layer index.
            
        Returns:
            Shared tensor or None if not created.
        """
        return self._tensors.get(layer_idx)
    
    def sync_before_read(self) -> None:
        """Synchronize before reading shared parameters.
        
        Call this before H2D transfer to ensure rank 0's updates are visible.
        This is a lightweight barrier using threading events.
        """
        # Memory barrier to ensure visibility
        if self.is_distributed:
            # Use a simple barrier - could be optimized with threading.Event
            dist.barrier()
    
    def cleanup(self) -> None:
        """Clean up shared memory resources."""
        # Close memory maps
        for mm in self._mmaps.values():
            try:
                mm.close()
            except:
                pass
        
        # Close files
        for f in self._files.values():
            try:
                f.close()
            except:
                pass
        
        # Remove files (only rank 0)
        if self.rank == 0:
            for path in self._file_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except:
                    pass
        
        self._mmaps.clear()
        self._files.clear()
        self._tensors.clear()


class SharedGradBuffer:
    """Shared gradient buffer for gradient reduction.
    
    Only rank 0 needs gradients for optimizer update, but we use a shared
    buffer for efficient gradient reduction.
    """
    
    def __init__(
        self,
        max_size: int,
        dtype: torch.dtype,
        session_id: str,
    ):
        """Initialize shared gradient buffer.
        
        Args:
            max_size: Maximum gradient size.
            dtype: Gradient data type (bf16/fp16).
            session_id: Training session ID.
        """
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_distributed = self.world_size > 1
        
        # For gradients, we use a local pinned buffer
        # Gradients are reduced via communication, not shared memory
        self.buffer = torch.empty(
            max_size,
            dtype=dtype,
            device=torch.device("cpu"),
            pin_memory=True,
        )
    
    def get_buffer(self) -> torch.Tensor:
        """Get the gradient buffer."""
        return self.buffer


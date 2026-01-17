"""SlideFSDP: Multi-GPU SlideFormer Implementation.

Extends SlideFormer's layer-wise CPU offload mechanism to multi-GPU training,
with configurable parameter transfer and gradient synchronization strategies.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import SlideFSDPConfig
from layer_state import LayerBufferPool, LayerRuntimeState
from distributed_sync import DistributedSync, create_distributed_sync
from shared_memory import SharedParamManager

# Import LayerAdam from local optimizer package
from optimizer import LayerAdam

# Import SlidingCheckpoint for activation offload
from sliding_checkpoint import SlidingCheckpoint


# Try to import Liger kernel for fused operations
try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False
    LigerFusedLinearCrossEntropyLoss = None


class OutputLayer(nn.Module):
    """Combined output layer: norm + lm_head + fused cross-entropy loss.
    
    This wraps norm and lm_head together so they can be managed as a single
    unit in LayerRuntimeState, ensuring both are on GPU when computing loss.
    """
    
    def __init__(self, norm: nn.Module, lm_head: nn.Module, hidden_size: int):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head
        self.hidden_size = hidden_size
        
        # Fused cross-entropy loss
        if HAS_LIGER:
            self.lce = LigerFusedLinearCrossEntropyLoss(reduction="mean")
        else:
            print("Warning: Liger kernel not available, using standard cross-entropy loss")
            self.lce = None
            self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass through norm + lm_head.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_length, hidden_size].
            labels: Labels for loss computation [batch_size, seq_length].
            
        Returns:
            Tuple of (loss, logits). If labels is None, loss is None.
        """
        # Apply layer norm
        hidden_states = self.norm(hidden_states)
        
        loss = None
        logits = None
        
        if labels is not None:
            if self.lce is not None:
                # Fused linear cross-entropy
                shift_hidden = hidden_states[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                shift_hidden = shift_hidden.view(-1, self.hidden_size)
                shift_labels = shift_labels.view(-1)
                
                loss = self.lce(
                    self.lm_head.weight,
                    shift_hidden,
                    shift_labels,
                )
            else:
                # Standard forward + cross-entropy
                logits = self.lm_head(hidden_states)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = self.loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
        else:
            logits = self.lm_head(hidden_states)
        
        return loss, logits


class SlideFSDP(nn.Module):
    """Multi-GPU SlideFormer with configurable synchronization strategies.
    
    Key features:
    - Layer-wise parameter offload to CPU (FP32)
    - Sliding window GPU buffer management
    - Configurable parameter transfer: full H2D vs shard + AllGather
    - Configurable gradient sync: GPU reduce vs CPU reduce
    - Integration with LayerAdam for CPU-side optimization
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        config: SlideFSDPConfig,
    ):
        """Initialize SlideFSDP wrapper.
        
        Args:
            model: HuggingFace PreTrainedModel (e.g., LlamaForCausalLM).
            config: SlideFSDP configuration.
        """
        super().__init__()
        
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.base_model = model
        self.model_config = model.config
        
        # Distributed info
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Split model into components
        self._split_model(model)
        
        # Calculate max parameter size across all layers
        self.max_param_size = self._calculate_max_param_size()
        self.total_params = sum(
            sum(p.numel() for p in layer.parameters())
            for layer in self.all_layers
        )
        
        # Create thread pools for async operations
        self.h2d_executor = ThreadPoolExecutor(max_workers=1)
        self.d2h_executor = self.h2d_executor
        self.update_executor = ThreadPoolExecutor(max_workers=1)
        
        # Create GPU buffer pool
        self.buffer_pool = LayerBufferPool(
            max_param_size=self.max_param_size,
            dtype=self.dtype,
            device=self.device,
            pool_size=config.gpu_buffer_pool_size,
        )
        
        # Create shared CPU buffers (pinned memory)
        self._cpu_grad_buffer = torch.empty(
            self.max_param_size,
            dtype=self.dtype,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        self._bf16_convert_buffer = torch.empty(
            self.max_param_size,
            dtype=self.dtype,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        
        # Create distributed synchronization handler
        self.dist_sync = create_distributed_sync(config)
        
        # Create shared parameter manager for cross-process memory sharing
        # All ranks share the same FP32 parameters via memory-mapped files
        self.shared_param_manager = SharedParamManager() if self.world_size > 1 else None
        
        # Create LayerAdam optimizer (only on rank 0 in distributed mode)
        self.layer_optimizer = self._create_optimizer(config)
        
        # Create layer runtime states
        self.layer_states: List[LayerRuntimeState] = []
        self._create_layer_states()
        
        # Setup autograd functions for layer management
        self._setup_autograd_functions()
        
        # Activation checkpoint storage
        self.layer_tensors = None
        self.position_embeddings = None
        
        # Print model info
        if self.rank == 0:
            self._print_model_info()
    
    def _split_model(self, model: PreTrainedModel) -> None:
        """Split model into embedding, decoder layers, and output components."""
        # Get decoder (handles different model architectures)
        decoder = model.get_decoder()
        
        # Get layer list
        if hasattr(decoder, "layers"):
            layers = decoder.layers
        elif hasattr(decoder, "block"):
            layers = decoder.block
        else:
            raise ValueError("Cannot find decoder layers in model")
        
        self.num_decoder_layers = len(layers)
        
        # Store references
        self.embed_tokens = model.get_input_embeddings()
        self.decoder_layers = nn.ModuleList(layers)
        
        # Rotary embeddings (if available)
        self.rotary_emb = getattr(decoder, "rotary_emb", None)
        self._update_causal_mask = getattr(decoder, "_update_causal_mask", None)
        
        # Create combined output layer (norm + lm_head)
        self.output_layer = OutputLayer(
            norm=decoder.norm,
            lm_head=model.get_output_embeddings(),
            hidden_size=model.config.hidden_size,
        )
        
        # Create combined layer list for iteration
        self.all_layers = [self.embed_tokens] + list(self.decoder_layers) + [self.output_layer]
    
    def _calculate_max_param_size(self) -> int:
        """Calculate maximum parameter count across all layers."""
        max_size = 0
        for layer in self.all_layers:
            layer_size = sum(p.numel() for p in layer.parameters())
            max_size = max(max_size, layer_size)
        
        return max_size
    
    def _create_optimizer(self, config: SlideFSDPConfig) -> Optional[Any]:
        """Create LayerAdam optimizer.
        
        In multi-GPU mode, only rank 0 creates and maintains the optimizer.
        This saves CPU memory (optimizer states only on rank 0) and avoids
        redundant computation (only one update per step).
        """   
        # Only rank 0 creates the optimizer in distributed mode
        if self.world_size > 1 and self.rank != 0:
            if self.rank == 1:  # Only print once
                print("Note: Only rank 0 maintains CPU Adam optimizer (single-optimizer mode)")
            return None
        
        # all_layers: embed + decoder_layers + output_layer (which includes norm + lm_head)
        num_layers = len(self.all_layers)
        
        optimizer = LayerAdam(
            lr=config.lr,
            betas=config.adam_betas,
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
            bias_correction=config.bias_correction,
            fp32_optimizer_state=True,
            num_layer=num_layers,
            nvme_offload_fraction=config.optimizer_nvme_offload_fraction,
            offload_dir=config.optimizer_offload_dir,
            prefetch=True,
        )
        
        return optimizer
    
    def _create_layer_states(self) -> None:
        """Create LayerRuntimeState for each layer."""
        # Embedding layer (idx=0)
        embed_state = LayerRuntimeState(
            layer=self.embed_tokens,
            layer_idx=0,
            config=self.config,
            buffer_pool=self.buffer_pool,
            layer_optimizer=self.layer_optimizer,
            dist_sync=self.dist_sync,
            shared_param_manager=self.shared_param_manager,
            h2d_executor=self.h2d_executor,
            d2h_executor=self.d2h_executor,
            update_executor=self.update_executor,
            cpu_grad_buffer=self._cpu_grad_buffer,
            bf16_convert_buffer=self._bf16_convert_buffer,
            is_last_layer=False,
        )
        self.layer_states.append(embed_state)
        
        # Decoder layers (idx=1 to num_decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            decoder_state = LayerRuntimeState(
                layer=layer,
                layer_idx=i + 1,
                config=self.config,
                buffer_pool=self.buffer_pool,
                layer_optimizer=self.layer_optimizer,
                dist_sync=self.dist_sync,
                shared_param_manager=self.shared_param_manager,
                h2d_executor=self.h2d_executor,
                d2h_executor=self.d2h_executor,
                update_executor=self.update_executor,
                cpu_grad_buffer=self._cpu_grad_buffer,
                bf16_convert_buffer=self._bf16_convert_buffer,
                is_last_layer=False,
            )
            self.layer_states.append(decoder_state)
        
        # Output layer (norm + lm_head combined, idx=num_decoder_layers+1)
        output_state = LayerRuntimeState(
            layer=self.output_layer,
            layer_idx=self.num_decoder_layers + 1,
            config=self.config,
            buffer_pool=self.buffer_pool,
            layer_optimizer=self.layer_optimizer,
            dist_sync=self.dist_sync,
            shared_param_manager=self.shared_param_manager,
            h2d_executor=self.h2d_executor,
            d2h_executor=self.d2h_executor,
            update_executor=self.update_executor,
            cpu_grad_buffer=self._cpu_grad_buffer,
            bf16_convert_buffer=self._bf16_convert_buffer,
            is_last_layer=True,
        )
        self.layer_states.append(output_state)
    
    def _setup_autograd_functions(self) -> None:
        """Setup autograd functions for layer management.
        
        We use autograd functions instead of forward hooks to manage prefetch/offload.
        This is because torch.utils.checkpoint recomputes forward during backward,
        which would trigger forward hooks and cause issues.
        
        Autograd functions are NOT re-executed during checkpoint recomputation -
        checkpoint only recomputes the function passed to it, not the surrounding code.
        
        Design:
        - PreLayerFunction: Before layer forward - prefetch next layer, wait for current
        - PostLayerFunction: After layer forward - record event, offload, setup backward
        
        Execution order:
        - Forward:  PreLayer.fwd → checkpoint(layer.fwd) → PostLayer.fwd
        - Backward: PostLayer.bwd (pre-backward) → checkpoint(layer.bwd) → PreLayer.bwd (no-op)
        """
        slide_fsdp = self  # Capture reference for inner classes
        
        class PreLayerFunction(torch.autograd.Function):
            """Executed before layer forward: prefetch and wait."""
            @staticmethod
            def forward(ctx, hidden_states, layer_state, next_layer_state):
                ctx.set_materialize_grads(False)
                
                # Prefetch next layer (if exists)
                if next_layer_state is not None:
                    # For output layer, we need grad buffer
                    is_output = (next_layer_state.layer_idx == len(slide_fsdp.layer_states) - 1)
                    next_layer_state.to_device_async(is_backward=is_output)
                
                # Reset post_called flag for backward
                layer_state._post_called = False
                
                # Wait for current layer to be loaded
                layer_state.wait_for_h2d()
                
                return hidden_states
            
            @staticmethod
            def backward(ctx, grad_output):
                # No-op: pre-backward is handled by PostLayerFunction
                return grad_output, None, None
        
        class PostLayerFunction(torch.autograd.Function):
            """Executed after layer forward: offload and setup backward."""
            @staticmethod
            def forward(ctx, hidden_states, layer_state, is_last_layer):
                ctx.layer_state = layer_state
                ctx.set_materialize_grads(False)
                
                # Record compute ready event
                layer_state.compute_ready.record()
                
                # Offload current layer (unless it's the last layer which stays for backward)
                if not is_last_layer:
                    layer_state.to_offload_async(is_backward=False)
                
                return hidden_states
            
            @staticmethod
            def backward(ctx, grad_output):
                # Pre-backward: prefetch previous layer, wait for current
                layer_state = ctx.layer_state
                slide_fsdp._on_pre_backward(layer_state)
                return grad_output, None, None
        
        # Store function classes for use in forward
        self._PreLayerFunction = PreLayerFunction
        self._PostLayerFunction = PostLayerFunction
        
        # Register gradient hooks for post-backward (gradient offload and update)
        for idx, layer_state in enumerate(self.layer_states):
            layer_state._post_called = False
            
            def make_grad_hook(layer_state, idx):
                def grad_hook(grad):
                    if not layer_state._post_called:
                        layer_state._post_called = True
                        self._on_post_backward(layer_state, idx)
                    return grad
                return grad_hook
            
            # Register on first parameter of each layer
            for param in layer_state.layer.parameters():
                param.register_hook(make_grad_hook(layer_state, idx))
                break
    
    def _on_pre_backward(self, layer_state: LayerRuntimeState) -> None:
        """Called before backward pass through a layer.
        
        Design: The current layer should already be on GPU because:
        - For the last layer: it was not offloaded after forward
        - For other layers: it was prefetched by the previous layer's _on_pre_backward
        """
        idx = layer_state.layer_idx
        prev_idx = idx - 1
        
        # Prefetch previous layer for backward (will be needed next)
        if prev_idx >= 0:
            self.layer_states[prev_idx].to_device_async(is_backward=True)
        
        # Wait for current layer's H2D to complete
        # (either from forward prefetch or from previous layer's backward prefetch)
        layer_state.wait_for_h2d()
    
    def _on_post_backward(self, layer_state: LayerRuntimeState, idx: int) -> None:
        """Called after backward pass through a layer."""
        # Record backward compute ready
        layer_state.compute_ready_bw.record(torch.cuda.current_stream())
        
        # Get previous layer's update event for synchronization
        prev_update = None
        if idx + 1 < len(self.layer_states):
            prev_update = self.layer_states[idx + 1].update_finished
        
        # Offload gradients
        layer_state.to_offload_async(is_backward=True, prev_update=prev_update)
        
        # Trigger CPU parameter update
        layer_state.update_params()
    
    def _allocate_activation_tensors(
        self,
        batch_size: int,
        seq_length: int,
        hidden_size: int,
    ) -> None:
        """Allocate CPU tensors for activation checkpointing."""
        if self.config.ac_offload_mode == "none":
            return
        
        self.layer_tensors = []
        
        for _ in range(self.num_decoder_layers):
            if self.model_config._attn_implementation == "flash_attention_2":
                layer_tensors = [
                    torch.empty(
                        (batch_size, seq_length, hidden_size),
                        dtype=self.dtype,
                        pin_memory=True,
                    ),
                    torch.empty(
                        (batch_size, seq_length),
                        dtype=torch.bool,
                        pin_memory=True,
                    ),
                ]
            else:
                layer_tensors = [
                    torch.empty(
                        (batch_size, seq_length, hidden_size),
                        dtype=self.dtype,
                        pin_memory=True,
                    ),
                    torch.empty(
                        (batch_size, 1, seq_length, seq_length),
                        dtype=self.dtype,
                        pin_memory=True,
                    ),
                ]
            self.layer_tensors.append(layer_tensors)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass with layer-wise offloading.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length].
            attention_mask: Attention mask [batch_size, seq_length].
            labels: Labels for loss computation [batch_size, seq_length].
            
        Returns:
            CausalLMOutputWithPast with loss and logits.
        """
        batch_size, seq_length = input_ids.shape
        hidden_size = self.model_config.hidden_size
        
        # Allocate activation tensors on first forward
        if self.layer_tensors is None and self.config.ac_offload_mode != "none":
            self._allocate_activation_tensors(batch_size, seq_length, hidden_size)
        
        # ===== Embedding layer =====
        embed_state = self.layer_states[0]
        next_state = self.layer_states[1] if len(self.layer_states) > 1 else None
        
        # Prefetch embedding layer first (no previous PreLayerFunction to trigger it)
        embed_state.to_device_async(is_backward=False)
        
        # For embedding layer, we handle prefetch/wait manually since input_ids is long tensor
        # Prefetch next layer (first decoder)
        if next_state is not None:
            next_state.to_device_async(is_backward=False)
        embed_state._post_called = False
        embed_state.wait_for_h2d()
        
        # Compute embedding
        hidden_states = embed_state.layer(input_ids)
        
        # Use PostLayerFunction for: record event, offload, setup backward
        # is_last_layer=False so it will offload
        hidden_states = self._PostLayerFunction.apply(hidden_states, embed_state, False)
        
        # Prepare causal mask
        cache_position = torch.arange(
            0, seq_length, device=hidden_states.device
        )
        
        causal_mask = None
        if self._update_causal_mask is not None:
            causal_mask = self._update_causal_mask(
                attention_mask, hidden_states, cache_position, None, None
            )
        
        # Compute position embeddings
        if self.position_embeddings is None and self.rotary_emb is not None:
            position_ids = cache_position.unsqueeze(0)
            self.position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # ===== Decoder layers =====
        # Using autograd functions ensures prefetch/offload are NOT re-executed
        # during checkpoint recomputation (only the checkpoint function is recomputed)
        num_decoder_states = len(self.layer_states) - 2  # excluding embed and output
        
        for i in range(num_decoder_states):
            layer_state = self.layer_states[i + 1]
            layer_idx = i + 1
            is_last_decoder = (i == num_decoder_states - 1)
            next_state = self.layer_states[i + 2] if (i + 2) < len(self.layer_states) else None
            
            # Pre-layer: prefetch next, wait for current (via autograd function)
            hidden_states = self._PreLayerFunction.apply(hidden_states, layer_state, next_state)
            
            # Layer forward (may be checkpointed)
            def _layer_forward(hidden_states, attention_mask, position_embeddings, layer):
                return layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    output_attentions=False,
                    use_cache=False,
                )[0]
            
            if self.config.enable_activation_checkpoint:
                with SlidingCheckpoint(
                    layer_idx=layer_idx,
                    layer_tensors=self.layer_tensors,
                    gds_offload=self.config.ac_offload_mode == "nvme",
                    is_last_layer=is_last_decoder,
                    no_mask=attention_mask is None,
                    device=str(self.device),
                ):
                    hidden_states = checkpoint(
                        _layer_forward,
                        hidden_states,
                        causal_mask,
                        self.position_embeddings,
                        layer_state.layer,
                        use_reentrant=False,
                    )
            else:
                hidden_states = _layer_forward(
                    hidden_states,
                    causal_mask,
                    self.position_embeddings,
                    layer_state.layer,
                )
            
            # Post-layer: record event, offload, setup backward (via autograd function)
            hidden_states = self._PostLayerFunction.apply(hidden_states, layer_state, is_last_decoder)
        
        # ===== Output layer =====
        output_state = self.layer_states[-1]
        
        # Pre-layer for output (no next layer to prefetch)
        hidden_states = self._PreLayerFunction.apply(hidden_states, output_state, None)
        
        # Compute loss via output layer
        loss, logits = self.output_layer(hidden_states, labels)
        
        # Post-layer for output (is_last_layer=True, so won't offload in forward)
        if loss is not None:
            loss = self._PostLayerFunction.apply(loss, output_state, True)
        
        # Backward pass if training
        if loss is not None:
            loss.backward()
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )
    
    def update_learning_rate(self, new_lr: float) -> None:
        """Update optimizer learning rate.
        
        Args:
            new_lr: New learning rate.
        """
        if self.layer_optimizer is not None:
            self.layer_optimizer.update_learning_rate(new_lr)
    
    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        """Clip gradient norm across all layers.
        
        Args:
            max_norm: Maximum gradient norm.
            
        Returns:
            Total gradient norm before clipping.
        """
        # Wait for all updates to complete
        for layer_state in self.layer_states:
            layer_state.wait_for_update()
        
        # Compute total norm
        total_norm_sq = 0.0
        for layer_state in self.layer_states:
            for param in layer_state.layer.parameters():
                if param.grad is not None:
                    total_norm_sq += param.grad.data.norm(2).item() ** 2
        
        total_norm = total_norm_sq ** 0.5
        
        # Clip gradients
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for layer_state in self.layer_states:
                for param in layer_state.layer.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(clip_coef)
        
        return torch.tensor(total_norm)
    
    def _print_model_info(self) -> None:
        """Print model information."""
        param_bytes = self.total_params * 2  # bf16
        opt_state_bytes = self.total_params * 4 * 2  # fp32 exp_avg + exp_avg_sq
        
        print(f"\n{'='*60}")
        print(f"SlideFSDP Model Info")
        print(f"{'='*60}")
        print(f"Model type: {self.model_config.model_type}")
        print(f"Num decoder layers: {self.num_decoder_layers}")
        print(f"Total parameters: {self.total_params:,}")
        print(f"Max layer param size: {self.max_param_size:,}")
        print(f"Param transfer mode: {self.config.param_transfer_mode}")
        print(f"Grad sync mode: {self.config.grad_sync_mode}")
        print(f"World size: {self.world_size}")
        print(f"Estimated CPU memory:")
        print(f"  - Parameters (FP32): {self.total_params * 4 / 1e9:.2f} GB")
        print(f"  - Optimizer states: {opt_state_bytes / 1e9:.2f} GB")
        print(f"  - Total: {(self.total_params * 4 + opt_state_bytes) / 1e9:.2f} GB")
        print(f"GPU buffer pool: {self.config.gpu_buffer_pool_size} units")
        print(f"{'='*60}\n")
    
    def save_pretrained(self, output_dir: str) -> str:
        """Save model to pretrained format.
        
        Args:
            output_dir: Output directory.
            
        Returns:
            Output directory path.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Wait for all updates
        for layer_state in self.layer_states:
            layer_state.wait_for_update()
        
        # Convert FP32 params to bf16
        with torch.no_grad():
            for layer_state in self.layer_states:
                for name, param in layer_state.layer.named_parameters():
                    param.data = param.data.to(dtype=self.dtype)
        
        # Save base model
        self.base_model.save_pretrained(output_dir)
        
        if self.rank == 0:
            print(f"Model saved to {output_dir}")
        
        return output_dir
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "h2d_executor"):
            self.h2d_executor.shutdown(wait=False)
        if hasattr(self, "d2h_executor"):
            self.d2h_executor.shutdown(wait=False)
        if hasattr(self, "update_executor"):
            self.update_executor.shutdown(wait=False)
        if hasattr(self, "dist_sync"):
            self.dist_sync.cleanup()


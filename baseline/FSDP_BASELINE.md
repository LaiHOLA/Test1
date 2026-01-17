# 原生 FSDP 微调基线

本节提供 `fsdp_baseline.py` 的使用说明，以及 PyTorch FSDP 关键配置项、ZeRO 等级映射、CPU 卸载策略与代码结构解读，帮助你在多卡环境下快速跑通原生 FSDP 微调流程。

---

## 运行方式

1. **准备分布式环境**：建议使用 `torchrun`，并确保已经设置好 NCCL/Gloo 等通信后端。
2. **编辑配置**：可直接修改 `TrainConfig` 默认值，或创建 JSON 配置文件并通过 `--config-path` 传入。
3. **启动训练**：

```bash
torchrun --nproc_per_node=8 SSDP/fsdp_baseline.py --config-path fsdp_config.json --save-config
```

脚本会在 rank0 上输出吞吐、TFLOPS、loss 等信息，并在 `--save-final` 为真时保存完整模型权重与 tokenizer。

---

## 主要可调参数

下表罗列了 `TrainConfig` 中的关键字段（可通过 JSON 配置覆盖）：

| 参数 | 说明 | 典型取值 |
| --- | --- | --- |
| `model_path` | Hugging Face 权重路径，示例使用 Llama 3.1-8B | `/home/scc/models/Llama-3.1-8B-Instruct/` |
| `train_batch_size` / `eval_batch_size` | 每个节点（local rank）上的批大小 | 4 |
| `max_seq_length` | DummyDataset 生成的序列长度 | 1024 |
| `dtype` | 参数主存储 dtype（`bf16` / `fp16` / `fp32`）| `bf16` |
| `sharding_strategy` | FSDP 分片策略，见 ZeRO 映射 | `full_shard` |
| `cpu_offload` | 是否启用 `CPUOffload(offload_params=True)` | `true` / `false` |
| `wrap_policy` | 模块自动切分策略 (`transformer` / `size` / `none`) | `transformer` |
| `size_min_params` | 当 `wrap_policy=size` 时的最小参数量阈值 | `30_000_000` |
| `backward_prefetch` | 反向预取策略：`backward_pre` / `backward_post` / `none` | `backward_pre` |
| `forward_prefetch` | 是否启用前向预取下一段参数 | `true` |
| `limit_all_gathers` | 限制同一时刻活跃的 all-gather 层数，降低峰值显存 | `true` |
| `clip_grad_norm` | 全局梯度裁剪阈值 | `1.0` |
| `save_final` | 训练结束是否保存全量权重 | `false` |

> 混合精度：当 `dtype` 非 `fp32` 时，会构造 `MixedPrecision`，参数、梯度按 `dtype` 存储，Reduce dtype 可通过 `mixed_precision_reduce_dtype` 单独控制。

### 与 FSDP 构造参数的对应关系

`TrainConfig` 的字段对齐 FSDP 构造器的大部分核心入参，便于快速定位：

| FSDP `__init__` 参数 | 本项目中对应配置 | 说明 |
| --- | --- | --- |
| `module` | `get_model()` 返回的 HuggingFace 模型 | 先在 CPU 上构造，再交给 FSDP 包装 |
| `sharding_strategy` | `config.sharding_strategy` | 直接调用 `ShardingStrategy(...)` 映射 ZeRO 等级 |
| `cpu_offload` | `CPUOffload(offload_params=config.cpu_offload)` | 仅控制参数是否回落 CPU，梯度/状态依赖分片策略 |
| `auto_wrap_policy` | `config.wrap_policy` + `build_auto_wrap_policy` | 可选 Transformer Block 自动切分或按参数量阈值切分 |
| `backward_prefetch` | `config.backward_prefetch` | 控制反向聚合提前量，配合通信重叠 |
| `mixed_precision` | `get_mixed_precision(config)` | 统一设置参数 dtype、reduce dtype、缓冲区 dtype |
| `forward_prefetch` | `config.forward_prefetch` | 针对 CPU-bound 场景预取下一层聚合 |
| `limit_all_gathers` | `config.limit_all_gathers` | 控制是否限制并发 all-gather，降低峰值显存 |
| `use_orig_params` | 固定启用 (`use_orig_params=True`) | 保留原始 Parameter 句柄，方便优化器与 ckpt 对齐 |
| `device_id` | `device` (local rank) | 通过 `torch.cuda.set_device` + FSDP `device_id` 绑定本地 GPU |

其余如 `ignored_modules`、`sync_module_states` 等参数目前使用 PyTorch 默认值，可按需要扩展到配置文件。

---

## ZeRO 等级与 `ShardingStrategy` 映射

| ZeRO Stage | 对应的 `ShardingStrategy` | 说明 |
| --- | --- | --- |
| Stage 1 | `NO_SHARD` | 只在优化器状态上做分片，梯度和参数完整保留在本地；适合较小模型或高带宽场景。 |
| Stage 2 | `SHARD_GRAD_OP` | 梯度按 rank 分片，参数仍是完整副本；对应 ZeRO-2。 |
| Stage 3 | `FULL_SHARD` | 参数、梯度、优化器状态全部分片；与 ZeRO-3 等价，是默认设置。 |
| Stage 3 + 混合 | `HYBRID_SHARD` | 在节点内全分片、跨节点复制；适合多机多卡时减少跨节点通信。 |

脚本默认使用 `FULL_SHARD`（ZeRO-3），可根据显存/通信能力调整。

---

## CPU Offload 行为

在 `setup_fsdp` 中，当 `cpu_offload=True` 时，构造 `CPUOffload(offload_params=True)`：

- **参数张量**：在不需要 GPU 参与时常驻 CPU，只有前向/反向时才搬运到 GPU。
- **梯度张量**：FSDP 会在反向结束后立即将梯度释放/规约，结合 `FULL_SHARD` 可显著减少显存峰值。
- **优化器状态**：本脚本使用的是标准 `AdamW`，状态默认保留在 GPU。若需进一步节省显存，可结合 `torch.distributed.fsdp.sharded_grad_scale` 或第三方 ZeRO 优化器将状态迁移到 CPU/NVMe。

> 注意：`CPUOffload` 会增加 H2D/D2H 的通信开销，适合 CPU 内存充足、PCIe/NVLink 带宽允许的场景。

### 参数 / 梯度 / 优化器状态的分片与流转

以 `FULL_SHARD` + `use_orig_params=True` 为例，每个进程仅持有对应切片，关键时序如下：

1. **参数切片 (`FlatParameter`)**：初始化时将每个 FSDP 单元内的参数展平为单个张量并分片。常驻设备取决于 `cpu_offload`：为真时切片驻留 CPU，需要计算时再 `all_gather` 到 GPU。
2. **前向前聚合**：执行前向前，FSDP 会触发 `all_gather` 将本层参数片段还原成完整视图，在 GPU 上执行实际计算。
3. **反向后 `reduce_scatter`**：反向结束，梯度被规约并按 rank 分片；如果 `sharding_strategy=SHARD_GRAD_OP`，梯度分片而参数保留全量。
4. **优化器更新**：
   - `FULL_SHARD`：优化器仅看见本 rank 的参数切片，`adam.step()` 在每个切片上执行，并在需要时广播更新结果。
   - `NO_SHARD`：参数、梯度均保留全量，优化器逻辑与单机一致。
   - `HYBRID_SHARD`：节点内做 `FULL_SHARD`，节点间复制；梯度规约与复制分阶段完成。
5. **状态张量**：
   - 标准 `AdamW` 时，动量/二阶矩仍在 GPU，受 `sharding_strategy` 影响：`FULL_SHARD` 分片、`SHARD_GRAD_OP`/`NO_SHARD` 为全量。
   - 若需进一步节省内存，可自定义 ZeRO/DeepSpeed 优化器或结合 `torch.distributed.fsdp.sharded_grad_scale` 将状态落盘。

---

## 代码结构与关键步骤

`fsdp_baseline.py` 主要逻辑如下：

1. **配置解析**：通过 `TrainConfig` + 可选 JSON，构造训练与 FSDP 参数。
2. **分布式初始化**：`init_distributed` 设置进程组、local rank、CUDA 设备等。
3. **数据构造**：`DummyDataset` 生成随机 token，`DistributedSampler` 确保样本分到各个 rank。
4. **模型加载**：优先尝试 `AutoLigerKernelForCausalLM`（若可用），否则退回原生 `AutoModelForCausalLM`。模型初始放在 CPU。
5. **FSDP 封装**：`setup_fsdp` 依据配置设置 `ShardingStrategy`、`auto_wrap_policy`、`MixedPrecision`、`CPUOffload` 等。
6. **训练循环**：
   - 前向：FSDP 封装后直接调用模型，loss 为 rank 本地的标量。
   - 反向：`loss.backward()` 自动触发 FSDP 的通信和张量释放；可选梯度裁剪。
   - 优化：标准 `AdamW` 更新。
   - 统计：rank0 输出 tokens/s、TFLOPS、loss。
7. **评估与保存（可选）**：若启用 `do_eval` 将在验证集上计算 loss，`save_final` 时通过 `StateDictType.FULL_STATE_DICT` 聚合并保存全参数。

> 额外：脚本会在 rank0 写出 `resolved_config.json`（若指定 `--save-config`），方便记录实验设置。

### FSDP 内部执行流程（逐批次）

1. **Prefetch / All-gather**：根据 `forward_prefetch` 和第一轮记录的执行顺序，预拉取下一层需要的参数切片。
2. **Forward**：各层依次执行，期间可能触发更多 all-gather；当 `limit_all_gathers=True` 时保持两层窗口，避免内存过载。
3. **Backward**：自动调用 `reduce_scatter` 将梯度发送给拥有对应切片的 rank；若 `backward_prefetch` 为 `BACKWARD_PRE`，前一层的 all-gather 会在上一层 backward 过程中提前触发。
4. **Optimizer Step**：完成梯度同步后，由 `AdamW` 对本地切片执行更新；`use_orig_params=True` 确保优化器操作的是原始 Parameter。
5. **Post-backward Cleanup**：FSDP 释放临时的全量视图，保留分片后的参数/梯度，以控制显存峰值。

### 内部关键组件速览

- **`FlatParameter`**：FSDP 将所管理模块的 Parameter 展平为单张量以提升通信效率，`use_orig_params=True` 时会维护视图映射。
- **`ShardingStrategy` & ProcessGroup**：定义参数/梯度/状态在各 rank 之间的分布方式。`HYBRID_SHARD` 会创建两套通信组。
- **`Prefetcher` / Rate Limiter**：配合 `forward_prefetch`、`backward_prefetch`、`limit_all_gathers` 控制 all-gather 次序与数量。
- **`StateDictType`**：决定检查点导出的张量形态（全量、分片、局部）；脚本使用 `FULL_STATE_DICT` 聚合权重便于单机推理。
- **`MixedPrecision`**：封装参数 dtype、grad reduce dtype、buffer dtype 的统一策略，避免手动管理。
- **`CPUOffload`**：若启用则负责参数搬运与生命周期管理，配合 `limit_all_gathers` 限制峰值显存。

### FAQ：FSDP 单元粒度 / 分片 / Offload 记忆要点

- **一个 FSDP 单元的默认粒度是什么？**
   - FSDP 只会对 "被包裹的模块" 做 flatten + sharding。若你直接 `FSDP(model)` 且没有 `auto_wrap_policy`，整个模型只产生一个 FlatParameter。
   - 当指定 `transformer_auto_wrap_policy` 或自定义策略时，每个被 wrap 的子模块（常见做法是每个 Decoder Layer）都会成为独立 FSDP 单元，拥有自己的 FlatParameter。
   - 因此，实际训练中通常会启用 auto wrap；否则即便开了 `cpu_offload`，每个 rank 仍掌握 1/N 的模型份额，all-gather 的峰值显存仍接近全量，内存收益有限。

- **FlatParameter 如何和 ShardingStrategy 协同？**
   - `FULL_SHARD`：参数、梯度、优化器状态全都分片，每个 rank 只常驻一段 FlatParameter 切片，前向时 `all_gather`，反向后 `reduce_scatter`。
   - `SHARD_GRAD_OP`：只有梯度分片，参数保留全量；FlatParameter 仍存在，但 gradient reduce-scatter 后保留切片。
   - `NO_SHARD`：不分片，相当于传统 DDP。
   - `HYBRID_SHARD`：节点内 `FULL_SHARD`，节点间复制；需要额外的通信组。

- **是否 “一个 Layer 有多个分片”？**
   - 若 wrap 粒度是单层，则该 Layer 的参数被展平为一条 FlatParameter，并按 `ShardingStrategy` 切成多个分片分布在不同 rank；因此 “同一层的参数” 在不同 rank 上会有不同切片。
   - 若你未按层 wrap，而是一个巨大的 FSDP 单元，则切片只在整体 FlatParameter 上进行，不再区分层。
   - `transformer_auto_wrap_policy` 默认会递归遍历子模块，并在遇到匹配的层类型（如 `LlamaDecoderLayer`）时包裹；不会深入到更底层的注意力子模块，除非你额外在策略中声明那些子类。
   - 嵌入层、最终 `lm_head` 若未被策略匹配，则依旧由外层 FSDP 单元（通常是顶层）统一管理。若需要为它们单独建 FSDP 单元，可在策略里显式列出相应类名。

- **启用 CPU Offload 时，CPU 内存是什么结构？**
   - 仍旧是 FlatParameter 的切片驻留在 CPU（常是 pinned memory），而不是 "每层一个未切分的副本"。
   - FSDP 根据需要将这些切片 `all_gather` 到 GPU 执行前向，结束后释放 GPU 缓冲，只保留 CPU 上的切片；梯度同理通过 `reduce_scatter` 落回。

- **为什么 `use_orig_params=True` 很重要？**
   - 它保证优化器、checkpoint、第三方工具依然看到原始 Parameter 的句柄。FSDP 在内部维护原始 tensor 的视图映射，但真正计算是在 FlatParameter 切片上完成的。

- **CPU offload 与参数预取是怎么协同的？**
   - FSDP 的 prefetch 逻辑由 `forward_prefetch`、`backward_prefetch` 和内部的 `PrefetchPolicy` 控制：在每轮迭代的执行顺序被记录后，下一轮会在前向/后向过程中按顺序提前触发 `all_gather`。
   - 若启用 CPU offload，调用 `all_gather` 时会把下一个 FSDP 单元所需的切片从 CPU 拉回 GPU，配合 `limit_all_gathers` 维持 1~2 层的窗口，避免同时聚合太多层。
   - 当某个单元执行完后，其参数视图会立刻释放，保留 CPU 切片以待下次使用，从而实现单层活跃、其它层驻留 CPU 的流水线。

---

## 拓展建议

- **混合精度 Reduce**：当跨节点通信瓶颈明显时，可设置 `mixed_precision_reduce_dtype="fp16"` 以降低梯度 AllReduce 负载。
- **Activation Checkpointing**：可在 `TrainConfig.activation_checkpointing=True` 后，自行在模型中注册 checkpoint（脚本提供布尔开关，实际注册需在模型代码中实现）。
- **NVMe / CPU 优化器状态卸载**：结合 DeepSpeed ZeRO-3 Offload 或 PyTorch 未来的 FSDP 优化器 sharding 支持，可进一步降低显存压力。

如需对照 SlideFormer 的按层卸载实现，可一边阅读 `fsdp_baseline.py`，一边比较 `slide_fsdp.py` 中的缓冲区/线程池设计，逐步替换为 FSDP 自带的 sharding 策略与通信接口。

## CPU/GPU内存分析
先看你当前的组合：`sharding_strategy="full_shard"` + `cpu_offload=True`，并且 FSDP 会把模型按 Transformer block 自动切成一个个单元。这里的行为可以分成几个阶段来理解。

## GPU 显存里的内容

- **参数**：默认常驻 CPU，只有当某个 FSDP 单元即将参与前向或反向时，才会把该单元的参数切片 `all_gather` 到当前 rank 的 GPU 上；执行完这一层后就会释放 GPU 上的全量视图。因为是 per-layer wrap，所以同一时刻 GPU 只需要容纳 1~2 个邻近层的参数（受 `limit_all_gathers` 控制）。
- **梯度**：反向传播每个 FSDP 单元时，梯度是按 rank 分片的，立即走 `reduce_scatter`，所以梯度的常驻副本也只是 1/4（world_size=4）的切片，而且在步内只保存当前层的切片，随即转入下一层。
- **优化器状态**：标准 `AdamW` 不会自动挪到 CPU；在 `full_shard` 下它同样会被分片，每张卡只保持自己那份动量与平方梯度切片。这部分是训练过程中 GPU 显存的主要持久占用之一。
- **激活/临时 Buffer**：正向和反向中间结果还是在 GPU 上，它们和 `forward_prefetch`/activation checkpointing 的配置有关，与参数是否 offload 没直接冲突。

所以 GPU 上真正长期驻留的只有：当前处理层的参数视图（瞬时）、对应层的梯度切片（瞬时）、以及该 rank 的优化器状态切片（持久）。其余大部分参数都在 CPU。

## CPU 内存里的内容

- **参数切片**：`CPUOffload(offload_params=True)` 会让每个 FSDP 单元的参数切片常驻在 CPU（通常是 pinned memory）。每个 rank 保存自己负责的 1/4 切片，并不会存储整层的完整副本。
- **梯度 / 优化器状态**：梯度在 `full_shard` + offload 下并不会被长期放在 CPU；它们在 GPU 上完成 `reduce_scatter` 后就用于优化器更新。标准 `AdamW` 的状态默认仍留在 GPU。
- **数据加载 & tokenizer**：DummyDataset 生成的批次、tokenizer 等常规对象也会占用一些 CPU 内存。

换句话说，`cpu_offload=True` 只把 **参数切片** 挪到了 CPU，梯度和优化器状态仍主要消耗 GPU 显存。每个 rank 的内存/显存占用情况大致是：

- **CPU**：各层参数切片（1/4 模型），加上数据/loader/运行时开销。
- **GPU**：当前激活层的参数全量视图（瞬时）、梯度切片（1/4 模型，瞬时）、优化器状态切片（1/4 模型，常驻），以及激活值和通信 buffer。

如果你想进一步减轻 GPU 压力，需要：

- 把优化器状态也卸载（例如换用 DeepSpeed ZeRO-Offload 或 PyTorch 将来的 FSDP 优化器 offload 支持）。
- 或者选择 `NO_SHARD`/`SHARD_GRAD_OP` 在配套的场景下减少通信，代价是显存负担更重。

目前的配置能比较好地让参数常驻 CPU、按层搬运，显存压力主要来自激活 + 优化器状态 + 一小段参数视图。

先粗算一下整模的体量，再分摊到 4 张卡上：  
- Llama‑3.1‑8B 约等于 8×10⁹ 个参数。  
- 你在配置里用 `bf16`，也就是每个参数 2 字节。  

基于这一点，可以得到下面的估算（四卡 `full_shard + cpu_offload=True`）。

### GPU 显存（每张卡）
- **参数瞬时视图**：一次只聚合当前/下一个 FSDP 单元的参数，大约不到 1 GB（2 × 1/4 × 8B × 2 B，考虑到 `limit_all_gathers` 限制窗口为 2 层，并且只在短时间内驻留）。  
- **梯度切片（临时）**：约 4 GB（8B × 2 B / 4），不过是在反向的局部窗口内出现，很快被规约和释放。  
- **AdamW 优化器状态（常驻）**：约 16 GB。默认实现把动量和二阶矩保存在 FP32（4 B）里，各有一份；`full_shard` 下每个 rank 只持有 1/4，所以 8B × 2 × 4 B / 4 ≈ 16 GB。  
- **激活/中间缓冲**：取决于 batch/seq，粗略估计 4–6 GB。  

合计显存峰值通常在 20 GB 左右，再加上激活视窗，常见使用量大概 22–24 GB；如果 batch/seq 再调大，则按比例增加。

### CPU 内存（每张卡）
- **参数切片（常驻）**：`cpu_offload` 会让每个 rank 持有自己 1/4 的参数，在 bf16 下约 4 GB。  
- **Pinned 缓冲 & 通信开销**：为搬运准备的 pinned memory、Dataset 等，会再增加 1–2 GB。  
- **全局额外开销**：所有 rank 加起来，CPU 得腾出 ≈16 GB 给参数切片，再加上数据/运行时的一些额外占用。

### 额外说明
- 如果改用 `fp16` 或 `fp32`，参数/梯度/优化器状态都会按字节数线性变化。  
- 想进一步压低 GPU 常驻，可以考虑：  
  - 把 AdamW 状态也卸载（例如接入 DeepSpeed ZeRO‑Offload）。  
  - 开启激活重算（`activation_checkpointing=True` 已可配）。  
- 以上都是估算，真实占用还会受激活窗口、通信缓冲（NCCL）、日志等影响；建议用 `torch.cuda.memory_summary()` 或 `torch.distributed._tensor` 的 profiling 工具做一次采样来确认。

`limit_all_gathers` 控制了 FSDP 在前向/反向阶段对参数的 `all_gather` 使用方式，可以理解为“通信节流阀”。区别主要在于它如何安排多个 FSDP 单元的参数聚合，以及何时释放这些全量视图。

---
## limit_all_gathers 
### 关闭时（默认 False）

- **机制**：FSDP 在执行每个子模块前，会直接触发该单元的参数 `all_gather`，并把全量张量暂存到 GPU。  
- **窗口大小**：没有硬性限制，可能一次性聚合多个后续层的参数；尤其是前向时，为了减少等待，有时会预先把后面几层的参数准备好。  
- **后果**：通信更激进，适合高带宽场景，但在显存紧张时，多个层的参数全量视图可能同时驻留，从而抬高峰值显存占用。  
- **释放时机**：通常在该 FSDP 单元的前向/反向结束后才释放，若提前聚合了后面几层，就会延迟释放，临时占用显存。

---

### 开启时（True）

- **机制**：FSDP 会显式维护一个“聚合窗口”——只允许当前层和紧挨着的下一层（或下几层，取决于内部策略）同时持有全量参数。  
- **窗口大小**：PyTorch 默认是 1~2 层。当前层执行时，下一层的 `all_gather` 可以在后台启动，但不会再提前更多层；一旦当前层完成，就立刻释放它的全量视图，把窗口腾出来给下一层。  
- **后果**：显存占用更可控，适合启用 CPU offload 或显存本就有限的环境；代价是通信和计算叠加的“隐藏”机会少一些，对带宽较敏感的场景可能略微放慢速度。  
- **释放时机**：窗口内的层执行完即释放，未进入窗口的层不会提前聚合，确保任何时刻只持有很少的全量参数。

---

### 何时选择

| 场景 | 建议 |
| --- | --- |
| 显存吃紧（如 CPU offload、分片层很多） | 开启 `limit_all_gathers=True`，保证峰值受控 |
| 带宽很高、追求极致吞吐 | 可保持 `False`，让 FSDP 自行预取多层参数 |
| 观察到显存时常爆在 `all_gather` 期间 | 尝试开启，结合 `forward_prefetch` 调整通信与计算重叠 |

---

### 小结

- **开启** → 严格的“逐层”流水线：一次最多 1~2 个 FSDP 单元的参数在 GPU，全量视图用完即释放。  
- **关闭** → 更松弛的预取策略：可能提前把后面的层都聚合好，换取通信重叠，但峰值显存更高。  

如果你希望验证，可以用 `TORCH_SHOW_CPP_STACKTRACES=1` 加上内存统计（例如 `torch.cuda.memory_summary()`）在训练过程中对比两种模式的显存曲线，会看出窗口控制的差异。

我先给你看重点：`limit_all_gathers` 这个布尔值在 `_runtime_utils.py` 里贯穿了多个环节，主要用来“节流”参数 all‑gather，让同一时间段里驻留在 GPU 上的全量参数数量保持在一个很小的窗口。

### 1. 统一设置与传播
- 在 `_share_state_and_init_handle_attrs()` 里面，它被列入 `HOMOGENEOUS_ATTR_NAMES`，意味着所有嵌套的 FSDP state 都要共享同一个值；一旦根实例设成 `True`，整个 FSDP 树都会收到，用于后续逻辑判断。

### 2. 流优先级调整
- `_init_streams()` 会根据这个值调高通信相关 CUDA stream 的优先级（尤其是在混合分片 HSDP 情况下）。设置成 `True`（并且使用 HYBRID_SHARD）时，all-gather / reduce-scatter 会拿到更高优先级，避免被其它异步操作阻塞。这是第一道限流措施，确保参数聚合能及时完成。

### 3. 运行时限流的核心：`_unshard()` 与 `_reshard()`
- `limit_all_gathers=True` 时，`_unshard()` 在真正做 all-gather 前，会从 `state._free_event_queue` 里取出一个 `Event`（这个队列相当于“窗口计数器”）；如果没有可用的 event，就会触发 `FullyShardedDataParallel.rate_limiter` 这段 profiler 标记，其内部会阻塞等待，直到已有的某个 all-gather 完成并释放 event。
- 对应地，`_reshard()` 在完成参数释放后，会生成新的 `Event` 并放回队列——这就是窗口“归还”。这样一来，最多只有固定数量的 FSDP handle 能同时处于“参数已解片”状态，从而限制显存峰值。
- 当 `limit_all_gathers=False` 时，这些 event 操作统统跳过；每个 handle 想 `unshard()` 就直接走，导致可能同时堆积多个全量参数视图，换来更积极的流水线但显存占用也更高。

### 4. Prefetch / 执行序列
- `_prefetch_handle()`、`_pre_forward_unshard()` 等函数会在启用限流时调用上述 `_unshard`，所以预取操作同样受窗口约束：即使 forward/backward 想提前多解几个 handle，也必须等到窗口有“名额”。
- `_root_pre_forward()` 里也会在执行预取前同步控制流，确保前向的 unshard 顺序遵守窗口策略。

### 5. 梯度阶段
- `_post_backward_hook` → `_post_backward_reshard` 的路径在释放梯度后也会调用 `_reshard()`。如果 `limit_all_gathers=True`，它会把释放后的 event 放回 `_free_event_queue`，让下一个 handle 的 all-gather 得以推进。这保证了即便在反向阶段，窗口大小也不会被突破。

---

### 总结两种模式

| 模式 | all-gather 行为 | 显存消耗 | 适用场景 |
| --- | --- | --- | --- |
| `limit_all_gathers=False` | 想聚合多少层都不拦，all-gather 可以堆在一起 | 峰值高，但通信重叠更多 | 带宽足、显存宽裕时追求速度 |
| `limit_all_gathers=True` | 通过 event 队列把窗口锁在 1~2 层（取决于内部策略） | 峰值受控、层间串行更明显 | 显存吃紧、配合 CPU offload 或多层 wrap |

如果你想进一步跟踪，可以用 profiler 搜 `FullyShardedDataParallel.rate_limiter`，或者在 `_unshard`、`_reshard` 附近插日志，就能看到窗口的开启/释放顺序。
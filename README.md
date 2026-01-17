# SlideFSDP

SlideFSDP 是在 PyTorch 分布式训练基础上复刻 SlideFormer 单卡按层卸载思路的多 GPU 实现。核心目标是在多卡场景下仍然做到逐层 H2D 预取、逐层 D2H 卸载，以及梯度与 CPU Adam 更新的重叠执行。

## 主要特性

- **逐层参数流动**：通过 `LayerBufferPool` 复用 GPU bf16 参数缓存与 CPU 梯度缓存，只保留当前执行层在 GPU 上
- **可配置的参数传输策略**：
  - `full`：每 GPU 加载完整层参数（适合高带宽场景）
  - `shard`：H2D 分片 + GPU AllGather（节省 CPU→GPU 带宽）
- **可配置的梯度同步策略**：
  - `gpu_reduce`：GPU ReduceScatter → D2H 分片（节省 D2H 带宽）
  - `cpu_reduce`：D2H 全量 → CPU AllReduce（节省 GPU 显存）
- **梯度与参数更新重叠**：在 backward hook 中将梯度搬运回 CPU 并立即调度 LayerAdam 更新
- **内置 AVX512 优化的 CPU Adam**：LayerAdam 实现支持 FP32/FP16/BF16 参数和优化器状态
- **激活卸载支持**：内置 `SlidingCheckpoint` 支持激活值 CPU/NVMe 卸载

## 文件结构

```
SSDP/
├── config.py                    # 配置类 SlideFSDPConfig
├── layer_state.py               # LayerBufferPool + LayerRuntimeState
├── distributed_sync.py          # 可配置的参数/梯度同步策略
├── shared_memory.py             # 跨进程共享内存管理（单优化器模式）
├── slide_fsdp.py                # 核心包装器 SlideFSDP
├── sliding_checkpoint.py        # 激活检查点滑动窗口实现
├── train.py                     # 多卡训练示例脚本
├── optimizer/                   # 优化器模块
└── utils/                       # 功能方法
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch transformers

# （可选）如需 NVMe 激活卸载
pip install tensornvme kvikio
```

### 2. 训练示例

```bash
# 使用默认配置（4 GPU）
torchrun --nproc_per_node=4 train.py \
    --model-path /path/to/llama-3.1-8b

# 指定传输策略
torchrun --nproc_per_node=4 train.py \
    --model-path /path/to/model \
    --param-transfer-mode shard \
    --grad-sync-mode cpu_reduce \
    --batch-size 4 \
    --seq-length 1024
```

### 2.1 NUMA 亲和 + 线程数

在 torchrun 下默认 OMP 线程可能被限制为 1。使用基于 numactl 的启动脚本为每个 rank 绑定对应 NUMA 节点并配置线程数（默认每个 rank 预留 1 个“控制核”，其余用于 OMP）：

```bash
chmod +x numa_launch.sh

# 示例：单机 4 卡，NUMA 拓扑 2 节点，各 32c64t（可根据需要调整）
torchrun --nproc_per_node=4 --nnodes=1 --no_python ./numa_launch.sh -- \
    train.py --model-path /path/to/model

# 覆盖控制核数量或线程数（可选）
CONTROLLER_CORES_PER_RANK=2 THREADS_PER_RANK=56 \
torchrun --nproc_per_node=4 --no_python ./numa_launch.sh -- train.py --model-path /path/to/model

# 如需手动指定每个 NUMA 的 CPU 列表（空格或逗号分隔），用于不规则拓扑
CPU_LIST_NODE0="0-31,64-95" CPU_LIST_NODE1="32-63,96-127" \
torchrun --nproc_per_node=4 --no_python ./numa_launch.sh -- train.py --model-path /path/to/model
```

环境变量说明：
- `CONTROLLER_CORES_PER_RANK`：在分配给该 rank 的 CPU 段前面预留的核心数（默认 4），其余用于 OMP/MKL。
- `THREADS_PER_RANK`：强制 OMP/MKL 线程数；缺省时为“分配段减去预留核数”。
- `CPU_LIST_NODE{N}`：手动覆盖第 N 个 NUMA 节点的 CPU 列表。
- `PYTHON_BIN`：自定义 Python 可执行文件。
- `NUMA_LAUNCH_DEBUG=1`：打印每个 rank 的绑定详情。

### 3. 代码集成

```python
import torch
from config import SlideFSDPConfig
from slide_fsdp import SlideFSDP

# 初始化分布式
torch.distributed.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)

# 加载模型
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("model_path", device_map="cpu")

# 配置 SlideFSDP
config = SlideFSDPConfig(
    dtype=torch.bfloat16,
    device=device,
    param_transfer_mode="full",    # 或 "shard"
    grad_sync_mode="gpu_reduce",   # 或 "cpu_reduce"
    lr=1e-5,
    enable_activation_checkpoint=True,
)

# 包装模型
slide_model = SlideFSDP(model, config=config)

# 训练循环
for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = slide_model(**batch)  # 自动执行 backward
    # 无需调用 optimizer.step()，LayerAdam 自动更新
```

## 传输策略选择指南

| 场景 | param_transfer_mode | grad_sync_mode | 理由 |
|------|---------------------|----------------|------|
| 高带宽 PCIe/NVLink | `full` | `gpu_reduce` | 最大化 GPU 利用率 |
| 大模型 + 有限显存 | `shard` | `cpu_reduce` | 节省 GPU 显存 |
| CPU 内存充足 | `full` | `cpu_reduce` | 减少 GPU 通信开销 |
| 多节点训练 | `shard` | `gpu_reduce` | 利用 NVLink 卡间通信 |

## 工作机制概览

```
Forward Pass:
┌─────────┐    H2D     ┌─────────┐   Compute   ┌─────────┐
│  CPU    │ ─────────> │  GPU    │ ──────────> │ Output  │
│ FP32    │  Prefetch  │  BF16   │   Layer N   │         │
└─────────┘            └─────────┘             └─────────┘
                           │
                           │ D2H Offload
                           ▼
                    Next Layer Prefetch

Backward Pass:
┌─────────┐    D2H     ┌─────────┐  AllReduce  ┌─────────┐
│  GPU    │ ─────────> │  CPU    │ ──────────> │  Adam   │
│ Grads   │   Sync     │ Grads   │   (Gloo)    │ Update  │
└─────────┘            └─────────┘             └─────────┘
```

## 配置参数说明

```python
@dataclass
class SlideFSDPConfig:
    # 基础配置
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = None  # 自动检测 LOCAL_RANK
    
    # 传输策略（核心配置）
    param_transfer_mode: str = "full"    # "full" | "shard"
    grad_sync_mode: str = "gpu_reduce"   # "gpu_reduce" | "cpu_reduce"
    
    # 优化器
    lr: float = 1e-5
    weight_decay: float = 0.01
    adam_betas: tuple = (0.9, 0.999)
    
    # 运行时
    gpu_buffer_pool_size: int = 2  # 滑动窗口缓存池大小
    max_workers: int = 2           # 后台线程数
    enable_activation_checkpoint: bool = True
    ac_offload_mode: str = "cpu"   # "cpu" | "nvme" | "none"
```

## CPU Adam 优化器

内置的 `LayerAdam` 支持：
- **AVX512/AVX2 SIMD 加速**：自动检测 CPU 特性并启用向量化
- **多精度支持**：FP32、FP16、BF16 参数和优化器状态
- **逐层更新**：可单独更新指定层，配合流水线执行
- **NVMe 状态卸载**：（可选）将优化器状态卸载到 NVMe

```python
from optimizer import LayerAdam

optimizer = LayerAdam(
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    adamw_mode=True,
    fp32_optimizer_state=True,
    num_layer=32,
)

# 逐层添加参数
for idx, layer in enumerate(model.layers):
    optimizer.add_layer_params(idx, layer.parameters())

# 逐层更新
optimizer.step(layer_idx=0)
```

## 激活检查点

内置的 `SlidingCheckpoint` 支持：
- **CPU 内存卸载**：将激活值异步卸载到 pinned memory
- **NVMe 卸载**：（需要 kvikio）直接卸载到 NVMe 存储
- **预取优化**：在 backward 时提前预取下一层激活值

```python
from sliding_checkpoint import SlidingCheckpoint

# 在 checkpoint 中使用
with SlidingCheckpoint(
    layer_idx=layer_idx,
    layer_tensors=layer_tensors,  # 预分配的 CPU tensors
    is_last_layer=(layer_idx == num_layers - 1),
    device=str(device),
):
    output = checkpoint(layer_forward, hidden_states, ...)
```

## TODO / 后续扩展

- [ ] 完善 NVMe 激活卸载集成
- [ ] 支持更多模型架构（Qwen、Mistral 等）
- [ ] 性能计时与 profiling 工具
- [ ] 与 DeepSpeed ZeRO 策略的兼容性测试
- [ ] 多节点分布式训练验证

## 致谢

CPU Adam 优化器的 C++ 实现基于 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 项目。

## License

MIT License

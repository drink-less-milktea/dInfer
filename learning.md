# dInfer 框架系统学习路线

> 本文档提供一个循序渐进的学习路线，帮助你深入理解 dInfer 扩散语言模型（Diffusion Language Models, dLLMs）推理框架。

---

## 📚 目录

1. [框架概览](#框架概览)
2. [前置知识](#前置知识)
3. [项目结构详解](#项目结构详解)
4. [核心概念](#核心概念)
5. [学习路线](#学习路线)
6. [实践项目](#实践项目)
7. [进阶主题](#进阶主题)
8. [常见问题](#常见问题)

---

## 🎯 框架概览

### dInfer 是什么？

dInfer 是一个高效、可扩展的**扩散语言模型（Diffusion Language Models, dLLMs）**推理框架，由 inclusionAI 开发。与传统的自回归语言模型不同，扩散语言模型采用迭代去噪的方式生成文本，可以实现**并行解码**，大幅提升推理速度。

### 核心特点

- **模块化设计**: 将推理过程分解为 4 个独立组件，便于算法组合和扩展
- **多种算法**: 支持多种解码策略（Threshold、Hierarchy、Credit）和缓存机制（Prefix、Dual、Vicinity）
- **高性能优化**: 
  - 张量并行（TP）和专家并行（EP）
  - 动态批处理（Dynamic Batching）
  - PyTorch 编译和 CUDA Graphs
  - 循环展开（Loop Unrolling）消除 CUDA 流气泡
- **支持多模型**: LLaDA、LLaDA-MoE、LLaDA2（包括 mini 和 flash 版本）
- **高吞吐量**: 在 HumanEval 上单样本超过 1100 TPS，比 Fast-dLLM 快 10 倍

### 四大核心组件

```
┌────────────────────────────────────────────────────┐
│              dInfer 架构设计                        │
├────────────────────────────────────────────────────┤
│  1. Model (模型层)                                 │
│     • LLaDA: 8B 参数的扩散语言模型                 │
│     • LLaDA-MoE: 7B 参数的 MoE 扩散模型            │
│     • LLaDA2: 16B-100B 的块扩散模型                │
│                                                     │
│  2. Diffusion Iteration Manager (迭代管理器)       │
│     • BlockWise: 块级逐步生成                      │
│     • IterSmooth: 迭代平滑策略                     │
│     • Vicinity: 邻近窗口缓存更新                   │
│     • BlockDiffusion: 块扩散（仅 LLaDA2）          │
│                                                     │
│  3. Decoder (并行解码器)                           │
│     • Threshold: 基于置信度阈值解码                │
│     • Hierarchy: 层次化解码（每段选最优）          │
│     • Credit: 信用加权的阈值解码                   │
│                                                     │
│  4. KV-Cache Manager (缓存管理器)                  │
│     • Prefix Cache: 仅缓存前缀                     │
│     • Dual Cache: 双缓存优化                       │
│     • Vicinity Refresh: 邻近窗口刷新策略           │
└────────────────────────────────────────────────────┘
```

### 与传统 LLM 的区别

| 特性 | 自回归 LLM (如 GPT) | 扩散 LLM (dInfer) |
|------|---------------------|-------------------|
| 生成方式 | 逐个 token 串行生成 | 并行迭代去噪生成 |
| 解码速度 | 受序列长度线性限制 | 通过并行解码加速 |
| 初始状态 | 从前缀开始 | 全部为 mask token |
| 迭代次数 | O(N) 次前向传播 | O(sqrt(N)) 次扩散迭代 |
| KV-Cache | 逐步累积 | 需要特殊管理策略 |

---

## 📖 前置知识

### 必备知识 ⭐⭐⭐

1. **Python 编程**
   - 面向对象编程（类、继承、多态）
   - 装饰器（`@torch.no_grad()`, `@torch.compile()`）
   - 上下文管理器（`with` 语句）
   - 多进程/多线程编程

2. **PyTorch 深度学习框架**
   - 张量操作（`torch.Tensor`）
   - 自动求导机制
   - 模型定义与前向传播
   - CUDA 编程基础（`.to(device)`, `torch.cuda.set_device()`）
   - 分布式训练基础（`torch.distributed`）

3. **Transformer 架构**
   - 自注意力机制（Self-Attention）
   - 多头注意力（Multi-Head Attention）
   - 位置编码（Position Encoding）
   - Feed-Forward Networks
   - **KV-Cache 原理**（重要！）

### 推荐预习 ⭐⭐

4. **扩散模型基础** (非常重要!)
   - 扩散过程（前向加噪过程）
   - 去噪过程（反向生成过程）
   - 噪声调度策略（Noise Schedule）
   - 推荐论文:
     - DDPM (Denoising Diffusion Probabilistic Models)
     - DDIM (Denoising Diffusion Implicit Models)
     - **LLaDA 论文**: https://arxiv.org/abs/2510.08666

5. **混合专家模型（MoE）**
   - 专家路由机制（Router）
   - Top-K 选择策略
   - 负载均衡
   - Expert Parallel 并行策略

6. **分布式推理**
   - Data Parallel (DP): 数据并行
   - Tensor Parallel (TP): 张量并行
   - Pipeline Parallel (PP): 流水线并行
   - Expert Parallel (EP): 专家并行（MoE 特有）

### 可选但推荐 ⭐

7. **vLLM 框架**
   - PagedAttention 机制
   - 连续批处理（Continuous Batching）
   - 模型并行策略
   - dInfer 基于 vLLM v0.10.2 构建

8. **HuggingFace 生态**
   - Transformers 库
   - Model Hub 使用
   - `lm-eval-harness` 评估框架

---

## 📁 项目结构详解

```
dInfer/
│
├── python/dinfer/              # 核心代码库
│   ├── __init__.py            # 对外 API 入口
│   │   # 导出: ThresholdParallelDecoder, HierarchyDecoder, 
│   │   #       BlockWiseDiffusionLLM, KVCacheFactory 等
│   │
│   ├── model/                 # 模型实现模块
│   │   ├── __init__.py                  # 导出模型类
│   │   ├── modeling_llada.py            # LLaDA 8B 模型实现
│   │   ├── modeling_fused_olmoe.py      # LLaDA-MoE 7B 模型（融合版）
│   │   ├── modeling_llada2_moe.py       # LLaDA2 模型（16B-100B）
│   │   ├── modeling_llada2_moe_sglang.py # LLaDA2 SGLang 版本
│   │   ├── modeling_llada_fastdllm.py   # Fast-dLLM 实现（对比用）
│   │   ├── configuration_llada.py       # LLaDA 配置类
│   │   ├── configuration_olmoe.py       # OLMoE 配置类
│   │   ├── configuration_llada2_moe.py  # LLaDA2 配置类
│   │   ├── configuration_bailing_moe_v2.py # Bailing MoE 配置
│   │   └── tp_linear.py                 # 张量并行线性层实现
│   │
│   └── decoding/              # 解码逻辑模块（核心！）
│       ├── __init__.py                   # 导出解码器和生成类
│       ├── utils.py                      # 工具类（TokenArray, KVCache, 迭代器）
│       ├── parallel_strategy.py          # 并行解码策略实现
│       │   # - ThresholdParallelDecoder: 阈值解码
│       │   # - CreditThresholdParallelDecoder: 信用解码
│       │   # - HierarchyDecoder: 层次化解码
│       │
│       ├── generate_uniform.py           # 主推理逻辑（最重要！）
│       │   # - DiffusionLLM: 基类
│       │   # - BlockWiseDiffusionLLM: 块级扩散
│       │   # - IterSmoothDiffusionLLM: 迭代平滑
│       │   # - VicinityCacheDiffusionLLM: 邻近缓存
│       │   # - BlockDiffusionLLM: 块扩散（LLaDA2）
│       │
│       ├── generate_fastdllm.py          # Fast-dLLM 实现
│       ├── generate_hierarchy.py         # 层次化生成
│       ├── generate_merge.py             # 合并生成策略
│       ├── generate_dist.py              # 分布式生成（序列并行）
│       ├── generate_cache.py             # 缓存管理实现
│       ├── diffusion_runner.py           # 扩散迭代执行器
│       └── serving.py                    # 在线服务接口（实验性）
│
├── tests/                     # 单元测试
│   ├── test_llada.py          # LLaDA 模型测试
│   ├── test_llada_moe.py      # LLaDA-MoE 模型测试
│   ├── test_bd.py             # 块扩散测试
│   ├── test_bd_serving.py     # 块扩散服务测试
│   ├── test_generate.py       # 生成逻辑测试
│   └── test_wo_model.py       # 无模型测试（逻辑验证）
│
├── benchmarks/                # 性能基准测试
│   ├── benchmark.py           # 单样本速度测试
│   ├── benchmark_dataset.py   # 数据集批量测试
│   ├── benchmark_dataset_fastdllm.py  # Fast-dLLM 对比
│   ├── benchmark_dataset_sglang.py    # SGLang 对比
│   └── benchmark_dataset_sorted.py    # 排序批处理测试
│
├── evaluations/               # 模型质量评估
│   ├── eval_dinfer.py         # 评估脚本（基于 lm-eval-harness）
│   ├── eval_guide.md          # 评估使用指南
│   ├── eval_llada_moe.sh      # LLaDA-MoE 评估脚本
│   └── tasks/                 # 自定义评估任务
│       ├── gsm8k_llada/       # GSM8K 数学推理
│       └── mbpp_sanitized_llada/  # MBPP 代码生成
│
├── tools/                     # 工具脚本
│   ├── transfer.py            # 模型转换脚本（转为 FusedMoE）
│   ├── fuse_moe.py            # MoE 融合逻辑
│   ├── configuration_lladamoe.py  # LLaDA-MoE 配置
│   └── modeling_fused_lladamoe.py # FusedMoE 模型实现
│
├── assets/                    # 资源文件（图片、logo）
├── main.py                    # 简单推理示例脚本
├── setup.py                   # 安装配置
├── README.md                  # 项目说明文档
└── LICENSE                    # Apache 2.0 许可证
```

---

## 💡 核心概念

### 1. 扩散语言模型（dLLM）工作原理

#### 传统自回归生成（GPT 风格）
```
输入: "What is the capital"
步骤: 
  1. 预测 -> "of"
  2. 预测 -> "France"
  3. 预测 -> "?"
  4. 预测 -> "\n"
  5. 预测 -> "Paris"
输出: "What is the capital of France?\nParis"
```

#### 扩散模型生成（dInfer 风格）
```
输入: "What is the capital"
初始化: [MASK] [MASK] [MASK] [MASK] [MASK]

迭代 1: [of] [MASK] [MASK] [MASK] [MASK]      # 高置信度 token 先解码
迭代 2: [of] [France] [?] [MASK] [MASK]       # 并行解码多个 token
迭代 3: [of] [France] [?] [\n] [MASK]
迭代 4: [of] [France] [?] [\n] [Paris]        # 最终完成

输出: "What is the capital of France?\nParis"
```

**核心优势**: 在每次迭代中可以**并行解码多个 token**，而不是逐个生成。

### 2. 四大组件详解

#### 组件 1: Model（模型层）

**功能**: 实现扩散语言模型的前向传播逻辑

**支持的模型**:
- **LLaDA**: 8B 参数，基于 Llama 架构改造
- **LLaDA-MoE**: 7B 参数，使用混合专家（MoE）架构
- **LLaDA2**: 16B-100B 参数，支持块扩散机制

**关键文件**:
```python
# python/dinfer/model/modeling_llada.py
class LLaDAModelLM:
    def forward(self, input_ids, positions, kv_caches, ...):
        """
        前向传播，预测每个 MASK 位置的 token 分布
        """
        # 1. Embedding
        # 2. Multi-layer Transformer
        # 3. Output logits
        return logits
```

#### 组件 2: Diffusion Iteration Manager（迭代管理器）

**功能**: 控制扩散迭代的执行流程和优化策略

**主要策略**:
- **BlockWise**: 将生成序列分成多个块，逐块生成
- **IterSmooth**: 迭代平滑，逐步降低阈值以平滑生成
- **Vicinity Cache**: 邻近窗口缓存刷新策略
- **BlockDiffusion**: LLaDA2 的块级扩散（减少计算开销）

**关键类**:
```python
# python/dinfer/decoding/generate_uniform.py
class BlockWiseDiffusionLLM:
    def generate(self, prompt, gen_length, block_length):
        """
        块级扩散生成
        1. 初始化: 创建全 MASK 序列
        2. 迭代: 对每个块执行扩散迭代
        3. 解码: 使用 Decoder 选择高置信度 token
        4. 更新: 更新 KV-Cache
        """
```

#### 组件 3: Decoder（并行解码器）

**功能**: 在每次扩散迭代中，决定哪些 MASK token 应该被解码

**三种解码策略**:

1. **Threshold Decoder（阈值解码）**
```python
class ThresholdParallelDecoder:
    def decode(self, logits, mask_index):
        """
        如果 token 的预测置信度 > threshold，则解码该 token
        
        例如: threshold = 0.9
        - "of" 置信度 0.95 -> 解码 ✓
        - "France" 置信度 0.87 -> 保持 MASK ✗
        """
```

2. **Hierarchy Decoder（层次解码）**
```python
class HierarchyDecoder:
    def decode(self, logits, mask_index):
        """
        将 MASK 序列分段，每段选择置信度最高的 token 解码
        
        优势: 保证每次迭代都有进展，避免卡住
        """
```

3. **Credit Decoder（信用解码）**
```python
class CreditThresholdParallelDecoder:
    def decode(self, logits, mask_index, history):
        """
        基于历史置信度加权，给"表现好"的位置更高的信用
        """
```

#### 组件 4: KV-Cache Manager（缓存管理器）

**功能**: 高效管理 Transformer 的 Key-Value 缓存

**三种缓存策略**:

1. **Prefix Cache（前缀缓存）**
   - 只缓存输入前缀的 KV
   - 适用于固定前缀的场景

2. **Dual Cache（双缓存）**
   - 同时维护两个缓存：当前块缓存 + 历史缓存
   - 在块间切换时合并

3. **Vicinity Refresh（邻近刷新）**
   - 定义一个窗口，只刷新窗口内的 KV-Cache
   - 减少缓存更新开销

**关键类**:
```python
# python/dinfer/decoding/utils.py
class DiffusionKVCacheManager:
    def update_cache(self, block_loc, new_kv):
        """
        根据策略更新 KV-Cache
        """
```

### 3. 工作流程示例

```
用户输入 prompt
    ↓
[初始化阶段]
• TokenArray: 创建 [prompt] + [MASK * gen_length]
• KVCache: 初始化缓存
    ↓
[迭代阶段 - 第一个块]
循环:
  1. Model.forward(input_ids, kv_cache)
      ↓ 输出 logits
  2. Decoder.decode(logits, mask_index)
      ↓ 选择高置信度 token
  3. 更新 input_ids (MASK -> decoded token)
  4. KVCacheManager.update(...)
      ↓ 更新缓存
  直到: 当前块所有 MASK 都被解码
    ↓
[迭代阶段 - 第二个块]
（重复上述过程）
    ↓
[输出阶段]
• 移除 EOS 后的所有 token
• 返回生成结果
```

---

## 🚀 学习路线

### 📅 阶段 0: 环境准备（0.5天）

#### 目标
- 搭建开发环境
- 成功运行第一个推理示例

#### 步骤

**1. 安装 dInfer**
```bash
# 克隆仓库
git clone https://github.com/inclusionAI/dInfer.git
cd dInfer

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install .

# 验证安装
python -c "import dinfer; print(dinfer.__version__)"
```

**2. 下载模型（可选，用于测试）**
```bash
# 安装下载工具
pip install -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# 下载 LLaDA-MoE 模型（需要先转换为 FusedMoE）
huggingface-cli download inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --local-dir ./models/LLaDA-MoE-7B-A1B-Instruct

# 转换为 FusedMoE
python -m tools.transfer \
  --input ./models/LLaDA-MoE-7B-A1B-Instruct \
  --output ./models/LLaDA-MoE-7B-A1B-Instruct-fused
```

**3. 运行第一个推理**
```bash
# 使用 main.py 示例脚本
python main.py
```

**预期输出**:
```
[1/5] 初始化分布式环境...
使用设备: cuda:0
[2/5] 加载模型: /path/to/model
分词器加载完成
模型加载完成 (bfloat16)
[3/5] 创建扩散语言模型推理器...
解码器: ThresholdParallelDecoder (threshold=0.9)
KV-Cache: Dual Cache
[4/5] 运行推理测试 (gen_length=128, block_length=32)...
================================================================================

【测试 1/3】
提示词: Lily can run 12 kilometers per hour...
--------------------------------------------------------------------------------
生成结果:
Lily can run 12 * 4 = 48 kilometers in 4 hours...
================================================================================
```

---

### 📅 阶段 1: 基础理解（1-2天）

#### 目标
- 理解扩散语言模型的基本原理
- 掌握 dInfer 的整体架构
- 理解四大核心组件的作用

#### 学习材料

**1. 阅读文档**
```bash
# 项目 README
cat README.md

# 评估指南
cat evaluations/eval_guide.md
```

**2. 阅读论文（强烈推荐）**
- **dInfer 技术报告**: https://arxiv.org/abs/2510.08666
- **LLaDA 论文**: 查看 HuggingFace 模型卡片

**3. 理解核心概念**
- 什么是扩散模型？
- 扩散语言模型与传统 LLM 的区别
- 并行解码如何工作？

#### 实践任务

**任务 1: 运行简单推理**
```bash
# 使用 benchmark.py 测试单样本推理
python benchmarks/benchmark.py \
  --model_name your_model_path \
  --model_type llada_moe \
  --gen_len 512 \
  --block_length 32 \
  --gpu 0 \
  --parallel_decoding threshold \
  --threshold 0.9
```

**任务 2: 对比不同解码策略**
```bash
# Threshold 解码
python benchmarks/benchmark.py --parallel_decoding threshold --threshold 0.9

# Hierarchy 解码
python benchmarks/benchmark.py --parallel_decoding hierarchy --threshold 0.9 --low_threshold 0.5
```

观察输出，理解不同策略的差异。

**任务 3: 阅读入口代码**
```bash
# 打开核心入口文件
vim python/dinfer/__init__.py
vim main.py
```

理解 API 设计：
```python
from dinfer import (
    BlockWiseDiffusionLLM,      # 推理引擎
    ThresholdParallelDecoder,   # 解码器
    KVCacheFactory,             # 缓存工厂
    BlockIteratorFactory,       # 迭代器工厂
)
```

---

### 📅 阶段 2: 深入解码逻辑（2-3天）

#### 目标
- 理解扩散迭代的执行流程
- 掌握并行解码策略的实现
- 理解 KV-Cache 管理机制

#### 学习路径

**第 1 天: 工具类和数据结构**

阅读文件: `python/dinfer/decoding/utils.py`

**关键类**:
```python
# 1. TokenArray: 管理生成序列
class TokenArray:
    """
    存储 prompt + 生成的 token（包括 MASK）
    """
    def __init__(self, prompt, gen_length, mask_id, eos_id, device)
    def get_generated_tokens(self)
    def select_seqs(self, idx)

# 2. BlockIterator: 迭代块
class BlockIterator:
    """
    遍历生成序列中的每个块
    """
    def __iter__(self)
    def __next__(self)  # 返回 (block, block_loc)

# 3. KVCacheFactory: 创建缓存
class KVCacheFactory:
    """
    根据策略创建不同的 KV-Cache
    """
    def __call__(self, strategy):
        if strategy == "prefix":
            return PrefixKVCache(...)
        elif strategy == "dual":
            return DualKVCache(...)
```

**实践**: 写一个简单的测试脚本
```python
import torch
from dinfer.decoding.utils import TokenArray, BlockIterator

# 创建 TokenArray
prompt = torch.tensor([[1, 2, 3, 4, 5]])
token_array = TokenArray(prompt, gen_length=16, mask_id=999, eos_id=888, device='cuda:0')

# 遍历块
iterator = BlockIterator(token_array, block_length=4)
for block, block_loc in iterator:
    print(f"Block: {block}, Location: {block_loc.start}-{block_loc.end}")
```

**第 2 天: 并行解码策略**

阅读文件: `python/dinfer/decoding/parallel_strategy.py`

**关键函数**:
```python
# 1. get_transfer_index_hierarchy_fast_v2
def get_transfer_index_hierarchy_fast_v2(
    logits,         # 模型输出的 logits
    temperature,    # Gumbel 噪声温度
    remasking,      # 重新 mask 策略
    mask_index,     # 当前哪些位置是 MASK
    x,              # 当前序列
    num_transfer_tokens,  # 每次解码多少 token
    mask_id,
    threshold=None,
    low_threshold=None
):
    """
    核心解码逻辑:
    1. 添加 Gumbel 噪声（用于采样）
    2. 计算每个位置的置信度
    3. 根据阈值或层次策略选择要解码的 token
    4. 返回解码后的 token 和 transfer_index
    """
```

**实践**: 单元测试
```bash
# 运行解码策略测试
python tests/test_wo_model.py
```

分析输出，理解：
- 置信度如何计算？
- 阈值如何影响解码速度？
- 层次解码如何保证每次迭代都有进展？

**第 3 天: 主推理逻辑**

阅读文件: `python/dinfer/decoding/generate_uniform.py`（最重要！）

**核心类结构**:
```python
# 基类
class DiffusionLLM:
    def generate(self, prompt, gen_length, block_length):
        """基类，定义接口"""
        raise NotImplementedError

# 块级扩散
class BlockWiseDiffusionLLM(DiffusionLLM):
    def __init__(self, model, decoder, iterator_factory, cache_factory, ...):
        self.model = model           # 扩散语言模型
        self.decoder = decoder       # 并行解码器
        self.cache_factory = cache_factory  # KV-Cache 工厂
        
    def generate(self, prompt, gen_length, block_length):
        # 1. 初始化 TokenArray
        x = TokenArray(prompt, gen_length, mask_id, eos_id, device)
        
        # 2. 初始化 KV-Cache
        kv_cache = self.cache_factory.create(...)
        
        # 3. 遍历每个块
        for block, block_loc in iterator:
            # 4. 执行扩散迭代，直到块内所有 MASK 被解码
            while (block == mask_id).sum() > 0:
                # a. 前向传播
                logits = self.model.forward(x.data, kv_cache=kv_cache)
                
                # b. 解码（选择高置信度 token）
                x0, transfer_index = self.decoder.decode(logits, mask_index)
                
                # c. 更新序列
                x[block_loc.start:block_loc.end] = torch.where(
                    transfer_index, x0, x[block_loc.start:block_loc.end]
                )
                
                # d. 更新 KV-Cache
                kv_cache.update(block_loc, ...)
        
        # 5. 返回生成结果
        return x.get_generated_tokens()
```

**实践**: 添加日志，跟踪执行流程
```python
# 修改 generate_uniform.py，添加打印语句
def generate(self, prompt, gen_length, block_length):
    print(f"[Init] Prompt length: {prompt.shape[1]}, Gen length: {gen_length}")
    
    for block_id, (block, block_loc) in enumerate(iterator):
        print(f"[Block {block_id}] Processing {block_loc.start}-{block_loc.end}")
        iter_count = 0
        
        while (block == mask_id).sum() > 0:
            iter_count += 1
            mask_count = (block == mask_id).sum().item()
            print(f"  [Iter {iter_count}] Remaining MASK: {mask_count}")
            
            # ... 执行解码 ...
```

---

### 📅 阶段 3: 模型实现（2-3天）

#### 目标
- 理解扩散语言模型的架构
- 掌握张量并行（TP）和专家并行（EP）的实现
- 理解 MoE 架构的特点

#### 学习路径

**第 1 天: LLaDA 基础模型**

阅读文件: `python/dinfer/model/modeling_llada.py`

**核心类**:
```python
class LLaDAModelLM(PreTrainedModel):
    """
    LLaDA 模型实现，基于 Llama 架构
    """
    def __init__(self, config):
        self.model = LLaDAModel(config)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, positions, kv_caches, ...):
        """
        前向传播
        1. Token Embedding
        2. 多层 Transformer Block
        3. Output Projection
        """
        hidden_states = self.model(input_ids, positions, kv_caches)
        logits = self.lm_head(hidden_states)
        return logits
```

**关键组件**:
- `LLaDAAttention`: 自注意力层
- `LLaDAMLP`: 前馈网络
- `LLaDABlock`: Transformer 块

**第 2 天: LLaDA-MoE 模型**

阅读文件: `python/dinfer/model/modeling_fused_olmoe.py`

**MoE 核心**:
```python
class OlmoeMoE(nn.Module):
    """
    混合专家层
    """
    def __init__(self, config):
        self.num_experts = config.num_experts  # 专家数量
        self.top_k = config.num_experts_per_tok  # 每个 token 选择的专家数
        self.gate = nn.Linear(hidden_size, num_experts)  # 路由器
        self.experts = nn.ModuleList([
            OlmoeMLP(config) for _ in range(num_experts)
        ])
    
    def forward(self, hidden_states):
        # 1. 路由：选择 top-k 专家
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k)
        
        # 2. 专家计算
        expert_outputs = []
        for expert_idx in selected_experts:
            expert_outputs.append(self.experts[expert_idx](hidden_states))
        
        # 3. 加权合并
        output = sum(w * o for w, o in zip(routing_weights, expert_outputs))
        return output
```

**关键概念**:
- **Expert Parallel**: 不同 GPU 负责不同专家
- **Load Balancing**: 确保专家负载均衡
- **Fused MoE**: 融合多个专家的权重矩阵，提高效率

**第 3 天: LLaDA2 块扩散模型**

阅读文件: `python/dinfer/model/modeling_llada2_moe.py`

**块扩散特点**:
```python
class LLaDA2MoeModelLM:
    """
    LLaDA2 支持块扩散机制
    - 减少计算开销：不需要每次迭代都计算整个序列
    - 使用 Attention Mask 优化
    """
    def forward(self, input_ids, attention_mask=None, ...):
        # 块扩散：只计算当前块，不计算后续 MASK 块
        if use_block_diffusion:
            # 使用 attention_mask 屏蔽后续块
            attention_mask = create_block_mask(...)
        
        return super().forward(input_ids, attention_mask=attention_mask)
```

**实践**: 对比模型大小和速度
```bash
# LLaDA 8B
python benchmarks/benchmark.py --model_type llada --model_name GSAI-ML/LLaDA-8B-Instruct

# LLaDA-MoE 7B
python benchmarks/benchmark.py --model_type llada_moe --model_name inclusionAI/LLaDA-MoE-7B-A1B-Instruct-fused

# LLaDA2 16B
python benchmarks/benchmark.py --model_type llada2 --model_name inclusionAI/LLaDA2.0-mini-preview --use_bd
```

---

### 📅 阶段 4: 高级优化（2-3天）

#### 目标
- 理解系统级优化技术
- 掌握张量并行和专家并行
- 学习 PyTorch 编译和 CUDA Graphs

#### 学习内容

**1. 张量并行（Tensor Parallel）**

阅读文件: `python/dinfer/model/tp_linear.py`

```python
class TPLinear(nn.Module):
    """
    张量并行线性层
    将权重矩阵按列或行切分到多个 GPU
    """
    def __init__(self, in_features, out_features, world_size, rank, dim='column'):
        self.world_size = world_size
        self.rank = rank
        
        if dim == 'column':
            # 列切分：每个 GPU 负责部分输出维度
            self.weight = nn.Parameter(torch.randn(out_features // world_size, in_features))
        else:
            # 行切分：每个 GPU 负责部分输入维度
            self.weight = nn.Parameter(torch.randn(out_features, in_features // world_size))
    
    def forward(self, x):
        # 分布式矩阵乘法
        local_output = F.linear(x, self.weight)
        
        # All-reduce 或 All-gather 同步结果
        if self.dim == 'column':
            output = dist.all_reduce(local_output)
        else:
            output = dist.all_gather(local_output)
        
        return output
```

**实践**: 启用 TP
```bash
# 单 GPU
python benchmarks/benchmark.py --gpu 0

# 4-way TP
python benchmarks/benchmark.py --gpu 0,1,2,3 --use_tp
```

**2. PyTorch 编译优化**

```python
# 在 benchmark.py 中启用编译
model.forward = torch.compile(
    model.forward, 
    mode='reduce-overhead',  # 减少开销
    fullgraph=False,         # 允许 graph breaks
    dynamic=True             # 支持动态形状
)
```

**效果**: 减少 Python 开销，融合 CUDA 核函数

**3. CUDA Graphs**

CUDA Graphs 可以"记录"一系列 CUDA 操作，然后重放，减少 CPU-GPU 通信开销。

```python
# 使用 CUDA Graphs（在 vLLM 中自动启用）
with torch.cuda.graph():
    output = model(input_ids)
```

**4. Loop Unrolling（循环展开）**

在 `generate_uniform.py` 中的优化：
```python
# 展开循环，减少 Python 循环开销
while (block == mask_id).sum() > 0:
    unroll_k = min((block == mask_id).sum() // expected_tpf, maximum_unroll)
    for unroll_i in range(unroll_k):
        # 执行多次迭代而不检查条件
        self.diff_iteration.forward(model, decoder, ...)
```

**实践**: 对比优化效果
```bash
# 不使用编译
python benchmarks/benchmark.py --no_compile

# 使用编译
python benchmarks/benchmark.py --use_compile

# 观察 TPS 提升
```

---

### 📅 阶段 5: 评估与实验（2-3天）

#### 目标
- 学会使用评估框架
- 在标准 benchmark 上测试模型
- 分析性能和质量的权衡

#### 实践任务

**任务 1: 运行标准 benchmark**

```bash
cd evaluations

# GSM8K 数学推理
python eval_dinfer.py \
  --tasks gsm8k_llada_moe \
  --model dInfer_eval \
  --model_args model_path=your_model,gen_length=1024,block_length=64,threshold=0.8 \
  --output_path runs/gsm8k

# MBPP 代码生成
python eval_dinfer.py \
  --tasks mbpp_sanitized_llada_moe \
  --confirm_run_unsafe_code \
  --model_args model_path=your_model,gen_length=1024,block_length=64,threshold=0.8 \
  --output_path runs/mbpp
```

**任务 2: 参数调优实验**

创建实验脚本 `experiments/param_sweep.sh`:
```bash
#!/bin/bash

# 测试不同阈值
for threshold in 0.7 0.8 0.9 0.95; do
  python benchmarks/benchmark_dataset.py \
    --threshold $threshold \
    --output_dir runs/threshold_$threshold
done

# 测试不同块大小
for block_length in 16 32 64 128; do
  python benchmarks/benchmark_dataset.py \
    --block_length $block_length \
    --output_dir runs/block_$block_length
done


# 测试不同缓存策略
for cache in prefix dual vicinity; do
  python benchmarks/benchmark_dataset.py \
    --cache $cache \
    --output_dir runs/cache_$cache
done
```

**任务 3: 分析结果**

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# 加载实验结果
results = []
for threshold in [0.7, 0.8, 0.9, 0.95]:
    with open(f'runs/threshold_{threshold}/metrics.json') as f:
        data = json.load(f)
        results.append({
            'threshold': threshold,
            'tps': data['tokens_per_second'],
            'accuracy': data['accuracy']
        })

df = pd.DataFrame(results)

# 绘制速度-质量曲线
plt.plot(df['tps'], df['accuracy'], 'o-')
plt.xlabel('Tokens Per Second')
plt.ylabel('Accuracy')
plt.title('Speed-Quality Tradeoff')
plt.savefig('speed_quality.png')
```

---

## 🛠️ 实践项目

### 项目 1: 自定义解码策略 ⭐⭐

**目标**: 实现一个新的解码策略

**任务**:
1. 在 `parallel_strategy.py` 中添加新类 `AdaptiveDecoder`
2. 策略：根据当前迭代次数动态调整阈值
   - 早期迭代：高阈值（0.95），只解码最确定的 token
   - 后期迭代：低阈值（0.7），加速解码

**实现框架**:
```python
class AdaptiveDecoder:
    def __init__(self, initial_threshold=0.95, final_threshold=0.7, mask_id, eos_id):
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.iter_count = 0
        self.total_iters = 10  # 预估总迭代次数
    
    def decode(self, logits, mask_index, x):
        self.iter_count += 1
        
        # 线性衰减阈值
        progress = self.iter_count / self.total_iters
        current_threshold = (
            self.initial_threshold * (1 - progress) + 
            self.final_threshold * progress
        )
        
        # 使用当前阈值解码
        # TODO: 实现解码逻辑
        
        return x0, transfer_index
```

**测试**:
```bash
python benchmarks/benchmark.py \
  --parallel_decoding adaptive \
  --output_dir runs/adaptive
```

---

### 项目 2: 可视化工具 ⭐⭐⭐

**目标**: 创建一个可视化工具，展示扩散过程

**任务**:
1. 记录每次迭代的状态（哪些 token 是 MASK，哪些被解码）
2. 生成动画，展示解码过程
3. 分析不同位置的解码速度

**实现框架**:
```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DiffusionVisualizer:
    def __init__(self):
        self.history = []  # 存储每次迭代的状态
    
    def record(self, x, mask_index, iteration):
        """记录当前状态"""
        self.history.append({
            'iteration': iteration,
            'tokens': x.clone(),
            'mask_index': mask_index.clone()
        })
    
    def animate(self, tokenizer, output_path='diffusion.gif'):
        """生成动画"""
        fig, ax = plt.subplots(figsize=(12, 2))
        
        def update(frame):
            ax.clear()
            state = self.history[frame]
            tokens = state['tokens'][0].cpu().numpy()
            mask_idx = state['mask_index'][0].cpu().numpy()
            
            # 可视化：绿色=已解码，红色=MASK
            colors = ['green' if not m else 'red' for m in mask_idx]
            ax.bar(range(len(tokens)), [1]*len(tokens), color=colors)
            ax.set_title(f"Iteration {state['iteration']}")
            ax.set_xlabel('Position')
            ax.set_ylim(0, 1.5)
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(self.history), interval=500
        )
        ani.save(output_path, writer='pillow')
        print(f"Animation saved to {output_path}")
```

**集成到推理**:
```python
# 修改 BlockWiseDiffusionLLM.generate()
visualizer = DiffusionVisualizer()

while (block == mask_id).sum() > 0:
    # ... 执行解码 ...
    visualizer.record(x, mask_index, iter_count)

visualizer.animate(tokenizer, 'output.gif')
```

---

### 项目 3: 在线服务 Demo ⭐⭐⭐

**目标**: 部署一个简单的在线推理服务

**任务**:
1. 使用 `serving.py` 中的 `DiffusionLLMServing`
2. 创建 FastAPI 接口
3. 支持流式输出

**实现**
:
```python
from fastapi import FastAPI
from dinfer import DiffusionLLMServing, SamplingParams

app = FastAPI()

# 初始化服务
serving = DiffusionLLMServing(
    model="your_model_path",
    is_moe=True,
    gpu_ids=[0, 1, 2, 3],
    use_tp=True
)

@app.post("/generate")
async def generate(prompt: str, max_length: int = 512):
    """生成文本"""
    sampling_params = SamplingParams(
        gen_length=max_length,
        block_length=32,
        threshold=0.9
    )
    
    result = serving.generate(prompt, sampling_params)
    return {"text": result}

@app.get("/health")
async def health():
    return {"status": "ok"}

# 运行: uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## 🚀 进阶主题

### 1. 与其他框架对比

**对比实验**:
```bash
# dInfer
python benchmarks/benchmark_dataset.py \
  --model_type llada_moe \
  --output_dir runs/dinfer

# Fast-dLLM（框架内置对比）
python benchmarks/benchmark_dataset_fastdllm.py \
  --model_type llada \
  --output_dir runs/fastdllm

# SGLang（框架内置对比）
python benchmarks/benchmark_dataset_sglang.py \
  --model_type llada2 \
  --output_dir runs/sglang
```

**分析维度**:
- TPS (Tokens Per Second)
- 延迟 (Latency)
- 内存使用 (Memory Usage)
- 生成质量 (Quality)

---

### 2. 源码贡献指南

如果你想为 dInfer 贡献代码：

**步骤**:
1. Fork 仓库
2. 创建分支: `git checkout -b feature/my-feature`
3. 编写代码和测试
4. 提交 PR

**代码规范**:
- 遵循 PEP 8
- 添加类型注解
- 编写文档字符串
- 添加单元测试

**测试**:
```bash
# 运行测试
pytest tests/

# 运行特定测试
pytest tests/test_llada_moe.py -v
```

---

### 3. 扩展阅读

**论文**:
- dInfer 技术报告: https://arxiv.org/abs/2510.08666
- LLaDA: Latent Diffusion for Language Models
- DDPM: Denoising Diffusion Probabilistic Models
- OLMoE: Mixture-of-Experts in Open Language Models

**相关项目**:
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- Fast-dLLM: (查看 dInfer 对比实现)

---

## ❓ 常见问题

### Q1: 为什么需要转换 MoE 模型为 FusedMoE？

**A**: 融合 MoE 可以：
- 减少内存碎片
- 提高专家计算效率
- 更好支持 Expert Parallel

使用 `tools/transfer.py` 进行转换。

---

### Q2: 如何选择合适的阈值（threshold）？

**A**: 
- **高阈值（0.9-0.95）**: 高质量，但速度较慢
- **中阈值（0.8-0.85）**: 平衡速度和质量
- **低阈值（0.7-0.75）**: 高速度，但质量可能下降

建议：根据任务需求调优。

---

### Q3: 块大小（block_length）如何影响性能？

**A**:
- **小块（16-32）**: 更频繁的 KV-Cache 更新，适合短文本
- **大块（64-128）**: 更少的 KV-Cache 更新，适合长文本

LLaDA-MoE 推荐: 64
LLaDA2 推荐: 32

---

### Q4: Dual Cache 和 Prefix Cache 有什么区别？

**A**:
- **Prefix Cache**: 只缓存固定前缀，适合 prompt 不变的场景
- **Dual Cache**: 双缓存策略，当前块 + 历史缓存，更灵活但内存开销更大

---

### Q5: 为什么 LLaDA2 只支持 4-way TP？

**A**: LLaDA2 只有 4 个注意力头（attention heads），因此最多只能切分到 4 个 GPU。如需更大并行度，可使用 LLaDA-MoE（支持 8-way TP）。

---

### Q6: 如何调试推理速度慢的问题？

**A**:
1. 检查是否启用 PyTorch 编译: `--use_compile`
2. 检查是否使用 TP: `--use_tp`
3. 检查 batch size 是否合适
4. 使用 `nvidia-smi` 检查 GPU 利用率
5. 使用 PyTorch Profiler 分析瓶颈

---

### Q7: 能否在 CPU 上运行？

**A**: dInfer 依赖 CUDA 和 NCCL，目前不支持纯 CPU 推理。最低要求 1 个 GPU。

---

## 📝 学习检查清单

### 基础知识 ✅
- [ ] 理解扩散模型的基本原理
- [ ] 理解 dLLM 与自回归 LLM 的区别
- [ ] 理解并行解码的概念
- [ ] 成功运行第一个推理示例

### 解码逻辑 ✅
- [ ] 理解 `TokenArray` 和 `BlockIterator` 的作用
- [ ] 理解 Threshold、Hierarchy、Credit 解码策略
- [ ] 理解 KV-Cache 管理机制
- [ ] 能够阅读 `generate_uniform.py` 的主流程

### 模型实现 ✅
- [ ] 理解 LLaDA 模型架构
- [ ] 理解 MoE 的路由和专家机制
- [ ] 理解 LLaDA2 的块扩散机制
- [ ] 理解张量并行和专家并行

### 系统优化 ✅
- [ ] 理解 PyTorch 编译优化
- [ ] 理解 CUDA Graphs
- [ ] 理解循环展开优化
- [ ] 能够分析性能瓶颈

### 实践能力 ✅
- [ ] 能够运行 benchmark 测试
- [ ] 能够使用评估框架
- [ ] 能够调优超参数
- [ ] 能够实现自定义解码策略

---

## 🎓 总结

恭喜你完成 dInfer 框架的学习！

**你已经掌握**:
1. ✅ 扩散语言模型的核心原理
2. ✅ dInfer 的四大组件设计
3. ✅ 并行解码策略的实现
4. ✅ 系统级优化技术
5. ✅ 模型评估和调优方法

**下一步建议**:
- 🔬 在实际项目中应用 dInfer
- 📝 为社区贡献代码或文档
- 🚀 探索新的解码算法
- 📊 在更多 benchmark 上测试

**加入社区**:
- GitHub: https://github.com/inclusionAI/dInfer
- 微信群: 见 README.md 中的二维码
- 技术报告: https://arxiv.org/abs/2510.08666

---

**Happy Coding! 🎉**
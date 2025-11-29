"""
简单的 LLaDA-MoE 离线推理测试脚本
使用方法: python simple_inference.py
"""
import os

import torch
from dinfer import (
    BlockIteratorFactory,
    BlockWiseDiffusionLLM,
    KVCacheFactory,
    ThresholdParallelDecoder,
)
from dinfer.model import LLaDAMoeModelLM
from transformers import AutoConfig, AutoTokenizer
from vllm import distributed
from vllm.config import (
    ParallelConfig,  #type: ignore
    VllmConfig,
    set_current_vllm_config,
)

# ============ 配置参数 ============
MODEL_PATH = "/public/home/caiyiwen/model/LLaDA-MoE-7B-A1B-Instruct-fused"
GPU_ID = 0  # 使用的GPU编号
GEN_LENGTH = 128  # 生成的token数量
BLOCK_LENGTH = 32  # 块大小
THRESHOLD = 0.9  # 解码阈值
MASK_ID = 156895  # LLaDA-MoE的mask token ID
EOS_ID = 156892  # LLaDA-MoE的eos token ID

# ============ 测试提示词 ============
TEST_PROMPTS = [
    "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours?",
    "What is the capital of France?",
    "Write a hello world program in Python.",
]


def init_distributed_env():
    """初始化分布式环境（单卡模式）"""
    print("[1/5] 初始化分布式环境...")

    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # 设置GPU设备
    device = torch.device(f"cuda:{GPU_ID}")
    torch.cuda.set_device(device)

    # 初始化 vLLM 分布式环境（单卡模式）
    distributed.init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method='env://',
        local_rank=0,
        backend='nccl'
    )

    # 初始化模型并行（单卡为1）
    distributed.initialize_model_parallel(tensor_model_parallel_size=1, backend='nccl')

    print(f"使用设备: {device}")
    return device


def load_model(model_path, device):
    """加载模型和分词器"""
    print(f"[2/5] 加载模型: {model_path}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    print("分词器加载完成")

    # 设置并行配置（启用专家并行）
    parallel_config = ParallelConfig(enable_expert_parallel=True)

    # 在 vLLM 配置上下文中加载模型
    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = LLaDAMoeModelLM(config=model_config).eval()
        model.load_weights(model_path, torch_dtype=torch.bfloat16)
        model = model.to(device)

    print("模型加载完成 (bfloat16)")
    return model, tokenizer


def create_diffusion_llm(model):
    """创建扩散语言模型推理器"""
    print("[3/5] 创建扩散语言模型推理器...")

    # 创建阈值并行解码器
    decoder = ThresholdParallelDecoder(
        0,
        threshold=THRESHOLD,
        mask_id=MASK_ID,
        eos_id=EOS_ID
    )

    # 创建块级扩散语言模型（使用dual KV-cache优化）
    dllm = BlockWiseDiffusionLLM(
        model,
        decoder,
        BlockIteratorFactory(True),
        cache_factory=KVCacheFactory("dual"),
        early_stop=True  # 启用提前停止
    )

    print(f"解码器: ThresholdParallelDecoder (threshold={THRESHOLD})")
    print("KV-Cache: Dual Cache")
    return dllm


def run_inference(dllm, tokenizer, device):
    """运行推理测试"""
    print(f"[4/5] 运行推理测试 (gen_length={GEN_LENGTH}, block_length={BLOCK_LENGTH})...")
    print("=" * 80)

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n【测试 {i}/{len(TEST_PROMPTS)}】")
        print(f"提示词: {prompt}")
        print("-" * 80)

        # 编码输入
        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        # 生成
        with torch.no_grad():
            result = dllm.generate(
                input_ids,
                gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH
            )

        # 解码输出
        generated_text = tokenizer.decode(result[0], skip_special_tokens=True)
        print(f"生成结果:\n{generated_text}")
        print("=" * 80)


def cleanup():
    """清理资源"""
    print("[5/5] 清理资源...")
    distributed.destroy_model_parallel()
    distributed.destroy_distributed_environment()
    print("   ✓ 完成")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("LLaDA-MoE 离线推理测试")
    print("=" * 80 + "\n")

    try:
        # 初始化
        device = init_distributed_env()

        # 加载模型
        model, tokenizer = load_model(MODEL_PATH, device)

        # 创建推理器
        dllm = create_diffusion_llm(model)

        # 运行推理
        run_inference(dllm, tokenizer, device)

        # 清理
        cleanup()

        print("\n✅ 测试完成！\n")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        cleanup()


if __name__ == "__main__":
    main()

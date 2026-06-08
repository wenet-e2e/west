# Copyright (c) 2026 Pengshen Zhang
"""Engine Loader: 模型资源加载、warmup 与 ready 状态。

- load_model: 初始化 processor、vLLM engine、sampling 参数和 VAD provider
- warmup_engine: 启动后执行一次轻量推理，提前暴露模型加载问题
- build_engine_kwargs: 汇总 vLLM/Qwen 后端启动参数
- 只负责进程级资源，不持有单连接 session 状态
"""
import logging
import os
import time
from typing import TYPE_CHECKING

from engine.env_snapshot import (log_startup_gpu_snapshot,
                                 log_startup_host_snapshot)
from engine.runtime import EnginePhase
from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
from qwen_omni_utils import process_mm_info as _process_mm_info
from transformers import Qwen3OmniMoeProcessor
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, ModelRegistry
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from engine.runtime import ServiceRuntime


def _configure_qwen2_tokenizer_special_tokens() -> None:
    Qwen2Tokenizer.image_token = "<|image_pad|>"
    Qwen2Tokenizer.audio_token = "<|audio_pad|>"
    Qwen2Tokenizer.video_token = "<|video_pad|>"
    Qwen2Tokenizer.vision_bos_token = "<|vision_bos|>"
    Qwen2Tokenizer.vision_eos_token = "<|vision_eos|>"
    Qwen2Tokenizer.audio_bos_token = "<|audio_bos|>"
    Qwen2Tokenizer.audio_eos_token = "<|audio_eos|>"


def build_engine_kwargs(
    *,
    model: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> dict:
    enforce_eager = os.getenv("ENFORCE_EAGER", "0").lower() in (
        "1", "true", "yes")
    return dict(
        model=model,
        trust_remote_code=True,
        tokenizer_mode="slow",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=32768,
        max_num_seqs=32,
        seed=1234,
        limit_mm_per_prompt={"image": 1, "video": 1, "audio": 1},
        enforce_eager=enforce_eager,
    )


async def init_engine(
        *,
        service_runtime: "ServiceRuntime",
        logger: logging.Logger,
        model: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.75,
        model_type_arg: str = "qwen3-omni"):
    service_runtime.settings.model_name = model
    service_runtime.settings.model_type = model_type_arg

    logger.info(
        "========== 模型初始化开始 "
        "(过程可能需要数分钟，请耐心等待...) ==========")

    log_startup_host_snapshot(logger)
    logger.info(f"Model type: {model_type_arg}, Backend: vLLM (standard)")
    log_startup_gpu_snapshot(logger)

    service_runtime.engine_state.set_phase(
        EnginePhase.LOADING_PROCESSOR,
        "1/3 加载 Processor",
    )
    logger.info(f"-> {service_runtime.engine_state.message}: {model}")
    start_time = time.time()

    if model_type_arg == "qwen3-omni":
        service_runtime.engine_state.process_mm_info = _process_mm_info
        _configure_qwen2_tokenizer_special_tokens()
        service_runtime.engine_state.processor = (
            Qwen3OmniMoeProcessor.from_pretrained(
                model, use_fast=False, fix_mistral_regex=True))
    elif model_type_arg == "qwen3-asr":
        service_runtime.engine_state.processor = (
            Qwen3ASRProcessor.from_pretrained(
                model, fix_mistral_regex=True))
        service_runtime.engine_state.process_mm_info = None

        # 注册 Qwen3-ASR 自定义模型到 vLLM ModelRegistry
        try:
            ModelRegistry.register_model(
                "Qwen3ASRForConditionalGeneration",
                Qwen3ASRForConditionalGeneration)
            logger.info(
                "Registered Qwen3ASRForConditionalGeneration "
                "to vLLM ModelRegistry")
        except ImportError as e:
            logger.warning(
                f"Could not register Qwen3ASR model: {e}. "
                f"Model may not load correctly.")
    else:
        raise ValueError(f"Unknown model_type: {model_type_arg}")

    logger.info(
        f"<- 1/3 Processor 加载完成，耗时: "
        f"{time.time() - start_time:.2f}s")

    enforce_eager = os.getenv("ENFORCE_EAGER", "0").lower() in (
        "1", "true", "yes")
    if enforce_eager:
        logger.info(
            "ENFORCE_EAGER=1 已生效，禁用 torch.compile + CUDA graph 捕获 "
            "(规避 vLLM 0.16 + Qwen3-Omni MoE 在 4×3090 PCIe 预热时的偶发崩溃)")

    engine_kwargs = build_engine_kwargs(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    service_runtime.engine_state.set_phase(
        EnginePhase.LOADING_ENGINE,
        "2/3 初始化推理引擎 (加载模型权重，耗时最久)",
    )
    logger.info(
        f"-> {service_runtime.engine_state.message}，"
        f"准备分配 {tensor_parallel_size} 张 GPU ...")
    vllm_start_time = time.time()

    engine_args = AsyncEngineArgs(**engine_kwargs)
    service_runtime.engine_state.engine = AsyncLLMEngine.from_engine_args(
        engine_args)

    # ==========================
    # 引擎预热：等待 Worker 加载完毕
    # ==========================
    service_runtime.engine_state.set_phase(
        EnginePhase.WARMING_UP,
        "3/4 执行引擎预热 (分配 KV Cache，约需1-2分钟)",
    )
    logger.info(f"-> {service_runtime.engine_state.message} ...")
    warmup_t0 = time.time()
    try:
        sp = SamplingParams(max_tokens=1, temperature=0.0)

        _proc = service_runtime.engine_state.processor
        if (hasattr(_proc, "apply_chat_template")
                and getattr(_proc, 'chat_template', None)):
            messages = [{"role": "user", "content": "你好"}]
            prompt_formatted = _proc.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True)
        else:
            prompt_formatted = (
                "<|im_start|>user\n你好<|im_end|>\n"
                "<|im_start|>assistant\n")

        async for _ in service_runtime.engine_state.engine.generate(
            prompt=prompt_formatted,
            sampling_params=sp,
            request_id="warmup-1"
        ):
            pass

        logger.info(
            f"<- 3/4 引擎预热成功！耗时: "
            f"{time.time() - warmup_t0:.2f}s")
    except Exception as e:
        logger.error(f"引擎预热失败，服务可能不可用: {e}")

    logger.info(
        f"<- 2/3 vLLM 引擎加载完成，含预热耗时: "
        f"{time.time() - vllm_start_time:.2f}s")
    logger.info(
        f"-> 4/4 模型初始化全部完成！总耗时: "
        f"{time.time() - start_time:.2f}s")
    logger.info("========== 服务就绪 ==========")

    try:
        service_runtime.vad_provider.load()
    except Exception as e:
        logger.warning(
            f"Silero VAD 加载失败 ({e})，"
            f"server_vad 模式不可用，none 模式正常工作")

    service_runtime.engine_state.set_phase(EnginePhase.READY, "就绪")

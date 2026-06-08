# Copyright (c) 2026 Pengshen Zhang
"""Model Inference: 单个音频片段的纯推理流程。

- build_messages: 构造 Qwen Omni/Qwen ASR 输入消息和音频 payload
- run_model_inference: 调用 engine.generate 并流式产出文本 delta
- history_rollback/history_reset/min_history_chars 在这里参与 prompt 拼接
- 模块会被 inference_loop 热 reload，避免持有全局模型资源
"""
import asyncio
import logging
import time
import traceback
import uuid
from typing import AsyncGenerator

import numpy as np
from engine.config import InferenceConfig
from model.history_rollback import apply_history_rollback
from model.transcript_postprocess import (format_history_prefix,
                                          postprocess_transcript)
from vllm.sampling_params import SamplingParams

logger = logging.getLogger("RealtimeASR")
EVENT_NAME_WIDTH = 18


def evt_prefix(session_id: str, event_name: str) -> str:
    """统一日志事件前缀: [sid=...][evt=...]."""
    return (
        f"[sid={session_id}]"
        f"[evt={event_name:<{EVENT_NAME_WIDTH}}]"
    )


def process_mm_info_worker(messages, process_mm_info_fn):
    """Run Qwen-Omni multimodal preprocessing in an executor worker."""
    return process_mm_info_fn(messages, use_audio_in_video=True)


async def run_model_inference(
    audio_numpy: np.ndarray,
    prompt_text: str,
    session,  # RealtimeSession (duck-typed, 不直接 import 避免循环依赖)
    engine,
    processor,
    process_mm_info_fn,
    infer_tag: str = "",
    model_type: str = "qwen3-omni",
    cfg: "InferenceConfig | None" = None,
) -> AsyncGenerator[str, None]:
    """
    封装调用标准 vLLM 引擎，yield 出本次推理的完整识别文本。
    engine/processor/process_mm_info_fn 从 server.py 传入，本模块不持有。
    返回值通过 session.accumulated_text 暴露给调用方。

    Args:
        model_type: "qwen3-omni" or "qwen3-asr"
        cfg: frozen InferenceConfig snapshot, 由 server.py 注入
    """
    loop = asyncio.get_running_loop()
    sid = session.session_id
    tag = f"[{sid}] {infer_tag}" if infer_tag else f"[{sid}]"
    infer_t0 = time.time()

    if cfg is None:
        cfg = InferenceConfig()

    # 构建 messages
    if model_type == "qwen3-omni":
        messages = [{
            "role": "user",
            "content": [{
                "type": "audio",
                "audio": audio_numpy
            }, {
                "type": "text",
                "text": prompt_text
            }]
        }]
    else:
        # qwen3-asr: system message 当 prompt，user message 承载音频
        messages = [
            {
                "role": "system",
                "content": prompt_text
            },
            {
                "role": "user",
                "content": [{
                    "type": "audio",
                    "audio": ""
                }]
            }
        ]

    feat_t0 = time.time()
    prompt_formatted = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    if model_type == "qwen3-omni" and process_mm_info_fn is not None:
        audios, _, _ = await loop.run_in_executor(
            None, process_mm_info_worker,
            messages, process_mm_info_fn)
    else:
        # qwen3-asr: 直接使用 audio_numpy
        audios = [audio_numpy]

    feat_cost = time.time() - feat_t0

    use_history = session.asr.use_history

    async with session.lock:
        history = session.asr.accumulated_text
        history_rollback_cfg = session.asr.history_rollback_config
        current_chunk_id = session.asr.chunk_id
        history_reset_chunk_num = session.asr.history_reset_chunk_num
        min_history_chars = session.asr.min_history_chars
        history_language = session.asr.language

    if use_history:
        text_to_append = apply_history_rollback(
            history, history_rollback_cfg,
            tokenizer=processor.tokenizer,
            current_chunk_id=current_chunk_id,
            history_reset_chunk_num=history_reset_chunk_num,
            min_history_chars=min_history_chars
        )
    else:
        text_to_append = ""

    if text_to_append:
        text_to_append_formatted = format_history_prefix(
            text_to_append,
            history_language or "Chinese",
            model_type)
        prompt_formatted += text_to_append_formatted
        dropped = history[len(text_to_append):]
        reset_info = (
            f" (chunk {current_chunk_id}/"
            f"{history_reset_chunk_num} reset)"
            if current_chunk_id <= history_reset_chunk_num
            else "")
        logger.info(
            f"{evt_prefix(sid, 'CHUNK_HISTORY')} "
            f"{tag} History: "
            f"\"{history[:60]}\"({len(history)}) "
            f"-> append \"{text_to_append[:60]}\""
            f"({len(text_to_append)}) "
            f"drop \"{dropped}\"({len(dropped)}) "
            f"[{history_rollback_cfg.strategy}/"
            f"{history_rollback_cfg.value}]{reset_info}")
    else:
        if current_chunk_id <= history_reset_chunk_num:
            reason = (
                f"[chunk {current_chunk_id}/"
                f"{history_reset_chunk_num} history_reset]")
        elif (min_history_chars > 0 and history
              and len(history) <= min_history_chars):
            reason = (
                f"[min_history_chars={min_history_chars} "
                f"history_len={len(history)} dropped]")
        elif use_history:
            reason = (
                f"[{history_rollback_cfg.strategy}/"
                f"{history_rollback_cfg.value}]")
        else:
            reason = "[use_history=False]"
        logger.info(
            f"{evt_prefix(sid, 'CHUNK_HISTORY')} "
            f"{tag} History: (empty) {reason}")

    logger.debug(
        f"{tag} FinalPrompt (tail 200): "
        f"...{prompt_formatted[-200:]}")

    request_id = f"{sid}-{uuid.uuid4().hex[:6]}"

    sp_cfg = cfg.sampling_params
    sp = SamplingParams(
        temperature=sp_cfg.temperature,
        top_p=sp_cfg.top_p,
        top_k=sp_cfg.top_k,
        max_tokens=sp_cfg.max_tokens,
        seed=sp_cfg.seed,
        detokenize=True,
        repetition_penalty=sp_cfg.repetition_penalty)

    logger.debug(
        f"{tag} SamplingParams: temp={sp.temperature}, "
        f"top_p={sp.top_p}, top_k={sp.top_k}, "
        f"max_tokens={sp.max_tokens}, "
        f"rep_penalty={sp.repetition_penalty}")

    try:
        gen_t0 = time.time()
        previous_text = ""
        raw_text = ""
        token_count = 0
        first_token_time = None

        vllm_inputs = {
            "prompt": prompt_formatted,
            "multi_modal_data": {"audio": audios},
        }
        if model_type == "qwen3-omni":
            vllm_inputs["mm_processor_kwargs"] = {
                "use_audio_in_video": True,
            }

        last_output = None
        async for output in engine.generate(
                vllm_inputs,
                sampling_params=sp,
                request_id=request_id):
            if not getattr(output, "outputs", None):
                continue
            last_output = output

            current_text = output.outputs[0].text or ""
            delta_text = current_text[len(previous_text):]
            previous_text = current_text

            if delta_text:
                token_count += 1
                raw_text += delta_text
                result = postprocess_transcript(
                    raw_text, model_type)
                if first_token_time is None and result.text:
                    first_token_time = time.time()
                    logger.info(
                        f"{evt_prefix(sid, 'CHUNK_FIRST_TOKEN')} "
                        f"{tag} FirstToken: "
                        f"\"{result.text[:20]}\" "
                        f"latency="
                        f"{first_token_time - gen_t0:.3f}s")
                if result.text:
                    yield result.text

        gen_cost = time.time() - gen_t0
        total_cost = time.time() - infer_t0

        if token_count == 0 and last_output is None:
            logger.warning(
                f"{evt_prefix(sid, 'DIAG_TOKENS_ZERO')} "
                f"{tag} vLLM generate yielded NO output at all | "
                f"history_prefix_len={len(text_to_append)} "
                f"history_prefix=\"{text_to_append[:60]}\" "
                f"(possible preemption/cancellation)")
        elif token_count == 0 and last_output is not None:
            out0 = last_output.outputs[0]
            diag_raw = getattr(out0, 'text', '')
            finish_r = getattr(out0, 'finish_reason', None)
            stop_r = getattr(out0, 'stop_reason', None)
            num_tokens = len(getattr(out0, 'token_ids', []))
            logger.warning(
                f"{evt_prefix(sid, 'DIAG_TOKENS_ZERO')} "
                f"{tag} vLLM returned tokens=0 | "
                f"finish_reason={finish_r} stop_reason={stop_r} "
                f"raw_text=\"{diag_raw[:100]}\" "
                f"output_token_ids_len={num_tokens} "
                f"history_prefix_len={len(text_to_append)} "
                f"history_prefix=\"{text_to_append[:60]}\" "
                f"prompt_tail=\"{prompt_formatted[-200:]}\"")

        # Keep history/cursor state punctuation-free. The caller commits
        # ``session.asr.accumulated_text`` after this generator finishes.
        result = postprocess_transcript(raw_text, model_type)
        language = result.language
        clean_text = result.text

        new_accumulated = (
            text_to_append + clean_text
            if use_history else clean_text)
        async with session.lock:
            old_accumulated = session.asr.accumulated_text
            session.asr.accumulated_text = new_accumulated
            session.asr.trailing_punct = result.trailing_punct
            session.asr.language = language if language else None

        logger.info(
            f"{evt_prefix(sid, 'CHUNK_INFER_DONE')} "
            f"{tag} Done \"{clean_text[:80]}\" "
            f"trailing_punct=\"{result.trailing_punct}\" | "
            f"accumulated: \"{old_accumulated[:40]}\" -> "
            f"\"{new_accumulated[:60]}\" | "
            f"tokens={token_count} "
            f"generate={gen_cost:.3f}s "
            f"feat={feat_cost:.3f}s "
            f"total={total_cost:.3f}s")
    except Exception:
        logger.error(
            f"{tag} Inference ERROR: {traceback.format_exc()}")
        yield ""

# Copyright (c) 2026 Pengshen Zhang
"""Model Inference: 单个音频片段的纯推理流程。

- build_messages: 构造 Qwen Omni/Qwen ASR 输入消息和音频 payload
- run_model_inference: 调用 engine.generate 并流式产出文本 delta
- history_base 由 inference_loop 统一计算并在这里参与 prompt 拼接
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
from model.transcript_postprocess import (merge_transcript_boundary,
                                          postprocess_transcript)
from vllm.sampling_params import SamplingParams

logger = logging.getLogger("RealtimeASR")
EVENT_NAME_WIDTH = 18
LANGUAGE_TEXT = {
    "Chinese": "中文",
    "English": "英文",
    "Cantonese": "粤语",
}


def evt_prefix(session_id: str, event_name: str) -> str:
    """统一日志事件前缀: [sid=...][evt=...]."""
    return (
        f"[sid={session_id}]"
        f"[evt={event_name:<{EVENT_NAME_WIDTH}}]"
    )


def process_mm_info_worker(messages, process_mm_info_fn):
    """Run Qwen-Omni multimodal preprocessing in an executor worker."""
    return process_mm_info_fn(messages, use_audio_in_video=True)


_DEFAULT_USER_PROMPT = "将这段语音转录为纯文本"
_CONTINUATION_PREFIX_PUNCT = frozenset("，。,.!?！？、；;：:…·—–")


def _strip_continuation_prefix_punct(text: str) -> str:
    """Remove boundary punctuation at the start of a continued chunk."""
    if not text:
        return text
    if text[0] in _CONTINUATION_PREFIX_PUNCT:
        return text[1:]
    return text


def build_qwen3omni_user_prompt(
    user_prompt: str = "",
    context: str = "",
    language: str = "",
) -> str:
    """Build Qwen3-Omni user text from task prompt and optional context."""
    if not user_prompt:
        user_prompt = _DEFAULT_USER_PROMPT
        logger.debug(
            f"user_prompt is empty, using default: \"{_DEFAULT_USER_PROMPT}\"")

    language_label = LANGUAGE_TEXT.get(language, language)
    user_prompt = user_prompt.replace("{language}", language_label)

    if context:
        return (
            f"参考上下文：\n{context}\n\n"
            f"请结合参考上下文，{user_prompt}。"
        )

    return user_prompt


async def run_model_inference(
    audio_numpy: np.ndarray,
    user_prompt: str,
    session,  # RealtimeSession (duck-typed, 不直接 import 避免循环依赖)
    engine,
    processor,
    process_mm_info_fn,
    infer_tag: str = "",
    model_type: str = "qwen3-omni",
    cfg: "InferenceConfig | None" = None,
    system_prompt: str = "",
    context: str = "",
    language: str = "",
    history_base: str = "",
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

    user_prompt = user_prompt.strip()
    system_prompt = system_prompt.strip()
    context = context.strip()
    language = language.strip()

    # 构建 messages：Omni 使用自然语言 prompt；ASR 使用 context + 协议前缀。
    if model_type == "qwen3-omni":
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": system_prompt
                }]
            })
        messages.append({
            "role": "user",
            "content": [{
                "type": "audio",
                "audio": audio_numpy
            }, {
                "type": "text",
                "text": build_qwen3omni_user_prompt(
                    user_prompt, context, language)
            }]
        })
    else:
        # qwen3-asr: system message 是上下文，任务/语种由 assistant 前缀约束。
        # user_prompt/system_prompt 仅适用于 Qwen3-Omni，ASR 模型忽略。
        if user_prompt:
            logger.debug(
                f"{tag} qwen3-asr ignores user_prompt "
                f"(only context/language apply)")
        if system_prompt:
            logger.debug(
                f"{tag} qwen3-asr ignores system_prompt "
                f"(only context/language apply)")
        messages = [
            {
                "role": "system",
                "content": context
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
    forced_asr_language = (
        language
        if model_type == "qwen3-asr" and language
        else None)
    if model_type == "qwen3-asr" and language:
        prompt_formatted += f"language {language}<asr_text>"

    if model_type == "qwen3-omni" and process_mm_info_fn is not None:
        audios, _, _ = await loop.run_in_executor(
            None, process_mm_info_worker,
            messages, process_mm_info_fn)
    else:
        # qwen3-asr: 直接使用 audio_numpy
        audios = [audio_numpy]

    feat_cost = time.time() - feat_t0

    async with session.lock:
        # ``history`` is only for diagnostics; ``history_base`` is the
        # single rollback result computed by inference_loop and used below.
        history = session.asr.accumulated_text
        current_chunk_id = session.asr.chunk_id
        history_reset_chunk_num = session.asr.history_reset_chunk_num
        history_language = session.asr.language

    if history_base:
        if model_type == "qwen3-asr":
            if forced_asr_language:
                history_base_formatted = history_base
            else:
                history_lang = history_language or "Chinese"
                history_base_formatted = (
                    f"language {history_lang}<asr_text>{history_base}"
                )
        else:
            history_base_formatted = history_base
        prompt_formatted += history_base_formatted
        dropped = history[len(history_base):]
        reset_info = (
            f" (chunk {current_chunk_id}/"
            f"{history_reset_chunk_num} reset)"
            if current_chunk_id <= history_reset_chunk_num
            else "")
        logger.info(
            f"{evt_prefix(sid, 'CHUNK_HISTORY')} "
            f"{tag} History: "
            f"\"{history[:60]}\"({len(history)}) "
            f"-> append \"{history_base[:60]}\""
            f"({len(history_base)}) "
            f"drop \"{dropped}\"({len(dropped)}) "
            f"{reset_info}")
    else:
        if current_chunk_id <= history_reset_chunk_num:
            reason = (
                f"[chunk {current_chunk_id}/"
                f"{history_reset_chunk_num} history_reset]")
        elif history:
            reason = "[history_base_empty]"
        else:
            reason = "[empty_history]"
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
                    raw_text, model_type,
                    user_language=forced_asr_language)
                display_text = (
                    _strip_continuation_prefix_punct(result.text)
                    if history_base else result.text)
                if first_token_time is None and display_text:
                    first_token_time = time.time()
                    logger.info(
                        f"{evt_prefix(sid, 'CHUNK_FIRST_TOKEN')} "
                        f"{tag} FirstToken: "
                        f"\"{display_text[:20]}\" "
                        f"latency="
                        f"{first_token_time - gen_t0:.3f}s")
                if display_text:
                    yield display_text

        gen_cost = time.time() - gen_t0
        total_cost = time.time() - infer_t0

        if token_count == 0 and last_output is None:
            logger.warning(
                f"{evt_prefix(sid, 'DIAG_TOKENS_ZERO')} "
                f"{tag} vLLM generate yielded NO output at all | "
                f"history_prefix_len={len(history_base)} "
                f"history_prefix=\"{history_base[:60]}\" "
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
                f"history_prefix_len={len(history_base)} "
                f"history_prefix=\"{history_base[:60]}\" "
                f"prompt_tail=\"{prompt_formatted[-200:]}\"")

        # Keep history/cursor state punctuation-free. The caller commits
        # ``session.asr.accumulated_text`` after this generator finishes.
        result = postprocess_transcript(
            raw_text, model_type,
            user_language=forced_asr_language)
        language = result.language
        clean_text = (
            _strip_continuation_prefix_punct(result.text)
            if history_base else result.text)

        async with session.lock:
            old_accumulated = session.asr.accumulated_text
            if clean_text:
                new_accumulated = merge_transcript_boundary(
                    history_base, clean_text)
                session.asr.accumulated_text = new_accumulated
                session.asr.confirmed_text = new_accumulated
                session.asr.trailing_punct = result.trailing_punct
                session.asr.language = language if language else None
            else:
                new_accumulated = old_accumulated
                if result.trailing_punct:
                    session.asr.trailing_punct = result.trailing_punct
                if language:
                    session.asr.language = language

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

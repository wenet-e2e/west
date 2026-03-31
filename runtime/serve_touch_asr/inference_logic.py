# Copyright (c) 2026 Pengshen Zhang
# ==============================================
# 推理逻辑模块
#
# 支持 importlib.reload 热更新，修改本文件后下次推理自动生效。
# 模型引擎 (engine/processor) 由 server.py 传入，本模块不持有。
# ==============================================
import asyncio
import logging
import os
import re
import sys
import time
import traceback
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

import numpy as np
import torch
import yaml as _yaml

logger = logging.getLogger("RealtimeASR")

# ==============================================
# 热更新推理配置
# ==============================================
_INFERENCE_CONFIG_PATH = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    "conf",
    "inference_config.yaml")

if not hasattr(sys, '_inference_config_state'):
    sys._inference_config_state = {"cache": {}, "mtime": 0.0}


def load_inference_config() -> Dict[str, Any]:
    """读取 inference_config.yaml，仅文件变更时重新加载并打日志。
    支持 log_level 字段热切换 logger 级别（DEBUG/INFO/WARNING）。
    """
    state = sys._inference_config_state
    try:
        mtime = os.path.getmtime(_INFERENCE_CONFIG_PATH)
        if mtime != state["mtime"]:
            with open(_INFERENCE_CONFIG_PATH, 'r', encoding='utf-8') as f:
                state["cache"] = _yaml.safe_load(f) or {}
            state["mtime"] = mtime

            new_level = state["cache"].get("log_level", "INFO").upper()
            numeric_level = getattr(logging, new_level, logging.INFO)
            if logger.level != numeric_level:
                logger.setLevel(numeric_level)
                logger.info(f"Log level changed to {new_level}")
            logger.info(f"Inference config reloaded: {state['cache']}")
    except FileNotFoundError:
        pass
    return state["cache"]


# ==============================================
# 文本后处理
# ==============================================
_PUNCT_PATTERN = re.compile(r'[^\w\s]')


def remove_punctuation(text: str) -> str:
    return _PUNCT_PATTERN.sub('', text)


# ==============================================
# Rollback
# ==============================================
try:
    import jieba as _jieba
    _jieba.setLogLevel(logging.WARNING)
    _jieba_available = True
except ImportError:
    _jieba_available = False


_CJK_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')


def _has_cjk(text: str) -> bool:
    return bool(_CJK_PATTERN.search(text))


def _rollback_words_by_space(text: str, n_words: int) -> int:
    """英文：按空格从尾部回退 N 个词，返回要删掉的字符数。"""
    count = 0
    idx = len(text)
    while idx > 0 and count < n_words:
        idx -= 1
        if idx == 0 or text[idx - 1] == ' ':
            count += 1
    return len(text) - idx


def _rollback_words_by_jieba(text: str, n_words: int) -> int:
    """中文：按 jieba 分词从尾部回退 N 个词，返回要删掉的字符数。"""
    segs = list(_jieba.cut(text))
    if n_words >= len(segs):
        return len(text)
    kept = "".join(segs[:-n_words])
    return len(text) - len(kept)


class RollbackConfig:
    VALID_STRATEGIES = ("none", "ratio", "chars", "words", "jieba")

    def __init__(self, strategy: str = "none", value: float = 0.0):
        if strategy not in self.VALID_STRATEGIES:
            strategy = "none"
        if strategy == "jieba":
            strategy = "words"
        if strategy == "words" and not _jieba_available:
            logger.warning(
                "jieba not installed, words strategy will only use space splitting for CJK text")
        self.strategy = strategy
        self.value = value

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'RollbackConfig':
        if not d or not d.get("enabled", False):
            return cls("none", 0.0)
        return cls(
            strategy=d.get("strategy", "none"),
            value=float(d.get("value", 0.0)),
        )

    def compute_rollback_chars(self, text: str) -> int:
        if not text or self.strategy == "none" or self.value <= 0:
            return 0
        if self.strategy == "ratio":
            return max(0, int(len(text) * min(self.value, 1.0)))
        elif self.strategy == "chars":
            return min(int(self.value), len(text))
        elif self.strategy == "words":
            n_words = int(self.value)
            if _jieba_available and _has_cjk(text):
                return _rollback_words_by_jieba(text, n_words)
            return _rollback_words_by_space(text, n_words)
        return 0

    def apply(self, text: str) -> str:
        rb = self.compute_rollback_chars(text)
        if rb <= 0:
            return text
        return text[:-rb]


def apply_history_rollback(
        history: str,
        rollback_config: Optional['RollbackConfig'] = None) -> str:
    """返回要拼接到 prompt 的历史文本。

    无回退: 拼接完整 history
    有回退: 去掉尾部不确定部分，只拼确定的前缀
    """
    if not history:
        return ""
    if rollback_config is None or rollback_config.strategy == "none":
        return history
    return rollback_config.apply(history)

# ==============================================
# Silero VAD 延迟加载
# ==============================================


# --- 将状态存在 sys 上避免 reload 时丢失 ---
_SILERO_REPO = os.environ.get("SILERO_REPO", "/path/to/silero-vad")
if not hasattr(sys, '_silero_vad_state'):
    sys._silero_vad_state = {"model": None, "get_ts": None, "available": False}


def _load_silero_vad():
    state = sys._silero_vad_state
    if state["model"] is not None:
        return True
    try:
        silero_src = os.path.join(_SILERO_REPO, "src")
        if silero_src not in sys.path:
            sys.path.insert(0, silero_src)
        from silero_vad.utils_vad import get_speech_timestamps, init_jit_model
        model_path = os.path.join(
            silero_src,
            "silero_vad",
            "data",
            "silero_vad.jit")
        state["model"] = init_jit_model(model_path)
        state["get_ts"] = get_speech_timestamps
        state["available"] = True
        logger.info(f"Silero VAD loaded from {model_path}")
        return True
    except Exception as e:
        logger.warning(
            f"Silero VAD load failed: {e}, falling back to energy VAD")
        state["available"] = False
        return False


# ==============================================
# 流式 VAD
# ==============================================
class StreamingVAD:
    """供 server.py 使用的流式 Silero VAD 检测器。
    每次调用 feed() 传入增量音频 chunk (float32, 16kHz)，
    返回 List[dict]，每个 dict 为 {"start": sample} 或 {"end": sample}。
    """

    def __init__(self, session_id: str, sr: int = 16000,
                 threshold: float = 0.5,
                 silence_duration_ms: int = 700,
                 prefix_padding_ms: int = 300):
        self.session_id = session_id
        self.sr = sr
        self.threshold = threshold
        self.silence_duration_ms = silence_duration_ms
        self.prefix_padding_ms = prefix_padding_ms
        self._total_samples = 0
        self._iterator = None
        self._init_iterator()

    def _init_iterator(self):
        state = sys._silero_vad_state
        if not state.get("available"):
            _load_silero_vad()
        if not state.get("available"):
            raise RuntimeError(
                f"[{self.session_id}] Silero VAD 加载失败，无法创建 StreamingVAD")
        silero_src = os.path.join(_SILERO_REPO, "src")
        if silero_src not in sys.path:
            sys.path.insert(0, silero_src)
        from silero_vad.utils_vad import VADIterator
        self._iterator = VADIterator(
            state["model"],
            threshold=self.threshold,
            sampling_rate=self.sr,
            min_silence_duration_ms=self.silence_duration_ms,
            speech_pad_ms=self.prefix_padding_ms,
        )
        logger.info(
            f"[{self.session_id}] StreamingVAD: silero iterator created "
            f"(threshold={self.threshold}, silence={self.silence_duration_ms}ms)")

    def feed(self, audio_chunk: np.ndarray) -> list:
        """传入增量音频 (float32, 16kHz)，返回事件列表。

        每个事件: {"start": sample_offset} 或 {"end": sample_offset}
        """
        events = []
        window = 512
        for i in range(0, len(audio_chunk), window):
            seg = audio_chunk[i:i + window]
            if len(seg) < window:
                seg = np.pad(seg, (0, window - len(seg)))
            t = torch.from_numpy(seg)
            result = self._iterator(t)
            if result is not None:
                if 'start' in result:
                    events.append({"start": result['start']})
                elif 'end' in result:
                    events.append({"end": result['end']})
        self._total_samples += len(audio_chunk)
        return events

    def reset(self):
        self._total_samples = 0
        if self._iterator is not None:
            try:
                self._iterator.reset_states()
            except Exception:
                pass

    @property
    def engine_name(self) -> str:
        return "silero"


def create_streaming_vad(
        session_id: str,
        turn_detection_cfg: dict) -> 'StreamingVAD':
    """工厂函数：根据 turn_detection 配置创建 StreamingVAD 实例。"""
    return StreamingVAD(
        session_id=session_id,
        threshold=turn_detection_cfg.get("threshold", 0.5),
        silence_duration_ms=turn_detection_cfg.get("silence_duration_ms", 700),
        prefix_padding_ms=turn_detection_cfg.get("prefix_padding_ms", 300),
    )


# ==============================================
# 特征提取 worker
# ==============================================
def process_mm_info_worker(messages, process_mm_info_fn):
    return process_mm_info_fn(messages, use_audio_in_video=True)

# ==============================================
# 核心推理函数
# ==============================================


async def run_omni_inference(
    audio_numpy: np.ndarray,
    prompt_text: str,
    session,  # AudioSession (duck-typed, 不直接 import 避免循环依赖)
    engine,
    processor,
    process_mm_info_fn,
    infer_tag: str = "",
) -> AsyncGenerator[str, None]:
    """
    封装调用 vllm-omni 引擎，yield 出本次推理的完整识别文本。
    engine/processor/process_mm_info_fn 从 server.py 传入，本模块不持有。
    返回值通过 session.accumulated_text 暴露给调用方。
    """
    from vllm.sampling_params import SamplingParams

    loop = asyncio.get_event_loop()
    sid = session.session_id
    tag = f"[{sid}] {infer_tag}" if infer_tag else f"[{sid}]"
    infer_t0 = time.time()

    len(audio_numpy) / 16000

    cfg = load_inference_config()

    messages = [
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_numpy},
            {"type": "text", "text": prompt_text}
        ]}
    ]

    feat_t0 = time.time()
    prompt_formatted = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    audios, _, _ = await loop.run_in_executor(
        None, process_mm_info_worker, messages, process_mm_info_fn)
    feat_cost = time.time() - feat_t0

    use_history = getattr(session, 'use_history', True)

    async with session.lock:
        history = session.accumulated_text
        rollback_cfg = session.rollback_config

    if use_history:
        text_to_append = apply_history_rollback(history, rollback_cfg)
    else:
        text_to_append = ""

    if text_to_append:
        prompt_formatted += text_to_append
        dropped = history[len(text_to_append):]
        logger.info(
            f"{tag} History: "
            f"\"{history[:60]}\"({len(history)}) "
            f"-> append \"{text_to_append[:60]}\"({len(text_to_append)}) "
            f"drop \"{dropped}\"({len(dropped)}) "
            f"[{rollback_cfg.strategy}/{rollback_cfg.value}]")
    else:
        reason = f"[{rollback_cfg.strategy}/{rollback_cfg.value}]" if use_history else "[use_history=False]"
        logger.info(f"{tag} History: (empty) {reason}")

    logger.debug(f"{tag} FinalPrompt (tail 200): ...{prompt_formatted[-200:]}")

    request_id = f"{sid}-{uuid.uuid4().hex[:6]}-{audios[0].shape[0]}"

    vllm_inputs = {
        "prompt": prompt_formatted,
        "multi_modal_data": {"audio": audios},
        "mm_processor_kwargs": {"use_audio_in_video": True},
        "limit_mm_per_prompt": {"audio": 1}
    }

    sp_cfg = cfg.get("sampling_params", {})
    sp = SamplingParams(
        temperature=sp_cfg.get("temperature", 0.01),
        top_p=sp_cfg.get("top_p", 0.1),
        top_k=sp_cfg.get("top_k", 1),
        max_tokens=sp_cfg.get("max_tokens", 512),
        seed=sp_cfg.get("seed", 42),
        detokenize=True,
        repetition_penalty=sp_cfg.get("repetition_penalty", 1.05)
    )
    sampling_params_list = [sp] * (len(engine.stage_list) if engine else 1)

    logger.debug(
        f"{tag} SamplingParams: temp={sp.temperature}, top_p={sp.top_p}, "
        f"top_k={sp.top_k}, max_tokens={sp.max_tokens}, rep_penalty={sp.repetition_penalty}")

    try:
        gen_t0 = time.time()
        previous_text = ""
        full_text = ""
        token_count = 0
        first_token_time = None

        async for output in engine.generate(
            prompt=vllm_inputs,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
            output_modalities=["text"]
        ):
            ro = getattr(output, "request_output", None) or output
            if not getattr(ro, "outputs", None):
                continue

            current_text = ro.outputs[0].text or ""
            delta_text = current_text[len(previous_text):]
            previous_text = current_text

            delta_text = remove_punctuation(delta_text)
            if delta_text:
                token_count += 1
                if first_token_time is None:
                    first_token_time = time.time()
                    logger.info(
                        f"{tag} FirstToken: \"{delta_text}\" latency={first_token_time - gen_t0:.3f}s")
                full_text += delta_text
                yield full_text

        gen_cost = time.time() - gen_t0
        total_cost = time.time() - infer_t0

        new_accumulated = text_to_append + full_text if use_history else full_text
        async with session.lock:
            old_accumulated = session.accumulated_text
            session.accumulated_text = new_accumulated

        logger.info(
            f"{tag} Done \"{full_text[:80]}\" | "
            f"accumulated: \"{old_accumulated[:40]}\" -> \"{new_accumulated[:60]}\" | "
            f"tokens={token_count} generate={gen_cost:.3f}s feat={feat_cost:.3f}s total={total_cost:.3f}s")
    except Exception as e:
        logger.error(f"{tag} Inference ERROR: {traceback.format_exc()}")
        yield f"[Error: {e}]"

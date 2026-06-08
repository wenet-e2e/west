# Copyright (c) 2026 Pengshen Zhang
"""Realtime Events: OpenAI Realtime 风格事件构造。

- session_snapshot: 输出 session.created/session.updated 的配置快照
- event builder: 构造 audio delta、commit、transcription、done 等 payload
- 只返回 dict，不直接操作 websocket，发送逻辑在 session/sender.py
"""
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from engine.runtime import ServiceRuntime

MISSING = object()


def session_snapshot(
        session: Any,
        session_id: str,
        service_runtime: "ServiceRuntime") -> dict:
    """构建符合 OpenAI Realtime 协议的 session 对象快照。"""
    cfg = service_runtime.inference_cfg.load()
    eff_td = cfg.turn_detection.type
    eff_prompt = session.asr.prompt or cfg.prompt
    return {
        "id": session_id,
        "object": "realtime.session",
        "model": service_runtime.settings.model_name,
        "instructions": eff_prompt,
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 16000},
                "turn_detection": {
                    "type": eff_td,
                    "threshold": cfg.turn_detection.threshold,
                    "prefix_padding_ms": cfg.turn_detection.prefix_padding_ms,
                    "silence_duration_ms": (
                        cfg.turn_detection.silence_duration_ms),
                },
            },
        },
        "extra": {
            "chunk_ms": session.asr.chunk_ms or cfg.chunk_ms,
            "use_history": session.asr.use_history,
            "history_rollback": {
                "enabled": (
                    session.asr.history_rollback_config.strategy
                    != "none"),
                "strategy": session.asr.history_rollback_config.strategy,
                "value": session.asr.history_rollback_config.value,
            },
        },
    }


def session_created(
        session: Any,
        session_id: str,
        service_runtime: "ServiceRuntime") -> dict:
    return {
        "type": "session.created",
        "session": session_snapshot(session, session_id, service_runtime),
    }


def session_updated(
        session: Any,
        session_id: str,
        service_runtime: "ServiceRuntime") -> dict:
    return {
        "type": "session.updated",
        "session": session_snapshot(session, session_id, service_runtime),
    }


def transcription_delta(
    cursor: int,
    delta: str,
    is_final: bool,
    chunk_id: Optional[int] = None,
    language: Any = MISSING,
) -> dict:
    msg = {
        "type": "conversation.item.input_audio_transcription.delta",
        "cursor": cursor,
        "delta": delta,
        "is_final": is_final,
    }
    if chunk_id is not None:
        msg["chunk_id"] = chunk_id
    if language is not MISSING:
        msg["language"] = language
    return msg


def transcription_completed(
    transcript: str,
    language: Optional[str],
) -> dict:
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": transcript,
        "language": language,
    }


def response_done() -> dict:
    return {"type": "response.done"}


def speech_started(audio_start_ms: int, item_id: str) -> dict:
    return {
        "type": "input_audio_buffer.speech_started",
        "audio_start_ms": audio_start_ms,
        "item_id": item_id,
    }


def speech_stopped(audio_end_ms: int, item_id: str) -> dict:
    return {
        "type": "input_audio_buffer.speech_stopped",
        "audio_end_ms": audio_end_ms,
        "item_id": item_id,
    }


def input_audio_committed(item_id: str) -> dict:
    return {
        "type": "input_audio_buffer.committed",
        "item_id": item_id,
    }

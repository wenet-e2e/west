# Copyright (c) 2026 Pengshen Zhang
"""Realtime Sender: WebSocket 事件发送封装。

- RealtimeSender: 包装 websocket.send 和 JSON 序列化
- session_created/session_updated: 发送连接配置快照
- delta/completed/done/VAD events: 统一发送 realtime_events 构造的 payload
- 让业务 loop 只关心语义事件，不拼 JSON 字符串
"""
import json
from typing import Any, Optional

import protocol


class RealtimeSender:
    """通过 WebSocket 发送标准 Realtime 协议事件。"""

    def __init__(self, websocket: Any):
        self.websocket = websocket

    async def send(self, obj: dict) -> None:
        await self.websocket.send(json.dumps(obj, ensure_ascii=False))

    async def session_created(
            self,
            session: Any,
            session_id: str,
            app: Any) -> None:
        await self.send(protocol.session_created(session, session_id, app))

    async def session_updated(
            self,
            session: Any,
            session_id: str,
            app: Any) -> None:
        await self.send(protocol.session_updated(session, session_id, app))

    async def delta(
        self,
        *,
        cursor: int,
        delta: str,
        is_final: bool,
        chunk_id: Optional[int] = None,
        language: Any = protocol.MISSING,
    ) -> None:
        await self.send(protocol.transcription_delta(
            cursor=cursor,
            delta=delta,
            is_final=is_final,
            chunk_id=chunk_id,
            language=language,
        ))

    async def completed(
        self,
        *,
        transcript: str,
        language: Optional[str],
    ) -> None:
        await self.send(protocol.transcription_completed(
            transcript=transcript,
            language=language,
        ))

    async def done(self) -> None:
        await self.send(protocol.response_done())

    async def speech_started(
            self,
            *,
            audio_start_ms: int,
            item_id: str) -> None:
        await self.send(protocol.speech_started(
            audio_start_ms=audio_start_ms,
            item_id=item_id,
        ))

    async def speech_stopped(self, *, audio_end_ms: int, item_id: str) -> None:
        await self.send(protocol.speech_stopped(
            audio_end_ms=audio_end_ms,
            item_id=item_id,
        ))

    async def input_audio_committed(self, *, item_id: str) -> None:
        await self.send(protocol.input_audio_committed(item_id=item_id))

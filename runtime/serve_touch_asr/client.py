# Copyright (c) 2026 Pengshen Zhang
"""Realtime Client: ASR WebSocket 命令行客户端。

- RealtimeASRClient: 读取音频、切包、发送 input_audio_buffer.append/commit
- transcribe_one/transcribe_batch: 支持单文件、列表文件和 JSONL 输出
- CLI 参数覆盖 streaming、chunk_ms、user_prompt、history_rollback 等 session 配置
- 用于本地调试、延迟评测和批量识别脚本
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import ssl
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import websockets
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stderr,
)
logger = logging.getLogger("ASRClient")

EVENT_TRANSCRIPTION_DELTA = (
    "conversation.item.input_audio_transcription.delta")
EVENT_TRANSCRIPTION_COMPLETED = (
    "conversation.item.input_audio_transcription.completed")
EVENT_TRANSCRIPTION_STARTED = (
    "conversation.item.input_audio_transcription.started")

REALTIME_ENDPOINT = "/v1/realtime"


def build_ws_endpoint(server_url: str) -> str:
    """Build the realtime endpoint from a host URL."""
    ws_url = server_url.rstrip('/')
    ws_url = ws_url.replace('http://', 'ws://', 1).replace(
        'https://', 'wss://', 1)
    if ws_url.endswith(REALTIME_ENDPOINT):
        raise ValueError(
            "--server 只需要服务地址，不要带 /v1/realtime；"
            f"错误示例: {server_url} -> 会拼成 {ws_url}{REALTIME_ENDPOINT}；"
            f"正确示例: {ws_url[:-len(REALTIME_ENDPOINT)]}"
        )
    return f"{ws_url}{REALTIME_ENDPOINT}"


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _ReceiveState:
    """receive_loop 与 transcribe_one 之间共享的可变状态。"""
    displayed_text: str = ""
    final_transcript: str = ""
    completed: asyncio.Event = dataclasses.field(
        default_factory=asyncio.Event)
    error_msg: Optional[str] = None
    delta_count: int = 0
    first_token_time: Optional[float] = None
    inference_start_time: Optional[float] = None
    last_token_time: Optional[float] = None
    last_token_chunk_id: Optional[int] = None
    last_token_is_final: bool = False
    chunk_last_token_times: Dict[int, float] = dataclasses.field(
        default_factory=dict)
    delta_records: List[Dict[str, Any]] = dataclasses.field(
        default_factory=list)


@dataclasses.dataclass
class _SendMetrics:
    """发送侧收集的时间戳与指标。"""
    first_chunk_sent_time: Optional[float] = None
    commit_time: Optional[float] = None
    chunk_sent_times: Dict[int, float] = dataclasses.field(
        default_factory=dict)


def _round_or_none(value: Optional[float]) -> Optional[float]:
    return round(value, 3) if value is not None else None


def parse_input(input_path: str) -> List[str]:
    """解析输入。

    - 如果是 wav/flac/mp3 等音频文件 → 返回 [该文件]
    - 如果是 txt/list 文本文件 → 逐行读取路径
    - 否则当作单个文件路径尝试
    """
    _, ext = os.path.splitext(input_path)
    if ext in ('.txt', '.list', '.lst', '.scp'):
        paths = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    paths.append(line)
        return paths
    return [input_path]


class RealtimeASRClient:
    """WebSocket 客户端，对接 server.py 的 /v1/realtime。

    参数:
        server_url:  WebSocket 地址 (ws://host:port)
        packet_ms:   每个发送包的音频时长 (毫秒)，模拟真实传输粒度
        sr:          音频采样率
        simulate_streaming: 是否模拟实时流式发送
            True  - 每发一个 packet sleep packet_ms（1 倍速实时）
            False - 尽快发完所有 packet
        history_rollback:    回退策略配置 dict (传给 session.update)
        chunk_ms:    服务端推理 chunk 触发阈值 (毫秒), None=使用服务端默认值
        system_prompt: Qwen3-Omni system prompt（映射到协议 instructions），
            None=使用服务端默认值
        user_prompt: Qwen3-Omni 当前任务指令（映射到 extra.user_prompt），
            None=使用服务端默认值
        context:     ASR 上下文/热词, None=使用服务端默认值
        language:    Qwen3-ASR 强制语种, None=使用服务端默认值
        use_history: 是否使用历史文本拼接 (默认 True)
    """
    def __init__(
        self,
        server_url: str = "ws://localhost:8001",
        packet_ms: int = 100,
        sr: int = 16000,
        simulate_streaming: bool = True,
        history_rollback: Optional[Dict[str, Any]] = None,
        chunk_ms: Optional[int] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        context: Optional[str] = None,
        language: Optional[str] = None,
        use_history: bool = True,
        record_deltas: bool = False,
    ):
        self.server_url = server_url.rstrip('/')
        self.ws_endpoint = build_ws_endpoint(server_url)
        self.packet_ms = packet_ms
        self.sr = sr
        self.simulate_streaming = simulate_streaming
        self.history_rollback = history_rollback
        self.chunk_ms = chunk_ms
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.context = context
        self.language = language
        self.use_history = use_history
        self.record_deltas = record_deltas

        self._ssl_context: Optional[ssl.SSLContext] = None
        if self.ws_endpoint.startswith("wss://"):
            self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            self._ssl_context.check_hostname = False
            self._ssl_context.verify_mode = ssl.CERT_NONE

    def _load_audio_pcm16(self, audio_path: str) -> np.ndarray:
        audio_f32, orig_sr = sf.read(audio_path,
                                     dtype='float32',
                                     always_2d=False)
        if audio_f32.ndim > 1:
            audio_f32 = audio_f32.mean(axis=1)
        if orig_sr != self.sr:
            ratio = self.sr / orig_sr
            out_len = int(len(audio_f32) * ratio)
            indices = np.arange(out_len) / ratio
            idx0 = np.floor(indices).astype(int)
            idx1 = np.minimum(idx0 + 1, len(audio_f32) - 1)
            frac = (indices - idx0).astype(np.float32)
            audio_f32 = audio_f32[idx0] * (1 - frac) + audio_f32[idx1] * frac
        return (audio_f32 * 32767).astype(np.int16)

    def _make_packets(self, pcm16: np.ndarray) -> List[np.ndarray]:
        """按 packet_ms 切分 PCM16 数组"""
        packet_samples = int(self.sr * self.packet_ms / 1000)
        packets = []
        for start in range(0, len(pcm16), packet_samples):
            end = min(start + packet_samples, len(pcm16))
            packets.append(pcm16[start:end])
        return packets

    # ------------------------------------------------------------------
    # transcribe_one 拆分的子方法
    # ------------------------------------------------------------------

    async def _connect_and_configure(
        self, ws, fname: str,
    ) -> str:
        """等待 session.created，发送 session.update，返回 session_id。"""
        raw = await ws.recv()
        created = json.loads(raw)
        session_id = created.get("session", {}).get("id", "")
        logger.info(f"[{fname}] Connected. session={session_id}")

        session_cfg: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        if self.history_rollback:
            extra["history_rollback"] = self.history_rollback
        if self.chunk_ms is not None:
            extra["chunk_ms"] = self.chunk_ms
        if not self.use_history:
            extra["use_history"] = False
        if self.user_prompt is not None:
            extra["user_prompt"] = self.user_prompt
        if self.system_prompt is not None:
            # instructions = 系统提示（对齐 OpenAI 语义）
            session_cfg["instructions"] = self.system_prompt
        if self.context is not None:
            extra["context"] = self.context
        if self.language is not None:
            extra["language"] = self.language
        if extra:
            session_cfg["extra"] = extra

        if session_cfg:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": session_cfg,
            }))
            await ws.recv()
            logger.info(f"[{fname}] Session config: {session_cfg}")

        return session_id

    async def _receive_loop(
        self, ws, rs: _ReceiveState,
        fname: str, t0: float,
    ) -> None:
        """接收服务端事件，更新 _ReceiveState。"""
        try:
            async for raw_msg in ws:
                msg = json.loads(raw_msg)
                msg_type = msg.get("type", "")

                if msg_type == EVENT_TRANSCRIPTION_DELTA:
                    cursor = msg.get("cursor", len(rs.displayed_text))
                    delta = msg.get("delta", "")
                    is_final = msg.get("is_final", False)
                    chunk_id = msg.get("chunk_id", 0)
                    old_len = len(rs.displayed_text)
                    rs.displayed_text = (
                        rs.displayed_text[:cursor] + delta)
                    rs.delta_count += 1

                    if self.record_deltas:
                        rs.delta_records.append({
                            "time": time.time() - t0,
                            "cursor": cursor,
                            "delta": delta,
                            "displayed_text": rs.displayed_text,
                            "is_final": is_final,
                            "chunk_id": chunk_id,
                        })

                    if delta or cursor < old_len:
                        token_time = time.time()
                        rs.last_token_time = token_time
                        rs.last_token_chunk_id = chunk_id
                        rs.last_token_is_final = is_final
                        if chunk_id > 0:
                            rs.chunk_last_token_times[chunk_id] = (
                                token_time)

                    if rs.first_token_time is None:
                        rs.first_token_time = time.time()
                    elapsed = time.time() - t0

                    if cursor < old_len:
                        rollback_chars = old_len - cursor
                        logger.info(
                            f"[{fname}] delta#{rs.delta_count} "
                            f"t={elapsed:.2f}s "
                            f"ROLLBACK {rollback_chars} chars "
                            f"cursor={cursor} +'{delta}' "
                            f"→ \"{rs.displayed_text}\"")
                    else:
                        logger.info(
                            f"[{fname}] delta#{rs.delta_count} "
                            f"t={elapsed:.2f}s "
                            f"cursor={cursor} +'{delta}' "
                            f"→ \"{rs.displayed_text}\"")

                    if is_final:
                        logger.info(
                            f"[{fname}] FINAL delta "
                            f"→ \"{rs.displayed_text}\"")

                elif msg_type == EVENT_TRANSCRIPTION_COMPLETED:
                    rs.final_transcript = msg.get(
                        "transcript", rs.displayed_text)
                    elapsed = time.time() - t0
                    logger.info(
                        f"[{fname}] Completed t={elapsed:.2f}s "
                        f"transcript=\"{rs.final_transcript}\"")

                elif msg_type == "response.done":
                    rs.completed.set()
                    return

                elif msg_type == EVENT_TRANSCRIPTION_STARTED:
                    rs.inference_start_time = time.time()
                    elapsed = time.time() - t0
                    logger.info(
                        f"[{fname}] Inference started "
                        f"t={elapsed:.2f}s")

        except websockets.ConnectionClosed:
            if not rs.completed.is_set():
                rs.error_msg = "connection_closed_unexpectedly"
                rs.completed.set()

    async def _send_audio(
        self, ws, packets: List[np.ndarray],
        audio_duration: float, fname: str, t0: float,
    ) -> _SendMetrics:
        """发送所有音频包并 commit，返回发送侧时间戳。"""
        sm = _SendMetrics()
        sleep_per_packet = (
            self.packet_ms / 1000.0 if self.simulate_streaming else 0)
        chunk_threshold = (
            self.chunk_ms if self.chunk_ms is not None else 1000)
        log_interval = max(1, len(packets) // 10)

        for i, pkt in enumerate(packets):
            await ws.send(pkt.tobytes())

            if (i + 1) % log_interval == 0 or i == len(packets) - 1:
                pkt_time = (i + 1) * self.packet_ms / 1000
                logger.info(
                    f"[{fname}] Sent pkt {i+1}/{len(packets)} "
                    f"({pkt_time:.2f}s / {audio_duration:.2f}s)")

            sent_audio_ms = (i + 1) * self.packet_ms
            current_chunk_idx = int(sent_audio_ms // chunk_threshold)

            if (current_chunk_idx > 0
                    and current_chunk_idx not in sm.chunk_sent_times):
                if sent_audio_ms >= current_chunk_idx * chunk_threshold:
                    sm.chunk_sent_times[current_chunk_idx] = time.time()

            if sm.first_chunk_sent_time is None:
                if sent_audio_ms >= chunk_threshold:
                    sm.first_chunk_sent_time = sm.chunk_sent_times.get(
                        1, time.time())

            if sleep_per_packet > 0:
                await asyncio.sleep(sleep_per_packet)

        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        sm.commit_time = time.time()
        elapsed = time.time() - t0
        logger.info(
            f"[{fname}] Commit sent at t={elapsed:.2f}s, "
            f"waiting for result...")
        return sm

    def _compute_result(
        self,
        audio_path: str,
        audio_duration: float,
        t0: float,
        rs: _ReceiveState,
        sm: _SendMetrics,
    ) -> Dict[str, Any]:
        """根据接收/发送状态计算最终结果与延迟指标。"""
        total_time = time.time() - t0
        text = rs.final_transcript or rs.displayed_text
        send_time = (sm.commit_time - t0) if sm.commit_time else 0
        fname = os.path.basename(audio_path)

        if self.simulate_streaming:
            chunk_decode_times: Dict[int, float] = {}
            for chunk_id, token_time in sorted(
                    rs.chunk_last_token_times.items()):
                upload_done_time = sm.chunk_sent_times.get(chunk_id)
                if upload_done_time is None:
                    upload_done_time = sm.commit_time
                if upload_done_time is None:
                    continue
                decode_time = max(0.0, token_time - upload_done_time)
                chunk_decode_times[chunk_id] = decode_time
            processing_time = (sum(chunk_decode_times.values())
                               if chunk_decode_times
                               else total_time - send_time)
        else:
            chunk_decode_times = {}
            processing_time = total_time
        rtf = (max(0, processing_time / audio_duration)
               if audio_duration > 0 else 0)

        logger.info(
            f"[{fname}] Done. time={total_time:.2f}s "
            f"decode_time={processing_time:.3f}s "
            f"RTF={rtf:.3f} deltas={rs.delta_count} "
            f"text=\"{text[:80]}\"")

        first_token_delay = (
            (rs.first_token_time - sm.first_chunk_sent_time)
            if rs.first_token_time and sm.first_chunk_sent_time
            else None)
        if first_token_delay is None and rs.first_token_time is not None:
            first_token_delay = rs.first_token_time - t0

        commit_to_result = (
            (rs.first_token_time - sm.commit_time)
            if rs.first_token_time and sm.commit_time
            else None)
        inference_latency = (
            (rs.inference_start_time - sm.commit_time)
            if rs.inference_start_time and sm.commit_time
            else None)

        last_chunk_delay = None
        if rs.last_token_time is not None:
            if rs.last_token_is_final:
                ref_time = sm.commit_time if sm.commit_time else t0
            else:
                if (rs.last_token_chunk_id is not None
                        and rs.last_token_chunk_id in sm.chunk_sent_times):
                    ref_time = sm.chunk_sent_times[rs.last_token_chunk_id]
                else:
                    ref_time = sm.commit_time if sm.commit_time else t0
            last_chunk_delay = rs.last_token_time - ref_time

        if first_token_delay is not None:
            logger.info(
                f"[{fname}] First-token latency (from 1st chunk): "
                f"{first_token_delay:.3f}s")
        if last_chunk_delay is not None:
            logger.info(
                f"[{fname}] Last-chunk latency (tail token): "
                f"{last_chunk_delay:.3f}s")

        result: Dict[str, Any] = {
            "audio_path": audio_path,
            "text": text,
            "audio_duration": round(audio_duration, 2),
            "total_time": round(total_time, 2),
            "rtf": round(rtf, 3),
            "decode_time": round(processing_time, 3),
            "chunk_decode_times": {
                str(k): round(v, 3)
                for k, v in chunk_decode_times.items()
            },
            "send_time": round(send_time, 3),
            "first_token_delay": _round_or_none(first_token_delay),
            "commit_to_result": _round_or_none(commit_to_result),
            "inference_latency": _round_or_none(inference_latency),
            "last_chunk_delay": _round_or_none(last_chunk_delay),
            "per_chunk_delay": round(total_time - send_time, 3),
            "num_deltas": rs.delta_count,
            "status": "failed" if rs.error_msg else "success",
        }
        if self.record_deltas:
            result["deltas"] = rs.delta_records
        if rs.error_msg:
            result["error"] = rs.error_msg
        return result

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    async def transcribe_one(self, audio_path: str) -> Dict[str, Any]:
        """通过 WebSocket 转录单个音频文件。"""
        if not os.path.exists(audio_path):
            logger.error(f"File not found: {audio_path}")
            return {
                "audio_path": audio_path,
                "text": "",
                "status": "failed",
                "error": "file_not_found",
            }

        fname = os.path.basename(audio_path)
        pcm16 = self._load_audio_pcm16(audio_path)
        packets = self._make_packets(pcm16)
        audio_duration = len(pcm16) / self.sr

        logger.info(
            f"[{fname}] duration={audio_duration:.2f}s, "
            f"packets={len(packets)}, packet_ms={self.packet_ms}, "
            f"streaming={'realtime' if self.simulate_streaming else 'fast'}")

        rs = _ReceiveState()
        t0 = time.time()

        async with websockets.connect(
            self.ws_endpoint,
            max_size=50 * 1024 * 1024,
            open_timeout=30,
            ping_interval=30,
            ping_timeout=60,
            ssl=self._ssl_context,
        ) as ws:
            await self._connect_and_configure(ws, fname)

            recv_task = asyncio.create_task(
                self._receive_loop(ws, rs, fname, t0))
            sm = await self._send_audio(
                ws, packets, audio_duration, fname, t0)

            try:
                await asyncio.wait_for(
                    rs.completed.wait(), timeout=120)
            except asyncio.TimeoutError:
                rs.error_msg = "timeout_waiting_for_response"

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

        return self._compute_result(
            audio_path, audio_duration, t0, rs, sm)

    async def transcribe_batch(
        self,
        audio_paths: List[str],
        output_file: Optional[str] = None,
        output_format: str = "jsonl",
    ) -> List[Dict[str, Any]]:
        """逐个转录多个音频文件，边处理边写入防止丢失。"""
        results: List[Dict[str, Any]] = []
        fout = (open(output_file, 'w', encoding='utf-8')
                if output_file else None)
        try:
            pbar = tqdm(audio_paths,
                        desc="ASR",
                        file=sys.stderr,
                        dynamic_ncols=True)
            for audio_path in pbar:
                pbar.set_postfix_str(
                    os.path.basename(audio_path)[:25],
                    refresh=False)
                result = await self.transcribe_one(audio_path)
                results.append(result)
                if fout:
                    if output_format == 'jsonl':
                        fout.write(
                            json.dumps(result, ensure_ascii=False)
                            + '\n')
                    else:
                        fname = os.path.basename(result['audio_path'])
                        text = result.get('text', '')
                        fout.write(f"{fname}\t{text}\n")
                    fout.flush()
        finally:
            if fout:
                fout.close()
        return results

    def run(
        self,
        audio_paths: List[str],
        output_file: Optional[str] = None,
        output_format: str = "jsonl",
    ) -> List[Dict[str, Any]]:
        """同步入口，统一走 transcribe_batch。"""
        return asyncio.run(
            self.transcribe_batch(
                audio_paths,
                output_file=output_file,
                output_format=output_format))


def _write_results_to_stdout(
    results: List[Dict[str, Any]],
    output_format: str,
) -> None:
    for result in results:
        if output_format == 'jsonl':
            sys.stdout.write(
                json.dumps(result, ensure_ascii=False) + '\n')
        else:
            fname = os.path.basename(result['audio_path'])
            text = result.get('text', '')
            sys.stdout.write(f"{fname}\t{text}\n")


def main():
    """Parse CLI options and run batch transcription."""
    parser = argparse.ArgumentParser(
        description='Qwen3-Omni Realtime ASR WebSocket Client',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单文件
  python client.py -i test.wav

  # 批量文件列表，输出到文件
  python client.py -i wav_list.txt -o results.txt

  # 模拟实时流式发送（1倍速）
  python client.py -i test.wav --simulate-streaming

  # 自定义 packet 大小
  python client.py -i test.wav --packet-ms 200

  # 带回退
  python client.py -i test.wav \\
      --history-rollback-strategy ratio --history-rollback-value 0.2

  # 覆盖服务端 chunk_ms（减少推理次数，提高吞吐）
  python client.py -i wav_list.txt --chunk-ms 3000

  # 自定义 user prompt
  python client.py -i test.wav --user-prompt "Transcribe the English audio."

  # 使用历史文本拼接
  python client.py -i test.wav --use-history
""")
    parser.add_argument(
        '--server',
        '-s',
        type=str,
        default='ws://localhost:8001',
        help='WebSocket server URL (default: ws://localhost:8001)')
    parser.add_argument('--input',
                        '-i',
                        type=str,
                        required=True,
                        help='wav 文件或 .txt/.list/.scp 路径列表')
    parser.add_argument('--output',
                        '-o',
                        type=str,
                        default=None,
                        help='输出文本文件路径（默认输出到 stdout）')
    parser.add_argument('--output-format',
                        type=str,
                        default='text',
                        choices=['text', 'jsonl'],
                        help='输出格式: text 或 jsonl（含延迟指标）')

    stream_group = parser.add_argument_group('流式传输配置')
    stream_group.add_argument('--packet-ms',
                              type=int,
                              default=100,
                              help='每包音频时长 ms (default: 100)')
    stream_group.add_argument('--sr',
                              type=int,
                              default=16000,
                              help='采样率 (default: 16000)')
    stream_group.add_argument(
        '--simulate-streaming',
        action='store_true',
        help=('模拟流式发送（仿照人说话速度，每发一包 sleep '
              'packet_ms），不加此参数则极速快发'))

    infer_group = parser.add_argument_group('推理配置 (覆盖服务端默认值)')
    infer_group.add_argument('--chunk-ms',
                             type=int,
                             default=None,
                             help='推理 chunk 阈值 ms（默认服务端配置）')
    infer_group.add_argument('--user-prompt',
                             type=str,
                             default=None,
                             help='Qwen3-Omni 当前任务指令 (默认服务端配置)')
    infer_group.add_argument('--system-prompt',
                             type=str,
                             default=None,
                             help='Qwen3-Omni system prompt (默认服务端配置)')
    infer_group.add_argument('--context',
                             type=str,
                             default=None,
                             help='ASR 上下文/热词 (默认服务端配置)')
    infer_group.add_argument('--language',
                             type=str,
                             default=None,
                             help='Qwen3-ASR 强制语种，如 Chinese (默认服务端配置)')
    infer_group.add_argument(
        '--use-history',
        action='store_true',
        help='启用历史文本拼接（默认不拼接）')
    infer_group.add_argument(
        '--history-rollback-strategy',
        type=str,
        default='none',
        choices=['none', 'ratio', 'chars', 'words', 'tokens'],
        help=('回退策略: none=不回退, ratio=按比例, chars=按字符数, '
              'words=按词数（中文 jieba）, tokens=按 token 数（推荐）'))
    infer_group.add_argument('--history-rollback-value',
                             type=float,
                             default=0.0,
                             help='回退参数值')

    parser.add_argument('--record-deltas',
                        action='store_true',
                        help='JSONL 输出记录每个 delta 的时间与文本')
    parser.add_argument('--quiet',
                        '-q',
                        action='store_true',
                        help='仅输出结果，抑制日志')

    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    history_rollback = None
    if args.history_rollback_strategy != 'none':
        if not args.use_history:
            logger.warning(
                '未开启 --use-history 时 history-rollback 无效，已忽略')
        elif args.history_rollback_value <= 0:
            logger.warning(
                f'history-rollback-value={args.history_rollback_value} '
                f'<= 0，已忽略')
        else:
            if (args.history_rollback_strategy == 'ratio'
                    and args.history_rollback_value > 1.0):
                logger.warning(
                    f'ratio 策略 value 应在 (0,1]，'
                    f'当前 {args.history_rollback_value} 将被截为 1.0')
            history_rollback = {
                "enabled": True,
                "strategy": args.history_rollback_strategy,
                "value": args.history_rollback_value,
            }
    elif args.history_rollback_value > 0:
        logger.warning(
            f'history-rollback-value={args.history_rollback_value} '
            f'但 strategy=none，已忽略')

    client = RealtimeASRClient(
        server_url=args.server,
        packet_ms=args.packet_ms,
        sr=args.sr,
        simulate_streaming=args.simulate_streaming,
        history_rollback=history_rollback,
        chunk_ms=args.chunk_ms,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        context=args.context,
        language=args.language,
        use_history=args.use_history,
        record_deltas=args.record_deltas,
    )

    audio_paths = parse_input(args.input)
    if not audio_paths:
        logger.error("No valid audio files found")
        sys.exit(1)

    logger.info(
        f"Input: {len(audio_paths)} file(s), "
        f"packet_ms={args.packet_ms}, "
        f"streaming={'realtime' if args.simulate_streaming else 'fast'}")

    t0 = time.time()
    results = client.run(audio_paths,
                         output_file=args.output,
                         output_format=args.output_format)
    elapsed = time.time() - t0

    if not args.output:
        _write_results_to_stdout(results, args.output_format)

    success = sum(1 for r in results if r.get('status') == 'success')
    failed = len(results) - success
    total_audio = sum(r.get('audio_duration', 0) for r in results)
    avg_rtf = elapsed / total_audio if total_audio > 0 else 0

    logger.info(
        f"Summary: {len(results)} file(s), "
        f"success={success}, failed={failed}, "
        f"audio={total_audio:.1f}s, wall={elapsed:.1f}s, "
        f"avg_RTF={avg_rtf:.2f}")
    if args.output:
        logger.info(f"Output saved to: {args.output}")


if __name__ == '__main__':
    main()

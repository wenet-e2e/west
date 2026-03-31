# Copyright (c) 2026 Pengshen Zhang
# ==============================================
# Qwen3-Omni Realtime ASR WebSocket Client
#
# 对接 server.py 的 /v1/realtime WebSocket 端点
# 特性：
# - 支持单文件 / 列表批量测试
# - 真实模拟流式发包（按 chunk_ms 切片）
# - 终端实时打印增量（delta）和覆盖回退过程
# ==============================================

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import websockets
from tqdm import tqdm

# ==============================================
# Logging
# ==============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stderr,
)
logger = logging.getLogger("ASRClient")


def parse_input(input_path: str) -> List[str]:
    """解析输入。

    - 如果是 wav/flac/mp3 等音频文件 → 返回 [该文件]
    - 如果是 txt/list 文本文件 → 逐行读取路径
    - 否则当作单个文件路径尝试
    """
    _, ext = os.path.splitext(input_path)
    if ext in ('.txt', '.list', '.scp'):
        paths = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    paths.append(line)
        return paths
    return [input_path]


# ==============================================
# WebSocket Realtime ASR Client
# ==============================================
class RealtimeASRClient:
    """WebSocket 客户端，对接 server.py 的 /v1/realtime。

    参数:
        server_url:  WebSocket 地址 (ws://host:port)
        packet_ms:   每个发送包的音频时长 (毫秒)，模拟真实传输粒度
        sr:          音频采样率
        simulate_streaming: 是否模拟实时流式发送
            True  - 每发一个 packet sleep packet_ms（1 倍速实时）
            False - 尽快发完所有 packet
        rollback:    回退策略配置 dict (传给 session.update)
        chunk_ms:    服务端推理 chunk 触发阈值 (毫秒), None=使用服务端默认值
        prompt:      ASR prompt, None=使用服务端默认值
        use_history: 是否使用历史文本拼接 (默认 True)
    """

    def __init__(
        self,
        server_url: str = "ws://localhost:8001",
        packet_ms: int = 100,
        sr: int = 16000,
        simulate_streaming: bool = True,
        rollback: Optional[Dict[str, Any]] = None,
        chunk_ms: Optional[int] = None,
        prompt: Optional[str] = None,
        use_history: bool = True,
    ):
        self.server_url = server_url.rstrip('/')
        ws_base = self.server_url.replace(
            'http://', 'ws://').replace('https://', 'wss://')
        self.ws_endpoint = f"{ws_base}/v1/realtime"
        self.packet_ms = packet_ms
        self.sr = sr
        self.simulate_streaming = simulate_streaming
        self.rollback = rollback
        self.chunk_ms = chunk_ms
        self.prompt = prompt
        self.use_history = use_history

    def _load_audio_pcm16(self, audio_path: str) -> np.ndarray:
        audio_f32, orig_sr = sf.read(
            audio_path, dtype='float32', always_2d=False)
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

    async def transcribe_one(self, audio_path: str) -> Dict[str, Any]:
        """通过 WebSocket 转录单个音频文件"""
        if not os.path.exists(audio_path):
            logger.error(f"File not found: {audio_path}")
            return {"audio_path": audio_path, "text": "",
                    "status": "failed", "error": "file_not_found"}

        fname = os.path.basename(audio_path)
        pcm16 = self._load_audio_pcm16(audio_path)
        packets = self._make_packets(pcm16)
        audio_duration = len(pcm16) / self.sr
        sleep_per_packet = (
            self.packet_ms / 1000.0 if self.simulate_streaming else 0
        )

        logger.info(
            f"[{fname}] duration={audio_duration:.2f}s, "
            f"packets={len(packets)}, packet_ms={self.packet_ms}, "
            f"streaming={'realtime' if self.simulate_streaming else 'fast'}")

        displayed_text = ""
        final_transcript = ""
        completed_event = asyncio.Event()
        error_msg = None
        t0 = time.time()
        delta_count = 0
        first_token_time = None
        commit_time = None
        inference_start_time = None

        async with websockets.connect(
            self.ws_endpoint,
            max_size=50 * 1024 * 1024,
            ping_interval=30,
            ping_timeout=60,
        ) as ws:
            raw = await ws.recv()
            created = json.loads(raw)
            session_id = created.get("session", {}).get("id", "")
            logger.info(f"[{fname}] Connected. session={session_id}")

            session_update = {}
            if self.rollback:
                session_update["rollback"] = self.rollback
            if self.chunk_ms is not None:
                session_update["chunk_ms"] = self.chunk_ms
            if self.prompt is not None:
                session_update["prompt"] = self.prompt
            if not self.use_history:
                session_update["use_history"] = False

            if session_update:
                await ws.send(json.dumps({
                    "type": "session.update",
                    "session": session_update,
                }))
                await ws.recv()
                logger.info(f"[{fname}] Session config: {session_update}")

            async def receive_loop():
                nonlocal displayed_text, final_transcript
                nonlocal error_msg, delta_count
                nonlocal first_token_time, inference_start_time
                try:
                    async for raw_msg in ws:
                        msg = json.loads(raw_msg)
                        msg_type = msg.get("type", "")

                        if msg_type == "conversation.item.input_audio_transcription.delta":
                            cursor = msg.get("cursor", len(displayed_text))
                            delta = msg.get("delta", "")
                            is_final = msg.get("is_final", False)
                            old_len = len(displayed_text)
                            displayed_text = displayed_text[:cursor] + delta
                            delta_count += 1
                            if first_token_time is None:
                                first_token_time = time.time()
                            elapsed = time.time() - t0

                            if cursor < old_len:
                                rollback_chars = old_len - cursor
                                logger.info(
                                    f"[{fname}] delta#{delta_count} "
                                    f"t={elapsed:.2f}s "
                                    f"ROLLBACK {rollback_chars} chars "
                                    f"cursor={cursor} +'{delta}' "
                                    f"→ \"{displayed_text}\"")
                            else:
                                logger.info(
                                    f"[{fname}] delta#{delta_count} "
                                    f"t={elapsed:.2f}s "
                                    f"cursor={cursor} +'{delta}' "
                                    f"→ \"{displayed_text}\"")

                            if is_final:
                                logger.info(
                                    f"[{fname}] FINAL delta "
                                    f"→ \"{displayed_text}\"")

                        elif msg_type == "conversation.item.input_audio_transcription.completed":
                            final_transcript = msg.get(
                                "transcript", displayed_text)
                            elapsed = time.time() - t0
                            logger.info(
                                f"[{fname}] Completed t={elapsed:.2f}s "
                                f"transcript=\"{final_transcript}\"")

                        elif msg_type == "response.done":
                            completed_event.set()
                            return

                        elif msg_type == "conversation.item.input_audio_transcription.started":
                            inference_start_time = time.time()
                            elapsed = time.time() - t0
                            logger.info(
                                f"[{fname}] Inference started "
                                f"t={elapsed:.2f}s")

                except websockets.ConnectionClosed:
                    if not completed_event.is_set():
                        error_msg = "connection_closed_unexpectedly"
                        completed_event.set()

            recv_task = asyncio.create_task(receive_loop())

            for i, pkt in enumerate(packets):
                b64 = base64.b64encode(pkt.tobytes()).decode('utf-8')
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": b64,
                }))

                if (i + 1) % max(1, len(packets) //
                                 10) == 0 or i == len(packets) - 1:
                    pkt_time = (i + 1) * self.packet_ms / 1000
                    logger.info(
                        f"[{fname}] Sent pkt {i+1}/{len(packets)} "
                        f"({pkt_time:.2f}s / {audio_duration:.2f}s)")

                if sleep_per_packet > 0:
                    await asyncio.sleep(sleep_per_packet)

            await ws.send(json.dumps({
                "type": "input_audio_buffer.commit"}))
            commit_time = time.time()
            elapsed = time.time() - t0
            logger.info(
                f"[{fname}] Commit sent at t={elapsed:.2f}s, "
                f"waiting for result...")

            try:
                await asyncio.wait_for(
                    completed_event.wait(), timeout=120)
            except asyncio.TimeoutError:
                error_msg = "timeout_waiting_for_response"

            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

        total_time = time.time() - t0
        text = final_transcript or displayed_text
        rtf = total_time / audio_duration if audio_duration > 0 else 0

        logger.info(
            f"[{fname}] Done. time={total_time:.2f}s "
            f"RTF={rtf:.2f} deltas={delta_count} "
            f"text=\"{text[:80]}\"")

        send_time = (commit_time - t0) if commit_time else 0
        first_token_delay = (
            (first_token_time - t0)
            if first_token_time else None
        )
        commit_to_result = (
            (first_token_time - commit_time)
            if first_token_time and commit_time else None
        )
        inference_latency = (
            (inference_start_time - commit_time)
            if inference_start_time and commit_time else None
        )

        result = {
            "audio_path": audio_path,
            "text": text,
            "audio_duration": round(
                audio_duration,
                2),
            "total_time": round(
                total_time,
                2),
            "rtf": round(
                rtf,
                3),
            "send_time": round(
                send_time,
                3),
            "first_token_delay": round(
                first_token_delay,
                3) if first_token_delay is not None else None,
            "commit_to_result": round(
                commit_to_result,
                3) if commit_to_result is not None else None,
            "inference_latency": round(
                inference_latency,
                3) if inference_latency is not None else None,
            "num_deltas": delta_count,
            "status": "failed" if error_msg else "success",
        }
        if error_msg:
            result["error"] = error_msg
        return result

    async def batch_transcribe(
        self, audio_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """逐个转录多个音频文件"""
        results = []
        pbar = tqdm(
            audio_paths, desc="ASR", file=sys.stderr,
            dynamic_ncols=True)
        for audio_path in pbar:
            pbar.set_postfix_str(
                os.path.basename(audio_path)[:25], refresh=False)
            result = await self.transcribe_one(audio_path)
            results.append(result)
        return results

    def run(self, audio_paths: List[str]) -> List[Dict[str, Any]]:
        """同步入口"""
        if len(audio_paths) == 1:
            return [asyncio.run(self.transcribe_one(audio_paths[0]))]
        return asyncio.run(self.batch_transcribe(audio_paths))


# ==============================================
# CLI
# ==============================================
def main():
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
  python client.py -i test.wav --rollback-strategy ratio --rollback-value 0.2

  # 覆盖服务端 chunk_ms（减少推理次数，提高吞吐）
  python client.py -i wav_list.txt --chunk-ms 3000

  # 自定义 prompt
  python client.py -i test.wav --prompt "Transcribe the English audio."

  # 关闭历史拼接（每次推理独立，不累积上下文）
  python client.py -i test.wav --use-history false
""")
    parser.add_argument(
        '--server', '-s', type=str, default='ws://localhost:8001',
        help='WebSocket server URL (default: ws://localhost:8001)')
    parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='单个 wav 文件，或文本列表文件 (.txt/.list/.scp，每行一个路径)')
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='输出文本文件路径（默认输出到 stdout）')
    parser.add_argument(
        '--output-format', type=str, default='text',
        choices=['text', 'jsonl'],
        help='输出格式: text=每行"文件名\\t文本", jsonl=每行完整JSON(含延迟指标)')

    stream_group = parser.add_argument_group('流式传输配置')
    stream_group.add_argument(
        '--packet-ms', type=int, default=100,
        help='每包音频时长 ms (default: 100)')
    stream_group.add_argument(
        '--sr', type=int, default=16000,
        help='采样率 (default: 16000)')
    stream_group.add_argument(
        '--simulate-streaming', action='store_true',
        help='模拟实时流式发送（1倍速），不加此参数则尽快发完')

    infer_group = parser.add_argument_group('推理配置 (覆盖服务端默认值)')
    infer_group.add_argument(
        '--chunk-ms', type=int, default=None,
        help='服务端推理 chunk 触发阈值 ms (default: 使用服务端配置)')
    infer_group.add_argument(
        '--prompt', type=str, default=None,
        help='ASR prompt (default: 使用服务端配置)')
    infer_group.add_argument(
        '--use-history', type=str, default='true', choices=['true', 'false'],
        help='是否使用历史文本拼接 (default: true, 设为 false 则每次推理独立)')
    infer_group.add_argument(
        '--rollback-strategy', type=str, default='none',
        choices=['none', 'ratio', 'chars', 'words'],
        help='回退策略: none=不回退, ratio=按比例, chars=按字符数, words=按词数 (中文自动用jieba，英文用空格)')
    infer_group.add_argument(
        '--rollback-value', type=float, default=0.0,
        help='回退参数值')

    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='仅输出结果，抑制日志')

    args = parser.parse_args()

    if args.quiet:
        logger.setLevel(logging.WARNING)

    use_history = args.use_history == 'true'

    rollback = None
    if args.rollback_strategy != 'none':
        if not use_history:
            logger.warning(
                "--use-history=false 时 --rollback-strategy 无效（没有历史可回退），已忽略")
        elif args.rollback_value <= 0:
            logger.warning(
                f"--rollback-strategy={args.rollback_strategy} 但 --rollback-value={args.rollback_value} <= 0，已忽略")
        else:
            if args.rollback_strategy == 'ratio' and args.rollback_value > 1.0:
                logger.warning(
                    f"--rollback-strategy=ratio 时 value 应在 (0,1]，"
                    f"当前 {args.rollback_value} 将被 server 端 clamp 到 1.0")
            rollback = {
                "enabled": True,
                "strategy": args.rollback_strategy,
                "value": args.rollback_value,
            }
    elif args.rollback_value > 0:
        logger.warning(
            f"--rollback-value={args.rollback_value} 但 --rollback-strategy=none，value 将被忽略")

    client = RealtimeASRClient(
        server_url=args.server,
        packet_ms=args.packet_ms,
        sr=args.sr,
        simulate_streaming=args.simulate_streaming,
        rollback=rollback,
        chunk_ms=args.chunk_ms,
        prompt=args.prompt,
        use_history=use_history,
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
    results = client.run(audio_paths)
    elapsed = time.time() - t0

    fout = (open(args.output, 'w', encoding='utf-8')
            if args.output else sys.stdout)
    try:
        for result in results:
            if args.output_format == 'jsonl':
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            else:
                fname = os.path.basename(result['audio_path'])
                text = result.get('text', '')
                fout.write(f"{fname}\t{text}\n")
    finally:
        if args.output:
            fout.close()

    # 汇总统计
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

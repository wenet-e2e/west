# Copyright (c) 2026 Pengshen Zhang
"""Session: 单连接生命周期管理。

- state: VadState / ConnectionLoopState / StreamingAsrState / ConnectionContext
- session: RealtimeSession（音频缓冲 + reset）
- sender: RealtimeSender（协议事件发送）
- inference_loop: 推理轮循环
- vad_loop: VAD 检测循环
"""

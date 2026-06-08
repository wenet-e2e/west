# Copyright (c) 2026 Pengshen Zhang
"""L3 Model: 推理算法 / rollback / 后处理 / VAD 算法。

- model_inference: vLLM 引擎调用 + 流式 yield
- history_rollback: rollback 策略与 apply 逻辑
- transcript_postprocess: 文本后处理
- vad_engine / vad_silence: VAD provider/detector 与静音检测
"""

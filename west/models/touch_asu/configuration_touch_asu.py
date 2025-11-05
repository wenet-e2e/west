# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from typing import Any, Dict, Optional

from transformers import PretrainedConfig


class TouchASUConfig(PretrainedConfig):
    model_type = "touch_asu"

    def __init__(
        self,
        llm_model_name_or_path: str = 'Qwen/Qwen2-7B',
        wenet_model_name_or_path: str = '',
        encoder_ds_rate: int = 4,
        encoder_projector_ds_rate: int = 2,
        projector_hidden_size: int = 2048,
        hidden_size: int = 0,  # Will override in TouchASU Model
        lora_config: Optional[Dict[str, Any]] = None,
        max_speech_frames: int = 2000,  # 20s
        min_speech_frames: int = 20,  # 0.2s
        projector_type: str = 'conv1d',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_model_name_or_path = llm_model_name_or_path
        self.wenet_model_name_or_path = wenet_model_name_or_path
        self.encoder_ds_rate = encoder_ds_rate
        self.encoder_projector_ds_rate = encoder_projector_ds_rate
        self.projector_hidden_size = projector_hidden_size
        self.lora_config = lora_config
        self.hidden_size = hidden_size
        self.max_speech_frames = max_speech_frames
        self.min_speech_frames = min_speech_frames
        self.projector_type = projector_type


__all__ = ["TouchASUConfig"]

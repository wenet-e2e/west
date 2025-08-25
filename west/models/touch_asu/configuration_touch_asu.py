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


__all__ = ["TouchASUConfig"]

# Copyright (c) 2025 Xuelong Geng(xlgeng@mail.nwpu.edu.cn)

from typing import Any, Dict, Optional

from transformers import PretrainedConfig


class OSUMEChatConfig(PretrainedConfig):
    model_type = "osum_echat"

    def __init__(
        self,
        llm_model_name_or_path: str = 'Qwen/Qwen2.5-3B-Instruct',
        no_init_llm: bool = True,
        wenet_model_name_or_path: str = 'whisper-medium',
        lora_config: Optional[Dict[str, Any]] = None,
        speech_token_num: int = 4097,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_model_name_or_path = llm_model_name_or_path
        self.no_init_llm = no_init_llm
        self.wenet_model_name_or_path = wenet_model_name_or_path
        self.lora_config = lora_config
        self.speech_token_num = speech_token_num


__all__ = ["OSUMEChatConfig"]

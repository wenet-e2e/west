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


class OSUMConfig(PretrainedConfig):
    model_type = "osum"

    def __init__(
        self,
        llm_model_name_or_path: str = 'Qwen/Qwen2.5-3B-Instruct',
        no_init_llm: bool = True,
        wenet_model_name_or_path: str = 'whisper-medium',
        encoder_ds_rate: int = 2,
        encoder_projector_ds_rate: int = 4,
        hidden_size: int = 0,  # Will override in OSUM Model
        lora_config: Optional[Dict[str, Any]] = None,
        freeze_encoder: bool = False,
        freeze_llm: bool = False,
        speech_token_num: int = 4097,
        prompt_conf_path: str = 'conf/prompt_config.yaml',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_model_name_or_path = llm_model_name_or_path
        self.no_init_llm = no_init_llm
        self.wenet_model_name_or_path = wenet_model_name_or_path
        self.encoder_ds_rate = encoder_ds_rate
        self.encoder_projector_ds_rate = encoder_projector_ds_rate
        self.hidden_size = hidden_size
        self.lora_config = lora_config
        self.freeze_encoder = freeze_encoder
        self.freeze_llm = freeze_llm
        self.speech_token_num = speech_token_num
        self.prompt_conf_path = prompt_conf_path


__all__ = ["OSUMEChatConfig", "OSUMConfig"]

# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from typing import Optional

from transformers import PretrainedConfig


class TouchFlowConfig(PretrainedConfig):
    model_type = 'touch_flow'

    def __init__(
        self,
        llm_model_name_or_path: str = '',
        s3tokenizer_model_name_or_path: str = '',
        speaker_model_path: Optional[str] = '',
        text_tokenizer_path: Optional[str] = '',
        num_speech_tokens: int = 4096,
        t_scheduler: Optional[str] = 'cosine',
        sigma_min: float = 1e-6,
        training_cfg_rate: float = 0.2,
        hidden_size: int = 0,
        inference_cfg_rate: float = 0.7,
        n_timesteps: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_model_name_or_path = llm_model_name_or_path
        self.s3tokenizer_model_name_or_path = s3tokenizer_model_name_or_path
        self.speaker_model_path = speaker_model_path
        self.num_speech_tokens = num_speech_tokens
        self.t_scheduler = t_scheduler
        self.sigma_min = sigma_min
        self.training_cfg_rate = training_cfg_rate
        self.hidden_size = hidden_size
        self.inference_cfg_rate = inference_cfg_rate
        self.n_timesteps = n_timesteps


__all__ = ["TouchFlowConfig"]

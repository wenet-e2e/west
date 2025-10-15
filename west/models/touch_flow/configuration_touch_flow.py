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
        max_speech_duration: float = 30,
        min_speech_duration: float = 0.2,
        decoding_chunk_size: int = 0,
        enable_full_context: bool = True,
        max_chunk_size: int = 86,
        num_decoding_left_chunks: int = 0,
        static_chunk_size: int = -1,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_model_name_or_path = llm_model_name_or_path
        self.s3tokenizer_model_name_or_path = s3tokenizer_model_name_or_path
        self.speaker_model_path = speaker_model_path
        self.text_tokenizer_path = text_tokenizer_path
        self.num_speech_tokens = num_speech_tokens
        self.t_scheduler = t_scheduler
        self.sigma_min = sigma_min
        self.training_cfg_rate = training_cfg_rate
        self.hidden_size = hidden_size
        self.inference_cfg_rate = inference_cfg_rate
        self.n_timesteps = n_timesteps
        self.max_speech_duration = max_speech_duration
        self.min_speech_duration = min_speech_duration
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.decoding_chunk_size = decoding_chunk_size
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks
        self.enable_full_context = enable_full_context
        self.max_chunk_size = max_chunk_size


__all__ = ["TouchFlowConfig"]

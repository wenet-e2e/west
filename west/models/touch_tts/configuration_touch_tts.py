# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from transformers import PretrainedConfig


class TouchTTSConfig(PretrainedConfig):
    model_type = "touch_tts"

    def __init__(
        self,
        llm_model_name_or_path: str = 'Qwen/Qwen2-7B',
        s3tokenizer_model_name_or_path: str = '',
        num_speech_tokens: int = 4096,
        hidden_size: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_model_name_or_path = llm_model_name_or_path
        self.s3tokenizer_model_name_or_path = s3tokenizer_model_name_or_path
        self.num_speech_tokens = num_speech_tokens
        self.hidden_size = hidden_size


__all__ = ["TouchTTSConfig"]

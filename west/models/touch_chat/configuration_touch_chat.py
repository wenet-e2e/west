# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from transformers import PretrainedConfig


class TouchChatConfig(PretrainedConfig):
    model_type = "touch_chat"

    def __init__(
        self,
        thinker_model_path: str = '',
        talker_model_path: str = '',
        projector_hidden_size: int = 0,
        hidden_size: int = 0,  # Will override in TouchChat Model
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.thinker_model_path = thinker_model_path
        self.talker_model_path = talker_model_path
        self.projector_hidden_size = projector_hidden_size
        self.hidden_size = hidden_size


__all__ = ["TouchChatConfig"]

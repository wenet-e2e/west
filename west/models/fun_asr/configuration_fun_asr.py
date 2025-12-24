# Copyright (c) 2025 Chengdong Liang(liangchengdongd@qq.com)

from west.models.touch_asu.configuration_touch_asu import TouchASUConfig


class FunASRConfig(TouchASUConfig):
    model_type = "fun_asr"

    def __init__(self, pretrained_checkpoint: str = None, **kwargs):
        super().__init__(**kwargs)
        self.pretrained_checkpoint = pretrained_checkpoint


__all__ = ["FunASRConfig"]

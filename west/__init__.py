# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from transformers import AutoConfig, AutoModel

from west.models.touch_asu import TouchASU, TouchASUConfig  # noqa


AutoConfig.register("touch_asu", TouchASUConfig)
AutoModel.register(TouchASUConfig, TouchASU)


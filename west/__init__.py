# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from transformers import AutoConfig, AutoModel

from west.models.touch_asu import TouchASU, TouchASUConfig
from west.models.touch_flow import TouchFlow, TouchFlowConfig
from west.models.touch_tts import TouchTTS, TouchTTSConfig

AutoConfig.register("touch_asu", TouchASUConfig)
AutoModel.register(TouchASUConfig, TouchASU)
AutoConfig.register("touch_flow", TouchFlowConfig)
AutoModel.register(TouchFlowConfig, TouchFlow)
AutoConfig.register("touch_tts", TouchTTSConfig)
AutoModel.register(TouchTTSConfig, TouchTTS)

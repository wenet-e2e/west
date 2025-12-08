# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from transformers import AutoConfig, AutoModel

from west.models.fireredasr_llm import FireredASRLLM, FireredASRLLMConfig
from west.models.osum_echat import OSUMEChat, OSUMEChatConfig
from west.models.touch_asu import TouchASU, TouchASUConfig
from west.models.touch_chat import TouchChat, TouchChatConfig
from west.models.touch_flow import TouchFlow, TouchFlowConfig
from west.models.touch_tts import TouchTTS, TouchTTSConfig

AutoConfig.register("touch_asu", TouchASUConfig)
AutoModel.register(TouchASUConfig, TouchASU)
AutoConfig.register("touch_chat", TouchChatConfig)
AutoModel.register(TouchChatConfig, TouchChat)
AutoConfig.register("touch_flow", TouchFlowConfig)
AutoModel.register(TouchFlowConfig, TouchFlow)
AutoConfig.register("touch_tts", TouchTTSConfig)
AutoModel.register(TouchTTSConfig, TouchTTS)

AutoConfig.register("osum_echat", OSUMEChatConfig)
AutoModel.register(OSUMEChatConfig, OSUMEChat)
AutoConfig.register("fireredasr_llm", FireredASRLLMConfig)
AutoModel.register(FireredASRLLMConfig, FireredASRLLM)

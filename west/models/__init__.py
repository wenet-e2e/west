from transformers import AutoConfig, AutoModel

from west.models.touch_asu import TouchASU, TouchASUConfig  # noqa
# from west.models.touch_flow import TouchFlow  # noqa
# from west.models.touch_tts import TouchTTS  # noqa
#

AutoConfig.register("touch_asu", TouchASUConfig)
AutoModel.register(TouchASUConfig, TouchASU)

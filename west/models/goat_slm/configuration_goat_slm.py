# -*- coding: utf-8 -*-
"""
@Author : songyd, chenhj
@File   : configuration_goat_slm.py
"""
from transformers import (CONFIG_MAPPING, PretrainedConfig, WhisperConfig,
                          logging)

from .configuration_transformer_adapter import AdapterConfig
from .modeling_telechat3 import Telechat3Config

logger = logging.get_logger(__name__)


class GOATSLMConfig(PretrainedConfig):

    def __init__(self,
                 whisper_config=None,
                 llm_config=None,
                 adapter_config=None,
                 conv_kernel_sizes="5,5,5",
                 adapter_inner_dim=512,
                 **kwargs):
        super().__init__(**kwargs)

        if whisper_config is None:
            whisper_config = {}
            logger.info(
                "whisper config is None. "
                "Initializing the WhisperConfig with default values"
            )

        if llm_config is None:
            llm_config = {}
            logger.info(
                "llm config is None. Initializing the llm with default values")

        if adapter_config is None:
            adapter_config = {}

        self.whisper_config = WhisperConfig(**whisper_config).to_dict()
        if llm_config:
            model_type = llm_config.get("model_type")
            if model_type in CONFIG_MAPPING:
                config_class = CONFIG_MAPPING[model_type]
                config = config_class.from_dict(llm_config)
            elif model_type == "telechat3":
                config = Telechat3Config.from_dict(llm_config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.llm_config = config.to_dict()
        else:
            self.llm_config = {}

        self.adapter_config = AdapterConfig(**adapter_config).to_dict()

        self.conv_kernel_sizes = conv_kernel_sizes
        self.adapter_inner_dim = adapter_inner_dim

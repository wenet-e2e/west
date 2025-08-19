# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from abc import ABC, abstractmethod


class Extractor(ABC):

    model_type = 'model'
    _registry = {}

    # Batch/Pack fileds for dataset
    fields_batch_static = {}
    fields_batch_dynamic = {}
    fields_pack_offset = {}

    def __init__(self, tokenizer, inference=False):
        self.tokenizer = tokenizer
        self.inference = inference

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.model_type.lower()] = cls

    @classmethod
    def get_class(cls, model_type):
        return cls._registry[model_type]

    @classmethod
    def get_extractor(cls, config):
        extractor_class = cls.get_class(config.model_type)
        return extractor_class(config)

    @abstractmethod
    def extract(self, item):
        pass

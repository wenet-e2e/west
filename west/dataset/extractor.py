# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from abc import ABC, abstractmethod


class Extractor(ABC):

    extractor_type = 'base_extractor'
    _registry = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.extractor_type.lower()] = cls

    @classmethod
    def get_class(cls, extractor_type):
        return cls._registry[extractor_type]

    @classmethod
    def get_extractor(cls, config):
        extractor_class = cls.get_class(config.extractor_type)
        return extractor_class(config)

    @abstractmethod
    def extract(self, item):
        pass

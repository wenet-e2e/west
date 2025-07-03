# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from dataclasses import dataclass


@dataclass
class ModelArgs:
    model_type: str = 'speech_llm'

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def register(cls, data_class):
        cls.__dataclass_fields__.update(data_class.__dataclass_fields__)
        cls.__annotations__.update(data_class.__annotations__)
        return data_class


class Model:
    model_type = 'model'
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.model_type.lower()] = cls

    @classmethod
    def get_class(cls, model_type):
        return cls._registry[model_type]

    @classmethod
    def get_model(cls, config):
        model_class = cls.get_class(config.model_type)
        return model_class(config)

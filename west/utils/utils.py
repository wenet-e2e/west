# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)


def freeze_module(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 Hao Yin(1049755192@qq.com)

import re


def get_tn_model(language):
    """仅加载指定语言所需的 TN 模型。"""
    if language == "zh":
        from tn.chinese.normalizer import Normalizer as ZhNormalizer
        return ZhNormalizer(overwrite_cache=False)
    elif language == "en":
        from tn.english.normalizer import Normalizer as EnNormalizer
        return EnNormalizer(overwrite_cache=False)
    else:
        raise ValueError(f"Unsupported language: {language}")


def normalize_text(text, language, tn_model):
    """对文本做 TN 归一化并只保留目标语言字符。"""
    if language == "zh":
        text = tn_model.normalize(text)
        # 只保留中文和英文字符
        text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbfa-zA-Z]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    elif language == "en":
        text = tn_model.normalize(text)
        text = re.sub(r"[^a-zA-Z0-9']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
    else:
        raise ValueError(f"Unsupported language: {language}")
    return text

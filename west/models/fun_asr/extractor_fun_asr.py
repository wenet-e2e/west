# Copyright (c) 2025 Chengdong Liang(liangchengdongd@qq.com)

import math

import torch
import wenet

from west.models.touch_asu.extractor_touch_asu import ExtractorTouchASU


class ExtractorFunASR(ExtractorTouchASU):
    model_type = "fun_asr"

    def __init__(self, tokenizer, model_config, inference=False):
        super().__init__(tokenizer, model_config, inference)
        self.compute_feature, self.feature_dim = self.load_feature()
        self.ds_rate = 8   # hard code
        self.default_prompt_text = '语音转写：'

    def load_feature(self):
        compute_feature, feature_dim = wenet.load_feature(
            self.model_config.wenet_model_name_or_path)

        def _warp_compute_feature(audio):
            mel = compute_feature(audio)
            mel = self.apply_lfr(mel)
            return mel
        return _warp_compute_feature, feature_dim

    def apply_lfr(self, inputs, lfr_m=7, lfr_n=6):
        T = inputs.shape[0]
        T_lfr = int(math.ceil(T / lfr_n))
        left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
        inputs = torch.vstack((left_padding, inputs))
        T = T + (lfr_m - 1) // 2
        feat_dim = inputs.shape[-1]
        strides = (lfr_n * feat_dim, 1)
        sizes = (T_lfr, lfr_m * feat_dim)
        last_idx = (T - lfr_m) // lfr_n + 1
        num_padding = lfr_m - (T - last_idx * lfr_n)
        if num_padding > 0:
            num_padding = (2 * lfr_m - 2 * T + (
                T_lfr - 1 + last_idx) * lfr_n) / 2 * (T_lfr - last_idx)
            inputs = torch.vstack([inputs] + [inputs[-1:]] * int(num_padding))
        LFR_outputs = inputs.as_strided(sizes, strides)
        return LFR_outputs.clone().type(torch.float32)

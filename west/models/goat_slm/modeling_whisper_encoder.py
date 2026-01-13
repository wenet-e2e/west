# -*- coding: utf-8 -*-
"""
@Author : songyd, chenhj
@File   : modeling_whisper_encoder.py
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import \
    WhisperEncoder as HFWhisperEncoder
from transformers.utils import ModelOutput


@dataclass
class WhisperOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    output_lengths: Optional[torch.LongTensor] = None


class WhisperEncoder(HFWhisperEncoder):
    """
    overwrite forward to support attention_mask
    overwrite from_pretrained to support split encoder parameters
    from pretrained WhisperModel
    """

    def from_pretrained(model_path):
        config = WhisperConfig.from_pretrained(model_path)

        model = WhisperEncoder(config)
        old_state_dict = torch.load(
            os.path.join(model_path, "pytorch_model.bin"))
        state_dict = {}
        for para_name in old_state_dict.keys():
            if "model.encoder." in para_name:
                new_name = para_name.replace("model.encoder.", "")
                state_dict[new_name] = old_state_dict[para_name]
        model.load_state_dict(state_dict)

        return model

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output = super().forward(input_features, attention_mask, head_mask,
                                 output_attentions, output_hidden_states,
                                 return_dict)

        last_hidden_state = output.last_hidden_state  # B x T x C
        input_lengths = attention_mask.sum(-1)
        output_lengths = self._get_feat_extract_output_lengths(input_lengths)
        max_length = output_lengths.max()
        last_hidden_state = last_hidden_state[:, :max_length, :]

        return WhisperOutput(last_hidden_state=last_hidden_state,
                             hidden_states=None,
                             attentions=None,
                             output_lengths=output_lengths)

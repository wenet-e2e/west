# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, GenerationMixin, PreTrainedModel

from west.utils.utils import freeze_module

from .configuration_touch_chat import TouchChatConfig


class TouchChat(PreTrainedModel, GenerationMixin):
    """ LLM based end to end Chat.
        TouchChat consists of pretrained 'thinker' and 'talker'.
    """
    model_type = 'touch_chat'
    config_class = TouchChatConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: TouchChatConfig):
        super().__init__(config)
        self.thinker = AutoModel.from_pretrained(config.thinker_model_path)
        self.talker = AutoModel.from_pretrained(config.talker_model_path)
        self.config.hidden_size = self.thinker.config.hidden_size
        proj_dim = self.config.projector_hidden_size
        self.projector = nn.Sequential(
            nn.Linear(self.thinker.config.hidden_size, proj_dim),
            torch.nn.SiLU(),
            nn.Linear(proj_dim, self.talker.config.hidden_size),
        )
        print(self.projector)
        freeze_module(self.thinker)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        talker_features: Optional[torch.FloatTensor] = None,
        talker_features_lengths: Optional[torch.LongTensor] = None,
        talker_offsets: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        # TODO(Binbin Zhang): Whether to use loss of thinker
        thinker_out = self.thinker(
            input_ids,
            attention_mask,
            labels=None,  # Do not compute loss on thinker
            position_ids=position_ids,
            audio_offsets=audio_offsets,
            audio_features=audio_features,
            audio_features_lengths=audio_features_lengths,
            batch_idx=batch_idx,
            has_audio=has_audio,
            output_hidden_states=True,  # return hidden states
            **kwargs)
        hidden_state = thinker_out.hidden_states[-1]  # last hidden
        hidden_embs = self.projector(hidden_state)
        out = self.talker(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          position_ids=position_ids,
                          audio_offsets=talker_offsets,
                          audio_features=talker_features,
                          audio_features_lengths=talker_features_lengths,
                          batch_idx=batch_idx,
                          inputs_embeds=hidden_embs,
                          **kwargs)
        return out

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        thinker_out = self.thinker.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_offsets=audio_offsets,
            audio_features=audio_features,
            audio_features_lengths=audio_features_lengths,
            batch_idx=batch_idx,
            has_audio=has_audio,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True)
        text_lengths = torch.tensor([len(thinker_out.sequences[0])],
                                    dtype=torch.long,
                                    device=input_ids.device)
        print('text len', text_lengths)
        print(self.tokenizer.batch_decode(thinker_out.sequences.tolist()))
        hidden_state = torch.cat([x[-1] for x in thinker_out.hidden_states],
                                 dim=1)
        hidden_embs = self.projector(hidden_state)
        model_outputs = self.talker.generate(
            text_lengths=text_lengths,
            inputs_embeds=hidden_embs,
        )
        print(model_outputs)
        return model_outputs

    def init_tokenizer(self):
        # Here we assume thinker and talker shares the same tokenizer
        self.tokenizer = self.talker.init_tokenizer()
        return self.tokenizer

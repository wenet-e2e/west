# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from typing import Optional

import safetensors
import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel

from .configuration_touch_chat import TouchChatConfig


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class TouchChat(PreTrainedModel):
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
        freeze_model(self.thinker)
        self._keys_to_ignore_on_save = set()
        for k in self.thinker.state_dict().keys():
            self._keys_to_ignore_on_save.add('thinker.' + k)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args,
                        **kwargs):
        """ The default `from_pretrained` does not init the parameters
            of `self.llm` and `self.encoder`, so we custom it.
        """
        config = TouchChatConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        weights_path = f"{pretrained_model_name_or_path}/model.safetensors"
        state_dict = safetensors.torch.load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)
        return model

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
                          input_embs=hidden_embs,
                          **kwargs)
        return out

    def init_tokenizer(self):
        return self.thinker.init_tokenizer()

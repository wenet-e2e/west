# Copyright (c) 2025 Chengdong Liang(liangchengdongd@qq.com)

import torch.nn as nn

from west.models.touch_asu.modeling_touch_asu import TouchASU

from .configuration_fireredasr_llm import FireredASRLLMConfig
from .convert_fireredasr_llm_weights_to_west import convert_to_west_state_dict


class ProjectorLinear(nn.Module):
    def __init__(self, config, encoder_dim, llm_dim):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.linear1 = nn.Linear(encoder_dim * self.k,
                                 config.projector_hidden_size)
        self.linear2 = nn.Linear(config.projector_hidden_size, llm_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.k, feat_dim * self.k
        )

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class FireredASRLLM(TouchASU):
    """
    FireredASRLLM model.
    Paper link: https://arxiv.org/pdf/2501.14350
    Model link: https://huggingface.co/FireRedTeam/FireRedASR-LLM-L/tree/main
    """
    model_type = "fireredasr_llm"
    config_class = FireredASRLLMConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: FireredASRLLMConfig):
        super().__init__(config)
        self.projector = ProjectorLinear(config,
                                         self.encoder.encoder.output_size(),
                                         config.hidden_size)

    @classmethod
    def _from_config(cls, config, **kwargs):
        model = super()._from_config(config, **kwargs)
        if config.pretrained_checkpoint is not None:
            state_dict = convert_to_west_state_dict(
                config.pretrained_checkpoint)
            model.load_state_dict(state_dict, strict=False)
        return model

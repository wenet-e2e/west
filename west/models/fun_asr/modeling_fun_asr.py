# Copyright (c) 2025 Chengdong Liang(liangchengdongd@qq.com)

import torch
import torch.nn as nn
from wenet.models.transformer.attention import MultiHeadedAttention
from wenet.models.transformer.encoder_layer import TransformerEncoderLayer
from wenet.models.transformer.positionwise_feed_forward import \
    PositionwiseFeedForward
from wenet.utils.mask import make_pad_mask

from west.models.touch_asu.modeling_touch_asu import TouchASU

from .configuration_fun_asr import FunASRConfig


class ProjectorTransformer(nn.Module):
    def __init__(self, config, encoder_dim, llm_dim, ffn_dim=2048, n_layers=2):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.linear1 = nn.Linear(encoder_dim * self.k, ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_dim, llm_dim)
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                size=llm_dim,
                self_attn=MultiHeadedAttention(
                    n_head=8,
                    n_feat=llm_dim,
                    dropout_rate=0.0,
                ),
                feed_forward=PositionwiseFeedForward(
                    idim=llm_dim,
                    hidden_units=llm_dim // 4,
                    dropout_rate=0.0,
                ),
                dropout_rate=0.0,
                normalize_before=True,
                layer_norm_type="layer_norm",
                norm_eps=1e-12,
            ) for _ in range(n_layers)
        ])

    def forward(self, x, ilens=None):
        batch_size, seq_len, dim = x.size()
        chunk_num = (seq_len - 1) // self.k + 1
        pad_num = chunk_num * self.k - seq_len
        x = nn.functional.pad(x, (0, 0, 0, pad_num, 0, 0), value=0.0)
        seq_len = x.size(1)
        x = x.contiguous().view(batch_size, chunk_num, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        olens = None
        olens = (ilens - 1) // self.k + 1
        masks = (~make_pad_mask(olens, seq_len)).unsqueeze(1)
        for block in self.blocks:
            x, masks, _, _ = block(x, masks, None, None)
        return x, olens


class FunASR(TouchASU):
    """
    FunASR: https://github.com/FunAudioLLM/Fun-ASR
    """
    model_type = "fun_asr"
    config_class = FunASRConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: FunASRConfig):
        super().__init__(config)
        self.projector = ProjectorTransformer(
            config, self.encoder.encoder.output_size(), config.hidden_size,
        )

    @classmethod
    def _from_config(cls, config, **kwargs):
        model = super()._from_config(config, **kwargs)
        if config.pretrained_checkpoint is not None:
            state_dict = convert_to_west_state_dict(
                config.pretrained_checkpoint)
            model.load_state_dict(state_dict, strict=False)
        return model

    def get_speech_embeddings(self, audio_features, audio_features_lengths):
        speech_emb, mask = self.encoder._forward_encoder(
            audio_features, audio_features_lengths)
        speech_emb_lens = mask.sum(-1).squeeze(-1)
        speech_proj, _ = self.projector(speech_emb, speech_emb_lens)
        speech_proj_lens = (audio_features_lengths - 1) // 8 + 1
        return speech_proj, speech_proj_lens


def convert_to_west_state_dict(pretrained_checkpoint):
    state_dict = torch.load(pretrained_checkpoint,
                            map_location="cpu")["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("audio_adaptor."):
            new_key = key.replace("audio_adaptor.", "projector.")
        elif key.startswith("audio_encoder."):
            new_key = key.replace("audio_encoder.", "encoder.encoder.")
        new_state_dict[new_key] = value
    return new_state_dict

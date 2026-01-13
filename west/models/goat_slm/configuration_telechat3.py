# -*- coding: utf-8 -*-
"""
@Author : songyd, chenhj
@File   : configuration_telechat3.py
"""
# coding=utf-8
# Copyright 2022 the Big Science Workshop and HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Telechat configuration

from transformers.configuration_utils import PretrainedConfig


class Telechat3Config(PretrainedConfig):
    model_type = "telechat3"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=1,
        embedding_size=1024,
        eos_token_id=2,
        head_dim=128,
        hidden_act="silu",
        hidden_size=6144,
        initializer_range=0.0048,
        intermediate_size=24576,
        max_position_embeddings=2048,
        mlp_bias=False,
        model_type="telechat3",
        num_attention_heads=48,
        num_hidden_layers=64,
        num_key_value_heads=None,
        pad_token_id=None,
        pretraining_tp=1,
        rms_norm_eps=1e-5,
        rope_scaling=None,
        rope_theta=1000000.0,
        share_attention=True,
        share_ffn=False,
        tie_word_embeddings=False,
        use_cache=True,
        vocab_size=131072,
        **kwargs,
    ):
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.share_attention = share_attention
        self.share_ffn = share_ffn

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.initializer_range = initializer_range

        self.pretraining_tp = pretraining_tp
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.use_cache = use_cache
        self.vocab_size = vocab_size

        if (head_dim is not None
                and head_dim != self.hidden_size // self.num_attention_heads):
            raise ValueError(
                "head_dim != hidden_size//num_attention_head. "
                "Please check the config."
            )
        self.head_dim = (
            head_dim if head_dim is not None
            else self.hidden_size // self.num_attention_heads
        )

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

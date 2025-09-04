# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from typing import Optional

import torch
import wenet
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel)

from west.utils.utils import freeze_module

from .configuration_touch_asu import TouchASUConfig


class ProjectorCov1d(nn.Module):

    def __init__(self, config, encoder_dim, llm_dim):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.conv1d = nn.Conv1d(in_channels=encoder_dim,
                                out_channels=encoder_dim,
                                kernel_size=self.k,
                                stride=self.k,
                                padding=0)
        self.linear1 = nn.Linear(encoder_dim, config.projector_hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(config.projector_hidden_size, llm_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x


class TouchASU(PreTrainedModel):
    """ LLM based Automatic Speech Understanding
    """
    model_type = 'touch_asu'
    config_class = TouchASUConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: TouchASUConfig):
        super().__init__(config)
        llm_config = AutoConfig.from_pretrained(config.llm_model_name_or_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name_or_path,
            config=llm_config,
            torch_dtype='auto',
            attn_implementation="flash_attention_2",  # or "flex_attention"
        )
        self.encoder = wenet.load_model_pt(config.wenet_model_name_or_path)
        encoder_dim = self.encoder.encoder.output_size()
        config.hidden_size = llm_config.hidden_size  # for deepseed training
        self.projector = ProjectorCov1d(config, encoder_dim,
                                        llm_config.hidden_size)
        total_params = sum(p.numel() for p in self.projector.parameters())
        print('Projector total params: {:.2f}M'.format(total_params / 1024 /
                                                       1024))
        if config.lora_config is not None:
            lora_config = LoraConfig(**config.lora_config)
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()

        self.freeze_encoder()
        if config.lora_config is None:
            self.freeze_llm()

    def tie_weights(self):
        return self.llm.tie_weights()

    def get_speech_embeddings(self, audio_features, audio_features_lengths):
        speech_emb, mask = self.encoder._forward_encoder(
            audio_features, audio_features_lengths)
        speech_emb = speech_emb.masked_fill(~mask.transpose(1, 2), 0.0)
        speech_proj = self.projector(speech_emb)
        speech_proj_lens = mask.squeeze(1).sum(1) // self.projector.k
        return speech_proj, speech_proj_lens

    def compute_mix_embedding(
        self,
        input_ids: torch.LongTensor = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
    ):
        text_emb = self.llm.get_input_embeddings()(input_ids)
        speech_emb, speech_emb_lens = self.get_speech_embeddings(
            audio_features, audio_features_lengths)
        inputs_embeds = text_emb
        for i in range(audio_features.size(0)):
            if not has_audio[i]:
                continue
            b = batch_idx[i]
            s, e = audio_offsets[i], audio_offsets[i] + speech_emb_lens[i]
            inputs_embeds[b, s:e, :] = speech_emb[i, :speech_emb_lens[i], :]
        return inputs_embeds

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
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.compute_mix_embedding(
            input_ids,
            audio_offsets,
            audio_features,
            audio_features_lengths,
            batch_idx,
            has_audio,
        )
        out = self.llm(inputs_embeds=inputs_embeds,
                       attention_mask=attention_mask,
                       labels=labels,
                       position_ids=position_ids,
                       **kwargs)
        return out

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        has_audio: Optional[torch.BoolTensor] = None,
        eos_token_id=None,
        decode_config=None,
        **kwargs,
    ):
        inputs_embeds = self.compute_mix_embedding(
            input_ids,
            audio_offsets,
            audio_features,
            audio_features_lengths,
            batch_idx,
            has_audio,
        )
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,  # TODO(Binbin Zhang): Fix me
            top_p=1.0,  # TODO(Binbin Zhang): Fix me
            eos_token_id=eos_token_id,
            **kwargs,
        )
        return model_outputs

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()

    def freeze_encoder(self):
        freeze_module(self.encoder)
        self.encoder.eval()

    def freeze_llm(self):
        freeze_module(self.llm)

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model_name_or_path,
            padding_side="right",
        )
        if 'Qwen' in self.config.llm_model_name_or_path:
            tokenizer.bos_token = tokenizer.eos_token
        elif 'llama' in self.config.llm_model_name_or_path:
            tokenizer.pad_token = '<|finetune_right_pad_id|>'
        return tokenizer

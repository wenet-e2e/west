# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import logging
from dataclasses import dataclass, field
from typing import Optional

import safetensors
import torch
import transformers
import wenet
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from west.models.model import Model, ModelArgs


@ModelArgs.register
@dataclass
class SpeechLLMArgs:
    llm_model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-7B")
    wenet_model_name_or_path: Optional[str] = field(default="")
    encoder_ds_rate: int = 2
    encoder_projector_ds_rate: int = 5
    projector_hidden_size: int = 2048
    projector_model_path: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length"},
    )


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


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class SpeechLLM(PreTrainedModel, Model):
    model_type = 'speech_llm'
    supports_gradient_checkpointing = True

    def __init__(self, config: SpeechLLMArgs):
        llm_config = transformers.AutoConfig.from_pretrained(
            config.llm_model_name_or_path)
        llm_config.use_cache = False
        super().__init__(llm_config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = wenet.load_model_pt(config.wenet_model_name_or_path)
        self.encoder = encoder.to(device)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name_or_path,
            config=llm_config,
            torch_dtype='auto',
            attn_implementation="flash_attention_2",  # or "flex_attention"
        )
        encoder_dim = encoder.encoder.output_size()
        llm_dim = llm_config.hidden_size
        self.projector = ProjectorCov1d(config, encoder_dim, llm_dim)
        total_params = sum(p.numel() for p in self.projector.parameters())
        print('Projector total params: {:.2f}M'.format(total_params / 1024 /
                                                       1024))
        if config.projector_model_path is not None:
            self.load_projector(config.projector_model_path)
        self.freeze_encoder()
        self.freeze_llm()
        self._keys_to_ignore_on_save = set()
        # Do not save the parameter of llm and speech encoder
        for k in self.llm.state_dict().keys():
            self._keys_to_ignore_on_save.add('llm.' + k)
        for k in self.encoder.state_dict().keys():
            self._keys_to_ignore_on_save.add('encoder.' + k)
        self.num_sentences = 0

    def get_speech_embeddings(self, audio_features, audio_feature_lengths):
        speech_emb, mask = self.encoder._forward_encoder(
            audio_features, audio_feature_lengths)
        speech_emb = speech_emb.masked_fill(~mask.transpose(1, 2), 0.0)
        speech_proj = self.projector(speech_emb)
        speech_proj_lens = mask.squeeze(1).sum(1) // self.projector.k
        return speech_proj, speech_proj_lens

    def compute_mix_embedding(
        self,
        input_ids: torch.LongTensor = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
    ):
        text_emb = self.llm.get_input_embeddings()(input_ids)
        speech_emb, speech_emb_lens = self.get_speech_embeddings(
            audio_features, audio_feature_lengths)
        inputs_embeds = text_emb
        for i in range(audio_features.size(0)):
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
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.compute_mix_embedding(
            input_ids,
            audio_offsets,
            audio_features,
            audio_feature_lengths,
            batch_idx,
        )
        out = self.llm(inputs_embeds=inputs_embeds,
                       attention_mask=attention_mask,
                       labels=labels,
                       position_ids=position_ids,
                       **kwargs)
        self.num_sentences += audio_features.size(0)
        logging.info('Train finish {} sentences'.format(self.num_sentences))
        return out

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        eos_token_id=None,
        decode_config=None,
    ):
        inputs_embeds = self.compute_mix_embedding(
            input_ids,
            audio_offsets,
            audio_features,
            audio_feature_lengths,
            batch_idx,
        )
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            top_p=1.0,
            num_beams=decode_config.num_beams,
            max_new_tokens=decode_config.max_new_tokens,
            eos_token_id=eos_token_id,
        )
        return model_outputs

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()

    def freeze_encoder(self):
        freeze_model(self.encoder)
        self.encoder.eval()

    def freeze_llm(self):
        freeze_model(self.llm)

    def load_projector(self, projector_path):
        projector_state_dict = safetensors.torch.load_file(projector_path)
        self.load_state_dict(projector_state_dict, strict=False)

    @staticmethod
    def init_tokenizer(config):
        tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model_name_or_path,
            model_max_length=config.model_max_length,
            padding_side="right",
        )
        if 'Qwen' in config.llm_model_name_or_path:
            tokenizer.bos_token = tokenizer.eos_token
        elif 'llama' in config.llm_model_name_or_path:
            tokenizer.pad_token = '<|finetune_right_pad_id|>'
        return tokenizer

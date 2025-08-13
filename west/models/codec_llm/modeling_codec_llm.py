# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

from dataclasses import dataclass, field
from typing import Optional

import s3tokenizer
import safetensors
import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel)

from west.models.model import Model, ModelArgs


@ModelArgs.register
@dataclass
class CodecArgs:
    s3tokenizer_model_name_or_path: Optional[str] = "speech_tokenizer_v1_25hz"
    llm_model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-7B")
    num_speech_tokens: int = 4096
    codec_llm_model_path: Optional[str] = field(default='')


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class CodecLLM(PreTrainedModel, Model):
    model_type = 'codec_llm'
    supports_gradient_checkpointing = True

    def __init__(self, config: CodecArgs):
        # Load llm model and tokenizer
        llm_config = AutoConfig.from_pretrained(config.llm_model_name_or_path)
        llm_config.use_cache = False
        # TODO(Binbin Zhang): now we just reuse LLM config, will try to figure
        # out the impact on training and generation.
        super().__init__(llm_config)
        speech_tokenizer = s3tokenizer.load_model(
            'speech_tokenizer_v1_25hz', config.s3tokenizer_model_name_or_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.speech_tokenizer = speech_tokenizer.to(device)
        # TODO(Binbin Zhang): rethink the pretrain and training model init
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name_or_path,
            config=llm_config,
            torch_dtype='auto',
            attn_implementation="flash_attention_2",  # or "flex_attention"
        )
        if config.codec_llm_model_path:
            self.load_llm(config.codec_llm_model_path)
        self.speech_tokenizer.freeze()
        self._keys_to_ignore_on_save = set()
        for k in self.speech_tokenizer.state_dict().keys():
            self._keys_to_ignore_on_save.add('speech_tokenizer.' + k)
        # We assume the last 4096 units are speech tokens
        self.speech_code_start_idx = llm_config.vocab_size - 4096

    def reorg_ids(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
    ):
        """ Extract speech codes by speech tokenizer, and reorg that in
            `input_ids`, `labels`
        """
        speech_codes, speech_codes_lens = self.speech_tokenizer.quantize(
            audio_features.transpose(1, 2), audio_feature_lengths)
        for i in range(audio_features.size(0)):
            b = batch_idx[i]
            s, e = audio_offsets[i], audio_offsets[i] + speech_codes_lens[i]
            ids = speech_codes[
                i, :speech_codes_lens[i]] + self.speech_code_start_idx
            input_ids[b, s:e] = ids
            labels[b, s:e] = ids
        return input_ids, labels

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
        input_ids, labels = self.reorg_ids(input_ids, labels, audio_offsets,
                                           audio_features,
                                           audio_feature_lengths, batch_idx)
        out = self.llm(input_ids=input_ids,
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
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        eos_token_id=None,
        decode_config=None,
    ):
        assert input_ids.size(0) == 1
        input_ids, labels = self.reorg_ids(input_ids, labels, audio_offsets,
                                           audio_features,
                                           audio_feature_lengths, batch_idx)
        token_length = audio_offsets[0]
        min_length = token_length * 2
        max_length = token_length * 20
        # There is no prompt token output if we use `inputs_embeds`
        # instead of `input_ids`
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=True,
            top_p=0.8,
            top_k=10,
            repetition_penalty=1.4,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
        )
        return model_outputs

    def load_llm(self, llm_path):
        print(f'Loading {llm_path}')
        llm_state_dict = safetensors.torch.load_file(llm_path)
        self.load_state_dict(llm_state_dict, strict=False)

    @staticmethod
    def init_tokenizer(config):
        tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name_or_path)
        if 'Qwen' in config.llm_model_name_or_path:
            tokenizer.bos_token = tokenizer.eos_token
        return tokenizer

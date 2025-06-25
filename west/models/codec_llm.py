# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import logging
from dataclasses import dataclass, field
from typing import Optional

import s3tokenizer
import safetensors
import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel)

from .model import Model, ModelArgs


@ModelArgs.register
@dataclass
class CodecArguments:
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

    def __init__(self, speech_tokenizer, llm, llm_config):
        super().__init__(llm_config)
        self.speech_tokenizer = speech_tokenizer
        self.llm = llm
        self._keys_to_ignore_on_save = set()
        for k in self.speech_tokenizer.state_dict().keys():
            self._keys_to_ignore_on_save.add('speech_tokenizer.' + k)
        # We assume the last 4096 units are speech tokens
        self.speech_code_start_idx = llm_config.vocab_size - 4096
        self.num_sentences = 0

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

    def freeze_speech_tokenizer(self):
        freeze_model(self.speech_tokenizer)

    def load_llm(self, llm_path):
        print(f'Loading {llm_path}')
        llm_state_dict = safetensors.torch.load_file(llm_path)
        self.load_state_dict(llm_state_dict, strict=False)

    @staticmethod
    def init_model(model_args):
        speech_tokenizer = s3tokenizer.load_model(
            'speech_tokenizer_v1_25hz',
            model_args.s3tokenizer_model_name_or_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        speech_tokenizer = speech_tokenizer.to(device)
        # Load llm model and tokenizer
        config = AutoConfig.from_pretrained(model_args.llm_model_name_or_path)
        config.use_cache = False
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_args.llm_model_name_or_path,
            config=config,
            torch_dtype='auto',
            attn_implementation="flash_attention_2",  # or "flex_attention"
        )
        model = CodecLLM(speech_tokenizer, llm_model, config)
        model.freeze_speech_tokenizer()
        if model_args.codec_llm_model_path:
            model.load_llm(model_args.codec_llm_model_path)
        return model

    @staticmethod
    def init_tokenizer(model_args):
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.llm_model_name_or_path)
        if 'Qwen' in model_args.llm_model_name_or_path:
            tokenizer.bos_token = tokenizer.eos_token
        return tokenizer

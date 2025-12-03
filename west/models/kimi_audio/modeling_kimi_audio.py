# Copyright (c) 2025 Chengdong Liang(liangchengdongd@qq.com)

import math
import os
from typing import Optional

import torch
from transformers import AutoConfig, GenerationMixin, PreTrainedModel
from transformers.models.whisper.modeling_whisper import WhisperModel

from .configuration_kimi_audio import KimiAudioConfig, WhisperVQConfig
from .modeling_moonshot_kimia import MoonshotKimiaForCausalLM, WhisperVQEncoder
from .tokenization_kimi_audio import KimiAudioTokenizer


class KimiAudio(PreTrainedModel, GenerationMixin):
    model_type = "kimi_audio"
    config_class = KimiAudioConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: KimiAudioConfig):
        super().__init__(config)
        self.llm_config = AutoConfig.from_pretrained(
            config.llm_model_name_or_path, trust_remote_code=True)
        self.llm = MoonshotKimiaForCausalLM.from_pretrained(
            config.llm_model_name_or_path,
            torch_dtype='auto',
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self.speech_tokenizer = WhisperVQEncoder(
            WhisperVQConfig.from_pretrained(
                config.speech_tokenizer_name_or_path, trust_remote_code=True)
        ).eval()
        self.speech_tokenizer._freeze_parameters()
        self.speech_encoder = WhisperModel.from_pretrained(
            os.path.join(config.llm_model_name_or_path, 'whisper-large-v3'),
            torch_dtype='auto',
            trust_remote_code=True,
        ).encoder

    def get_speech_embeddings(self, audio_features, audio_features_mask):
        audio_features = audio_features.transpose(1, 2)
        speech_encoder_embs = self.speech_encoder(
            input_features=audio_features,
            attention_mask=audio_features_mask,
            return_dict=True)
        speech_encoder_embs = speech_encoder_embs.last_hidden_state
        speech_encoder_embs = speech_encoder_embs.reshape(
            speech_encoder_embs.shape[0],
            int(speech_encoder_embs.shape[1] // 4),
            speech_encoder_embs.shape[2] * 4,
        )
        speech_encoder_embs = self.llm.get_adaptor()(speech_encoder_embs)

        with torch.no_grad():
            speech_tokenizer_ids = self.speech_tokenizer(
                input_features=audio_features,
                attention_mask=audio_features_mask,
                return_dict=True,
            )
            speech_tokenizer_ids += self.llm_config.kimia_token_offset
        speech_tokenizer_embs = self.llm.get_input_embeddings()(
            speech_tokenizer_ids.clone())
        # merge continuous audio and discrete audio
        speech_embs = (
            speech_encoder_embs + speech_tokenizer_embs) * math.sqrt(2.0)
        speech_emb_lens = audio_features_mask[:, ::2][:, ::4].sum(1)
        return speech_embs, speech_emb_lens

    def compute_mix_embedding(
        self,
        input_ids: torch.LongTensor = None,
        audio_offsets: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_features_mask: Optional[torch.Tensor] = None,
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
    ):
        text_emb = self.llm.get_input_embeddings()(input_ids)
        speech_emb, speech_emb_lens = self.get_speech_embeddings(
            audio_features, audio_features_mask)
        inputs_embeds = text_emb
        # https://github.com/MoonshotAI/Kimi-Audio/blob/master/finetune_codes/modeling_kimia.py#L720-L721  # noqa
        text_input_ids = [151666] * input_ids.size(1)
        text_input_ids = torch.tensor(text_input_ids,
                                      dtype=torch.long).to(input_ids.device)
        text_input_embs = self.llm.get_input_embeddings()(text_input_ids)
        for i in range(audio_features.size(0)):
            b = batch_idx[i]
            s, e = audio_offsets[i], audio_offsets[i] + speech_emb_lens[i]
            inputs_embeds[b, s:e, :] = speech_emb[i, :speech_emb_lens[i], :]
            inputs_embeds[b, :] = inputs_embeds[b, :] + text_input_embs

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
        audio_features_mask: Optional[torch.Tensor] = None,
        audio_features_mask_lengths: Optional[torch.LongTensor] = None,  # dummy
        batch_idx: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.compute_mix_embedding(
            input_ids,
            audio_offsets,
            audio_features,
            audio_features_mask,
            audio_features_lengths,
            batch_idx,
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
        audio_features_mask: Optional[torch.Tensor] = None,
        audio_features_mask_lengths: Optional[torch.Tensor] = None,  # dummy
        audio_features_lengths: Optional[torch.LongTensor] = None,
        batch_idx: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs_embeds = self.compute_mix_embedding(
            input_ids,
            audio_offsets,
            audio_features,
            audio_features_mask,
            audio_features_lengths,
            batch_idx,
        )
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=200,
            **kwargs,
        )
        return model_outputs[1]

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()

    def init_tokenizer(self):
        tokenizer = KimiAudioTokenizer.from_pretrained(
            self.config.llm_model_name_or_path,
            trust_remote_code=True,
        )

        # for unified interface with transformers tokenizer
        def batch_decode(self, batch_ids, **kwargs):
            if hasattr(batch_ids, 'tolist'):
                batch_ids = batch_ids.tolist()
            # one by one decode
            decoded_texts = []
            for ids in batch_ids:
                text = self.detokenize(ids)
                decoded_texts.append(text)
            return decoded_texts
        tokenizer.batch_decode = batch_decode.__get__(tokenizer)
        return tokenizer

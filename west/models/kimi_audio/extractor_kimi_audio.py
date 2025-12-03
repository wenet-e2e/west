# Copyright (c) 2025 Chengdong Liang(liangchengdongd@qq.com)

import os

import torch
import torchaudio
from transformers import WhisperFeatureExtractor
from transformers.trainer_pt_utils import LabelSmoother

from west.dataset.extractor import Extractor


class ExtractorKimiAudio(Extractor):
    model_type = 'kimi_audio'
    fields_batch_static = {'audio_offsets'}
    fields_batch_dynamic = {'audio_features', 'input_ids', 'labels',
                            'audio_features_mask'}
    fields_pack_offset = {'audio_offsets'}

    def __init__(self, tokenizer, model_config, inference=False):
        super().__init__(tokenizer, model_config, inference)
        self.whisper_feat_extractor = WhisperFeatureExtractor.from_pretrained(
            os.path.join(model_config.llm_model_name_or_path,
                         'whisper-large-v3'))
        self.ds_rate = (self.model_config.encoder_ds_rate *
                        self.model_config.encoder_projector_ds_rate)

    def compute_feat(self, wav):
        audio, _ = torchaudio.load(wav)
        audio = audio.squeeze(0).numpy()
        whisper_feature = self.whisper_feat_extractor(
            audio, sampling_rate=16000, return_attention_mask=True,
            return_tensors="pt", padding="max_length")
        # [D, T] pad 30s
        feature = whisper_feature['input_features'].squeeze(0).transpose(0, 1)
        feature_attention_mask = whisper_feature['attention_mask'].squeeze(0)
        num_audio_tokens = feature_attention_mask[::2][::4].sum()
        return feature, feature_attention_mask, num_audio_tokens

    def extract(self, item):
        """
        1. speech pretraining data (asr):
        messages = [
            {'role': 'user', 'content': [{
                'type': 'text', 'text': '请将音频内容转换为文字。'}]},
            {'role': 'assistant', 'content': item['txt']},
        ]
        """
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        if 'messages' in item:
            if not isinstance(item['messages'], list) \
               or len(item['messages']) < 2 \
               or item['messages'][-2]['role'] != 'user' \
               or item['messages'][-1]['role'] != 'assistant':
                return None
            messages = item['messages']
        else:  # Speech pretraining data
            messages = [
                {
                    'role': 'user',
                    'content': [{
                        'type': 'text',
                        'text': '请将音频内容转换为文字。'
                    }, {
                        'type': 'audio',
                        'audio': item['wav']
                    }]
                },
                {
                    'role': 'assistant',
                    'content': item['txt']
                },
            ]
        t0 = ""
        t1 = "<|im_media_end|><|im_kimia_speech_ct_id|><|im_msg_end|><|im_kimia_assistant_msg_start|>"  # noqa
        t2 = ""
        # NOTE(cdliang): Currently only support ASR task
        assert len(messages) <= 2
        for msg in messages:
            if msg['role'] == 'user':
                assert isinstance(msg['content'], list)
                assert len(msg['content']) == 2
                t0 += "<|im_kimia_user_msg_start|>" \
                      + msg['content'][0]['text'] \
                      + '<|im_media_begin|>'
                audio = msg['content'][1]['audio']
                mel, mel_mask, num_audio_tokens = self.compute_feat(audio)
                ids_audio = [0] * num_audio_tokens
            elif msg['role'] == 'assistant':
                t2 = msg['content'] + '<|im_kimia_text_eos|><|im_msg_end|>'
        if not self.inference:
            if mel.size(0) > self.model_config.max_speech_frames or mel.size(
                    0) < self.model_config.min_speech_frames:
                return None
        ids0 = self.tokenizer.encode(t0, bos=False, eos=False,
                                     allowed_special="all")
        ids1 = self.tokenizer.encode(t1, bos=False, eos=False,
                                     allowed_special="all")
        ids = ids0 + ids_audio + ids1
        tgt = [IGNORE_TOKEN_ID] * len(ids)
        if not self.inference:
            ids2 = self.tokenizer.encode(t2, bos=False, eos=False,
                                         allowed_special="all")
            ids = ids + ids2
            tgt = tgt + ids2

        input_ids = torch.tensor(ids, dtype=torch.int)
        tgt_ids = torch.tensor(tgt, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': tgt_ids,
            'audio_features': mel,
            'audio_features_mask': mel_mask,
            'audio_offsets': len(ids0),
        }

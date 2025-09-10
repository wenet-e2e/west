# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import math

import torch
import wenet
from transformers.trainer_pt_utils import LabelSmoother

from west.dataset.extractor import Extractor


class ExtractorTouchASU(Extractor):
    model_type = 'touch_asu'
    fields_batch_static = {'audio_offsets', 'has_audio'}
    fields_batch_dynamic = {'audio_features', 'input_ids', 'labels'}
    fields_pack_offset = {'audio_offsets'}

    def __init__(self, tokenizer, model_config, inference=False):
        super().__init__(tokenizer, model_config, inference)
        self.compute_feature, self.feature_dim = wenet.load_feature(
            self.model_config.wenet_model_name_or_path)
        self.ds_rate = (self.model_config.encoder_ds_rate *
                        self.model_config.encoder_projector_ds_rate)

    def extract(self, item):
        """
        1. speech pretraining data (asr):
        messages = [
            {'role': 'user', 'content': [{
                'type': 'text', 'text': 'Transcribe the Speech'}]},
            {'role': 'assistant', 'content': item['txt']},
        ]
        2. QA: SFT data (multi-turn)
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},  # optional # noqa
            {'role': 'user', 'content': 'What is the capital of China?'},   # optional # noqa
            {'role': 'assistant', 'content': 'The capital of China is Beijing.'},   # optional # noqa
            {'role': 'user', 'content': {'type': 'audio', 'audio': item['wav']}},  # last turn (for audio qa) # noqa
            or
            {'role': 'user', 'content': 'question text'},  # last turn (for text qa) # noqa
            {'role': 'assistant', 'content': item['txt']},
        ]
        """
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        # OpenAI role-content based SFT data
        # At least one pair of "user" and "assistant"
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
                        'text': 'Transcribe the Speech'
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

        t0 = ''
        t1 = '<|im_end|>\n' + '<|im_start|>assistant\n'
        t2 = ''
        has_audio = True
        for msg in messages[:-2]:
            t0 += '<|im_start|>' + msg['role'] + '\n' \
                  + msg['content'] + '<|im_end|>\n'
        for msg in messages[-2:]:
            if msg['role'] == 'user':
                t0 += '<|im_start|>user\n'
                if isinstance(msg['content'], dict):
                    assert msg['content']['type'] == 'audio'
                    audio = msg['content']['audio']
                elif isinstance(msg['content'], list):
                    # Here we assume the 1st is text, 2nd is audio
                    assert len(msg['content']) == 2
                    t0 += msg['content'][0]['text']
                    audio = msg['content'][1]['audio']
                elif isinstance(msg['content'], str):  # No audio
                    t0 += msg['content']
                    has_audio = False
                if has_audio:
                    t0 += '<|audio_bos|>'
                    t1 = '<|audio_eos|>' + t1
                    mel = self.compute_feature(audio)
                    ids_audio = [0] * math.ceil(mel.size(0) / self.ds_rate)
                    tgt_audio = [IGNORE_TOKEN_ID] * len(ids_audio)
                else:
                    # fake 1s mel feature
                    mel = torch.zeros((100, self.feature_dim),
                                      dtype=torch.float)
                    ids_audio = []
                    tgt_audio = []

            elif msg['role'] == 'assistant':
                t2 = msg['content'] + '<|im_end|>\n'
        # Filter some data
        if not self.inference:
            if mel.size(0) > self.model_config.max_speech_frames or mel.size(
                    0) < self.model_config.min_speech_frames:
                return None
        # TODO(Binbin Zhang): Mutil-turn support
        ids0 = self.tokenizer.encode(t0)
        ids1 = self.tokenizer.encode(t1)
        ids = ids0 + ids_audio + ids1
        tgt = ids0 + tgt_audio + ids1
        if not self.inference:
            ids2 = self.tokenizer.encode(t2)
            ids = ids + ids2
            tgt = tgt + ids2
        input_ids = torch.tensor(ids, dtype=torch.int)
        tgt_ids = torch.tensor(tgt, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': tgt_ids,
            'audio_features': mel,
            'audio_offsets': len(ids0),
            'has_audio': has_audio,
        }

# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import copy
import math

import torch
import torchaudio
from transformers.trainer_pt_utils import LabelSmoother

from west.dataset.extractor import Extractor
from west.models.touch_asu import ExtractorTouchASU


class ExtractorTouchChat(Extractor):
    model_type = 'touch_chat'
    fields_batch_static = {'audio_offsets', 'talker_offsets', 'has_audio'}
    fields_batch_dynamic = {
        'audio_features', 'input_ids', 'labels', 'talker_features'
    }
    fields_pack_offset = {'audio_offsets', 'talker_offsets'}

    def __init__(self, tokenizer, inference=False):
        super().__init__(tokenizer, inference)
        self.asu_extractor = ExtractorTouchASU(tokenizer, inference)
        if self.inference:
            self.fields_batch_static.remove('talker_offsets')
            self.fields_batch_dynamic.remove('talker_features')

    def extract(self, item):
        import s3tokenizer
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        assert 'messages' in item  # only support role-content based SFT data
        # Data format in `messages` for TouchChat
        # {'role': 'user', 'content': {'type': 'audio', 'audio', x, 'text':x}}
        # {'role': 'assistant',
        #  'content': {'type': 'audio', 'audio', x, 'text':x}}
        # Change assistant audio output to text output for Thinker(ASU)
        if self.inference:
            ret = self.asu_extractor.extract(item)
            return ret
        else:
            asu_item = copy.deepcopy(item)
            assert len(asu_item['messages']) == 2
            assert asu_item['messages'][1]['role'] == 'assistant'
            assistant_text = asu_item['messages'][1]['content']['text']
            asu_item['messages'][1]['content'] = assistant_text
            ret = self.asu_extractor.extract(asu_item)
            assert item['messages'][1]['content']['type'] == 'audio'
            audio_file = item['messages'][1]['content']['audio']
            wav, sample_rate = torchaudio.load(audio_file)
            wav = torchaudio.transforms.Resample(sample_rate, 16000)(wav)
            wav = wav[0]  # get the first channel
            mel = s3tokenizer.log_mel_spectrogram(wav)
            mel = mel.transpose(0, 1)
            # There is 100 frames mel per second, and 25 tokens per second
            num_audio_token = math.ceil(mel.size(0) / 100.0 * 25)
            ids = [0] * num_audio_token + [self.tokenizer.eos_token_id]
            ids_audio = torch.tensor(ids, dtype=torch.long)
            # We first ignore it in thinker, then override `tgt_audio` in talker
            tgt = [IGNORE_TOKEN_ID] * num_audio_token + [
                self.tokenizer.eos_token_id
            ]
            tgt_audio = torch.tensor(tgt, dtype=torch.long)
            # Merge Thinker/Talker tokens in TouchChat
            thinker_input_ids = ret['input_ids']
            thinker_labels = ret['labels']
            ret['talker_features'] = mel
            ret['talker_offsets'] = len(thinker_input_ids)
            ret['input_ids'] = torch.cat((thinker_input_ids, ids_audio), dim=0)
            ret['labels'] = torch.cat((thinker_labels, tgt_audio), dim=0)
            return ret

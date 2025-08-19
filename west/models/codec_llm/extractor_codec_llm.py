# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import math

import torch
import torchaudio
from transformers.trainer_pt_utils import LabelSmoother

from west.dataset.extractor import Extractor


class ExtractorTtsCodec(Extractor):
    model_type = 'codec_llm'
    fields_batch_static = {'audio_offsets', 'text_lengths'}
    fields_batch_dynamic = {'audio_features', 'input_ids', 'labels'}
    fields_pack_offset = {'audio_offsets'}

    def extract(self, item):
        import s3tokenizer
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        audio = torchaudio.transforms.Resample(item['sample_rate'],
                                               16000)(item['wav'])
        audio = audio[0]  # get the first channel
        mel = s3tokenizer.log_mel_spectrogram(audio)
        mel = mel.transpose(0, 1)
        # There is 100 frames mel per second, and 25 tokens per second
        num_audio_token = math.ceil(mel.size(0) / 100.0 * 25)
        if not self.inference:
            content = item['txt'] + '<|audio_bos|>'
        else:
            content = item['txt'] + item['syn_txt'] + '<|audio_bos|>'
        ids_text = [self.tokenizer.bos_token_id
                    ] + self.tokenizer.encode(content)
        tgt_text = [IGNORE_TOKEN_ID] * len(ids_text)
        ids_audio = [0] * num_audio_token
        if not self.inference:
            ids = ids_text + ids_audio + [self.tokenizer.eos_token_id]
            tgt = tgt_text + ids_audio + [self.tokenizer.eos_token_id]
        else:
            ids = ids_text + ids_audio
            tgt = tgt_text + ids_audio
        input_ids = torch.tensor(ids, dtype=torch.long)
        tgt_ids = torch.tensor(tgt, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': tgt_ids,
            'audio_features': mel,
            'audio_offsets': len(ids_text),
            'text_lengths': len(item.get('syn_txt', ''))
        }

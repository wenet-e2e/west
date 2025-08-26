# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import torch
import torchaudio
from transformers.trainer_pt_utils import LabelSmoother

from west.dataset.extractor import Extractor


class ExtractorTouchASU(Extractor):
    model_type = 'touch_asu'
    fields_batch_static = {'audio_offsets'}
    fields_batch_dynamic = {'audio_features', 'input_ids', 'labels'}
    fields_pack_offset = {'audio_offsets'}

    def extract(self, item):
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        if 'messages' in item:  # OpenAI role-content based SFT data
            messages = item['messages']
        else:  # Speech pretraining data
            messages = [
                {
                    'role':
                    'user',
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

        t0 = '<|im_start|>user\n'
        t1 = '<|audio_eos|><|im_end|>\n' + '<|im_start|>assistant\n'
        t2 = ''
        for msg in messages:
            if msg['role'] == 'system':
                t0 += msg['content']
            elif msg['role'] == 'user':
                if isinstance(msg['content'], dict):
                    assert msg['content']['type'] == 'audio'
                    t0 += '<|audio_bos|>'
                    audio = msg['content']['audio']
                elif isinstance(msg['content'], list):
                    # Here we assume the 1st is text, 2nd is audio
                    assert len(msg['content']) == 2
                    t0 += msg['content'][0]['text']
                    t0 += '<|audio_bos|>'
                    audio = msg['content'][1]['audio']
                # Feature extraction
                if isinstance(audio, str):  # path
                    wav, sample_rate = torchaudio.load(audio)
                else:
                    wav, sample_rate = item['wav'], item['sample_rate']
                wav = torchaudio.transforms.Resample(sample_rate, 16000)(wav)
                wav = wav * (1 << 15)
                mel = torchaudio.compliance.kaldi.fbank(wav,
                                                        num_mel_bins=80,
                                                        frame_length=25,
                                                        frame_shift=10,
                                                        dither=0.0,
                                                        energy_floor=0.0,
                                                        sample_frequency=16000)
                # Here 8 is the final subsampling rate
                ids_audio = [0] * (mel.size(0) // 8)
                tgt_audio = [IGNORE_TOKEN_ID] * len(ids_audio)

            elif msg['role'] == 'assistant':
                t2 = msg['content'] + '<|im_end|>\n'
        # TODO(Binbin Zhang): Mutil-turn support
        ids0 = self.tokenizer.encode(t0)
        ids1 = self.tokenizer.encode(t1)
        ids = [self.tokenizer.bos_token_id] + ids0 + ids_audio + ids1
        tgt = [self.tokenizer.bos_token_id] + ids0 + tgt_audio + ids1
        if not self.inference:
            ids2 = self.tokenizer.encode(t2)
            ids = ids + ids2 + [self.tokenizer.eos_token_id]
            tgt = tgt + ids2 + [self.tokenizer.eos_token_id]
        input_ids = torch.tensor(ids, dtype=torch.int)
        tgt_ids = torch.tensor(tgt, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': tgt_ids,
            'audio_features': mel,
            'audio_offsets': len(ids0) + 1,
        }

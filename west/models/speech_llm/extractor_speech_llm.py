# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import torch
import torchaudio
from transformers.trainer_pt_utils import LabelSmoother

from west.dataset.extractor import Extractor


class ExtractorAsrWenet(Extractor):
    model_type = 'speech_llm'
    fields_batch_static = {'audio_offsets'}
    fields_batch_dynamic = {'audio_features', 'input_ids', 'labels'}
    fields_pack_offset = {'audio_offsets'}

    def extract(self, item):
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        audio = torchaudio.transforms.Resample(item['sample_rate'],
                                               16000)(item['wav'])
        audio = audio * (1 << 15)
        # mel: (T, 80)
        mel = torchaudio.compliance.kaldi.fbank(audio,
                                                num_mel_bins=80,
                                                frame_length=25,
                                                frame_shift=10,
                                                dither=0.0,
                                                energy_floor=0.0,
                                                sample_frequency=16000)
        # TODO(Binbin Zhang): Refine to instruction + <AUDIO>
        ids_audio = [0] * (mel.size(0) // 8)  # 8 is the final subsampling rate
        tgt_audio = [IGNORE_TOKEN_ID] * len(ids_audio)
        instruction = 'Transcribe the speech'
        content = item['txt']
        t0 = '<|im_start|>system\n' + \
             'You are a helpful assistant<|im_end|>\n' + \
             '<|im_start|>user\n' + instruction + '<|audio_bos|>'
        t1 = '<|audio_eos|><|im_end|>\n' + '<|im_start|>assistant\n'
        ids0 = self.tokenizer.encode(t0)
        ids1 = self.tokenizer.encode(t1)
        ids = [self.tokenizer.bos_token_id] + ids0 + ids_audio + ids1
        tgt = [self.tokenizer.bos_token_id] + ids0 + tgt_audio + ids1
        if not self.inference:
            t2 = content + '<|im_end|>\n'
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

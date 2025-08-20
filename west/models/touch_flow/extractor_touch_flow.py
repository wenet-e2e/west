# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import torch
import torchaudio
from torchaudio.compliance import kaldi

from west.dataset.extractor import Extractor
from west.utils.audio import mel_spectrogram


class ExtractorTouchFlow(Extractor):
    model_type = 'touch_flow'

    fields_batch_dynamic = {'mel_speaker', 'mel_token', 'mel_vocoder'}

    def __init__(self, tokenizer, inference=False):
        super().__init__(tokenizer, inference)
        if self.inference:
            self.fields_batch_dynamic.add('llm_token')

    def extract(self, item):
        import s3tokenizer
        audio = torchaudio.transforms.Resample(item['sample_rate'],
                                               16000)(item['wav'])
        audio_22k = torchaudio.transforms.Resample(item['sample_rate'],
                                                   22050)(item['wav'])
        mel_vocoder = mel_spectrogram(audio_22k,
                                      n_fft=1024,
                                      num_mels=80,
                                      sampling_rate=22050,
                                      hop_size=256,
                                      win_size=1024,
                                      fmin=0,
                                      fmax=8000,
                                      center=False)
        mel_vocoder = mel_vocoder[0].transpose(0, 1)
        # for campplus-200k model, use povey window
        mel_speaker = kaldi.fbank(audio,
                                  num_mel_bins=80,
                                  frame_length=25,
                                  frame_shift=10,
                                  dither=0.0,
                                  sample_frequency=16000,)
        mel_speaker = mel_speaker - torch.mean(mel_speaker, 0)
        mel_token = s3tokenizer.log_mel_spectrogram(audio[0])
        mel_token = mel_token.transpose(0, 1)
        ret = {
            'mel_speaker': mel_speaker,
            'mel_token': mel_token,
            'mel_vocoder': mel_vocoder,
        }
        if self.inference:
            ids = [int(x) for x in item['llm_token'].split()]
            ret['llm_token'] = torch.tensor(ids, dtype=torch.int)
        return ret

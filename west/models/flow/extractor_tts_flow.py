# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torchaudio.compliance import kaldi
from west.utils.audio import mel_spectrogram

from west.dataset.extractor import Extractor


class ExtractorTtsFlow(Extractor):
    extractor_type = 'tts_flow'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inference = self.kwargs.get('inference', False)

    def extract(self, item):
        import s3tokenizer
        audio = torchaudio.transforms.Resample(item['sample_rate'],
                                               16000)(item['wav'])
        audio_22k = torchaudio.transforms.Resample(item['sample_rate'],
                                                   22050)(item['wav'])
        mel_vocoder = mel_spectrogram(audio_22k, 1024, 80, 22050, 256, 1024, 0,
                                      22050 / 2)
        mel_vocoder = mel_vocoder[0].transpose(0, 1)
        mel_speaker = kaldi.fbank(audio,
                                  num_mel_bins=80,
                                  frame_length=25,
                                  frame_shift=10,
                                  sample_frequency=16000,
                                  window_type='hamming')
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

    def batch(self, seqs):
        ret = {}
        for name in ['mel_speaker', 'mel_token', 'mel_vocoder']:
            mel_features = [s[name] for s in seqs]
            mel_lengths = torch.tensor([t.size(0) for t in mel_features],
                                       dtype=torch.int)
            mel_features = pad_sequence(mel_features, batch_first=True)
            ret[name] = mel_features
            ret[name + '_lengths'] = mel_lengths
        if self.inference:
            ret['llm_token'] = pad_sequence([s['llm_token'] for s in seqs],
                                            batch_first=True)
        return ret

# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)
import math
from abc import ABC, abstractmethod

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torchaudio.compliance import kaldi
from transformers.trainer_pt_utils import LabelSmoother

from west.utils.audio import mel_spectrogram


class Extractor(ABC):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def extract(self, item):
        pass


class ExtractorTtsCodec(Extractor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract(self, item):
        import s3tokenizer
        tokenizer = self.kwargs.get('tokenizer')
        inference = self.kwargs.get('inference', False)
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        audio = torchaudio.transforms.Resample(item['sample_rate'],
                                               16000)(item['wav'])
        audio = audio[0]  # get the first channel
        mel = s3tokenizer.log_mel_spectrogram(audio)
        mel = mel.transpose(0, 1)
        # There is 100 frames mel per second, and 25 tokens per second
        num_audio_token = math.ceil(mel.size(0) / 100.0 * 25)
        if not inference:
            content = item['txt'] + '<|audio_bos|>'
        else:
            content = item['txt'] + item['syn_txt'] + '<|audio_bos|>'
        ids_text = [tokenizer.bos_token_id] + tokenizer.encode(content)
        tgt_text = [IGNORE_TOKEN_ID] * len(ids_text)
        ids_audio = [0] * num_audio_token
        if not inference:
            ids = ids_text + ids_audio + [tokenizer.eos_token_id]
            tgt = tgt_text + ids_audio + [tokenizer.eos_token_id]
        else:
            ids = ids_text + ids_audio
            tgt = tgt_text + ids_audio
        input_ids = torch.tensor(ids, dtype=torch.long)
        tgt_ids = torch.tensor(tgt, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': tgt_ids,
            'mel': mel,
            'offset': len(ids_text),
        }


class ExtractorTtsFlow(Extractor):

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


class ExtractorAsrWenet(Extractor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract(self, item):
        tokenizer = self.kwargs.get('tokenizer')
        inference = self.kwargs.get('inference', False)
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
        ids0 = tokenizer.encode(t0)
        ids1 = tokenizer.encode(t1)
        ids = [tokenizer.bos_token_id] + ids0 + ids_audio + ids1
        tgt = [tokenizer.bos_token_id] + ids0 + tgt_audio + ids1
        if not inference:
            t2 = content + '<|im_end|>\n'
            ids2 = tokenizer.encode(t2)
            ids = ids + ids2 + [tokenizer.eos_token_id]
            tgt = tgt + ids2 + [tokenizer.eos_token_id]
        input_ids = torch.tensor(ids, dtype=torch.int)
        tgt_ids = torch.tensor(tgt, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'labels': tgt_ids,
            'mel': mel,
            'offset': len(ids0) + 1,
        }


class ExtractorFactory:

    @staticmethod
    def create(extractor_type: str) -> Extractor:
        classes = {
            'asr_wenet': ExtractorAsrWenet,
            'tts_codec': ExtractorTtsCodec,
            'tts_flow': ExtractorTtsFlow,
        }
        if extractor_type.lower() in classes:
            return classes[extractor_type.lower()]
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")

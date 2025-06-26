# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)
import math
from abc import ABC, abstractmethod
from typing import BinaryIO, Union

import torch
import torchaudio
from transformers.trainer_pt_utils import LabelSmoother


def load_audio(file: Union[str, BinaryIO], sr: int = 16000):
    audio, sample_rate = torchaudio.load(file)
    if sample_rate != sr:
        audio = torchaudio.transforms.Resample(sample_rate, sr)(audio)
    return audio


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
        audio = load_audio(item['wav'])
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


class ExtractorAsrWenet(Extractor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract(self, item):
        tokenizer = self.kwargs.get('tokenizer')
        inference = self.kwargs.get('inference', False)
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        audio = load_audio(item['wav'])
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
        }
        if extractor_type.lower() in classes:
            return classes[extractor_type.lower()]
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")

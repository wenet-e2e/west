# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import json
import logging
import os

import torch
import torchaudio
from torchaudio.compliance import kaldi

from west.dataset.extractor import Extractor
from west.utils.audio import mel_spectrogram


def _mel_speaker_fbank(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """16k mono fbank for WeSpeaker, same as training pipeline."""
    audio = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    mel_speaker = kaldi.fbank(audio,
                              num_mel_bins=80,
                              frame_length=25,
                              frame_shift=10,
                              dither=0.0,
                              sample_frequency=16000,)
    return mel_speaker - torch.mean(mel_speaker, 0)


class ExtractorTouchFlow(Extractor):
    model_type = 'touch_flow'

    fields_batch_dynamic = {'mel_speaker', 'mel_token', 'mel_vocoder'}

    def __init__(self,
                 tokenizer,
                 model_config,
                 inference=False,
                 spk_prompt_wav_map_path=None):
        super().__init__(tokenizer,
                         model_config,
                         inference,
                         spk_prompt_wav_map_path=spk_prompt_wav_map_path)
        if self.inference:
            self.fields_batch_dynamic.add('llm_token')
        self.spk_prompt_wav_map = {}
        path = (self.spk_prompt_wav_map_path or '').strip()
        if path and os.path.isfile(path):
            with open(path, 'r', encoding='utf8') as f:
                self.spk_prompt_wav_map = json.load(f)
            logging.info('ExtractorTouchFlow: loaded spk_prompt_wav_map from %s (%d entries)',
                         path, len(self.spk_prompt_wav_map))
        elif path:
            logging.warning('ExtractorTouchFlow: spk_prompt_wav_map_path not found: %s', path)

    def _mel_speaker_from_wav_path(self, wav_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.size(0) > 1:
            waveform = waveform[:1]
        return _mel_speaker_fbank(waveform, sample_rate)

    def extract(self, item):
        import s3tokenizer
        waveform, sample_rate = torchaudio.load(item['wav'])
        duration = waveform.size(1) / sample_rate
        if not self.inference and (
                duration < self.model_config.min_speech_duration
                or duration > self.model_config.max_speech_duration):
            return None

        audio = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        audio_22k = torchaudio.transforms.Resample(sample_rate, 22050)(waveform)
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
        spk_id = (item.get('spk') or '').strip()
        use_prompt_spk = bool(spk_id and spk_id in self.spk_prompt_wav_map)
        if use_prompt_spk:
            # spk: str speaker id (e.g. "biaobei");
            # lookup spk_prompt_wav_map[spk] -> prompt wav for mel_speaker
            prompt_path = self.spk_prompt_wav_map[spk_id]
            if not os.path.isfile(prompt_path):
                logging.warning(
                    'ExtractorTouchFlow: spk %s map path missing %s, fallback to item wav',
                    spk_id, prompt_path)
                mel_speaker = _mel_speaker_fbank(waveform, sample_rate)
            else:
                mel_speaker = self._mel_speaker_from_wav_path(prompt_path)
                if self.inference:
                    logging.info('ExtractorTouchFlow: using spk %s prompt wav: %s', spk_id, prompt_path)
        else:
            if spk_id and self.spk_prompt_wav_map:
                logging.warning(
                    'ExtractorTouchFlow: spk %r not in spk_prompt_wav_map, '
                    'mel_speaker from item wav', spk_id)
            mel_speaker = _mel_speaker_fbank(waveform, sample_rate)

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

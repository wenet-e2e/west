# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)

import math

import torch
import torchaudio
from transformers.trainer_pt_utils import LabelSmoother

from west.dataset.extractor import Extractor


class ExtractorTouchTTS(Extractor):
    model_type = "touch_tts"
    fields_batch_static = {"audio_offsets", "text_lengths"}
    fields_batch_dynamic = {"audio_features", "input_ids", "labels"}
    fields_pack_offset = {"audio_offsets"}

    def extract(self, item):
        import s3tokenizer

        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        spk = item.get("spk", "")
        ins = item.get("ins", "")
        sft_mode = bool(spk or ins)  # is sft mode

        # Training: always require txt and wav (both pretrain and SFT)
        # Inference: only SFT may optionally omit txt/wav (no prompt)
        if not self.inference:
            if "txt" not in item or "wav" not in item:
                return None
        else:
            # Inference SFT without prompt: no txt or wav
            if sft_mode and ("txt" not in item or "wav" not in item):
                include_prompt = False
            else:
                include_prompt = True
                if "txt" not in item or "wav" not in item:
                    return None

        if not self.inference or include_prompt:
            waveform, sample_rate = torchaudio.load(item["wav"])
            duration = waveform.size(1) / sample_rate
            if not self.inference and (
                duration < self.model_config.min_speech_duration
                or duration > self.model_config.max_speech_duration
            ):
                return None
            audio = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
            audio = audio[0]
            mel = s3tokenizer.log_mel_spectrogram(audio)
            mel = mel.transpose(0, 1)
            num_audio_token = math.ceil(mel.size(0) * 25 / 100.0 - 1e-9)
        else:
            mel = None
            num_audio_token = 0

        if not self.inference:
            if sft_mode:
                content = (
                    spk
                    + "<|spk_eos|>"
                    + ins
                    + "<|ins_eos|>"
                    + item["txt"]
                    + "<|audio_bos|>"
                )
            else:
                content = item["txt"] + "<|audio_bos|>"
            token_lengths = 0
        else:
            if sft_mode:
                if include_prompt:
                    content = (
                        spk
                        + "<|spk_eos|>"
                        + ins
                        + "<|ins_eos|>"
                        + item["txt"]
                        + item["syn_txt"]
                        + "<|audio_bos|>"
                    )
                else:
                    content = (
                        spk
                        + "<|spk_eos|>"
                        + ins
                        + "<|ins_eos|>"
                        + item["syn_txt"]
                        + "<|audio_bos|>"
                    )
            else:
                content = item["txt"] + item["syn_txt"] + "<|audio_bos|>"
            token_lengths = len(self.tokenizer.encode(item["syn_txt"]))

        if self.inference:
            print("content:", content)
        ids_text = [self.tokenizer.bos_token_id] + self.tokenizer.encode(content)
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
        result = {
            "input_ids": input_ids,
            "labels": tgt_ids,
            "audio_offsets": len(ids_text),
            "text_lengths": token_lengths,
        }
        if mel is not None:
            result["audio_features"] = mel
        return result

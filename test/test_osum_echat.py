# Copyright (c) 2025 Xuelong Geng(xlgeng@mail.nwpu.edu.cn)

import librosa
import torch
import torchaudio
from transformers import AutoConfig, AutoModel

from west.models.osum_echat.patch4generate import do_patch

do_patch()


def get_feat_from_wav_path(input_wav_path,
                           device: torch.device = torch.device('cuda')):
    """..."""
    waveform, sample_rate = torchaudio.load(input_wav_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                               new_freq=16000)
    waveform = resampler(waveform)
    waveform = waveform.squeeze(0)
    sample_rate = 16000
    window = torch.hann_window(400)
    stft = torch.stft(waveform, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs()**2
    filters = torch.from_numpy(
        librosa.filters.mel(sr=sample_rate, n_fft=400, n_mels=80))
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    feat = log_spec.transpose(0, 1)
    feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64).to(device)
    feat = feat.unsqueeze(0).to(device)
    feat = feat.to(torch.bfloat16)
    return feat, feat_lens


if __name__ == '__main__':
    from huggingface_hub import hf_hub_download

    # For natural language think model in west
    ckpt_path = hf_hub_download(repo_id="ASLP-lab/OSUM-EChat",
                                filename="language_think_west.pt")
    osum_config_path = "../examples/aishell/asr/conf/osum_echat.json"
    config_new = AutoConfig.from_pretrained(osum_config_path)
    osum_model = AutoModel.from_config(config_new)
    osum_model.eval()
    osum_model.to('cuda')
    missing_keys, unexpected_keys = osum_model.load_state_dict(torch.load(
        ckpt_path, map_location="cpu"),
                                                               strict=False)
    for key in missing_keys:
        print("missing tensor: {}".format(key))
    for key in unexpected_keys:
        print("unexpected tensor: {}".format(key))
    print(osum_model)
    test_wav_path = "./data/test_wave4osumechat.wav"
    fake_wav, faek_wav_lens = get_feat_from_wav_path(test_wav_path)
    osum_output = osum_model.generate(audio_features=fake_wav,
                                      audio_features_lengths=faek_wav_lens)
    print(osum_output)

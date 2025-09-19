#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025 Hao Yin(1049755192@qq.com)

# mel vocoder using bigvgan.
# ref: https://huggingface.co/nvidia/bigvgan_base_22khz_80band

import argparse
import os

import bigvgan
import numpy as np
import torch
from scipy.io import wavfile


def load_model(model_path, device="cuda"):
    """
    Load BigVGAN model from pretrained checkpoint.

    Args:
        model_path (str): Path to the pretrained model
        device (str): Device to load the model on

    Returns:
        torch.nn.Module: Loaded and prepared model
    """
    print(f"Loading BigVGAN model from: {model_path}")
    model = bigvgan.BigVGAN.from_pretrained(model_path, use_cuda_kernel=False)
    model.remove_weight_norm()
    model = model.eval().to(device)
    print("BigVGAN model loaded successfully")
    return model


def load_mel(mel_path, device="cuda"):
    """
    Load mel spectrogram from .npy file.

    Args:
        mel_path (str): Path to the .npy mel file
        device (str): Device to load the mel on

    Returns:
        torch.Tensor: Loaded mel spectrogram tensor
    """
    mel = np.load(mel_path)  # [80, T_frame]
    mel = torch.from_numpy(mel).unsqueeze(0).to(device)  # [1, 80, T_frame]
    return mel


def inference(model, mel, device="cuda"):
    """
    Perform inference using the model on mel spectrogram.

    Args:
        model (torch.nn.Module): Loaded model
        mel (torch.Tensor): Mel spectrogram tensor
        device (str): Device for inference

    Returns:
        torch.Tensor: Generated audio tensor
    """
    # mel is FloatTensor with shape [1, 80, T_frame]
    with torch.inference_mode():
        audio = model(mel)
    audio = audio.squeeze().cpu()
    return audio


def save_audio(audio, output_path, sample_rate=22050):
    """
    Save audio tensor to wav file.

    Args:
        audio (torch.Tensor): Audio tensor
        output_path (str): Output file path
        sample_rate (int): Sample rate for the audio file
    """
    audio_np = (audio * 32767.0).numpy().astype("int16")
    wavfile.write(output_path, sample_rate, audio_np)
    print(f"Audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mel_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/path/to/bigvgan_base_22khz_80band",
        help="Path to the pretrained BigVGAN model",
    )
    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path, args.device)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Process each mel file
    for mel_file in os.listdir(args.mel_path):
        mel_path = os.path.join(args.mel_path, mel_file)
        try:
            # Load mel spectrogram
            mel = load_mel(mel_path, args.device)
            # Perform inference
            audio = inference(model, mel, args.device)
            # Save audio
            output_filename = os.path.splitext(mel_file)[0] + ".wav"
            output_path = os.path.join(args.output_path, output_filename)
            save_audio(audio, output_path, args.sample_rate)

        except Exception as e:
            print(f"Error processing {mel_file}: {str(e)}")
            continue


if __name__ == "__main__":
    main()

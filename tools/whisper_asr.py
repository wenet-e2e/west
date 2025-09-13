#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025 Hao Yin(1049755192@qq.com)

# whisper asr for compute wer.

import sys
import json
import os
import re
from tqdm import tqdm
from tn.english.normalizer import Normalizer as EnNormalizer

import whisper

# TN
en_tn_model = EnNormalizer(overwrite_cache=False)
# ASR model
model = whisper.load_model(
    "/jfs-hdfs/user/xingchen.song/share/whisper/large-v3-turbo.pt"
)  # noqa


# normalize the text & keep english characters only.
def normalize_text(text):
    text = en_tn_model.normalize(text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# transcribe the audio using whisper.
def transcribe(audio_path):
    result = model.transcribe(audio_path, language="en")
    text = result["text"]
    return text


def main():
    wav_scp = sys.argv[1]
    syn_text = sys.argv[2]

    with open(wav_scp, "r") as f:
        with open(syn_text, "w") as out_f:
            for line in tqdm(f, desc="Whisper ASR transcribing"):
                key, wav_path = line.strip().split("\t")
                text = transcribe(wav_path).strip()
                text = normalize_text(text)

                # save as JSONL format
                json_obj = {"key": key, "txt": text}
                out_f.write(json.dumps(json_obj) + "\n")


if __name__ == "__main__":
    main()

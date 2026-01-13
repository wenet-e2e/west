#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025 Hao Yin(1049755192@qq.com)

# prepare the wav.scp & gt.jsonl for compute wer.

import json
import os
import re
import sys

from tn.english.normalizer import Normalizer as EnNormalizer

en_tn_model = EnNormalizer(overwrite_cache=False)


# normalize the text & keep english characters only.
def normalize_text(text):
    text = en_tn_model.normalize(text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


test_jsonl = sys.argv[1]
wav_dir = sys.argv[2]
wav_scp = sys.argv[3]
gt_jsonl = sys.argv[4]

# load the test jsonl file
wav_scp_data = []
gt_data = []
lines = open(test_jsonl).readlines()
for i, line in enumerate(lines):
    item = json.loads(line)
    key = item["key"]
    syn_txt = item["syn_txt"].strip()
    syn_txt = normalize_text(syn_txt)

    # For wav.scp
    wav_path = os.path.join(wav_dir, str(i) + ".wav")
    wav_path = os.path.abspath(wav_path)
    assert os.path.exists(wav_path), f"wav_path {wav_path} not exists"
    wav_scp_data.append(f"{key}\t{wav_path}")

    # For gt.jsonl
    gt_item = {"wav": key, "txt": syn_txt}
    gt_data.append(json.dumps(gt_item, ensure_ascii=False))

# save the wav.scp file
with open(wav_scp, "w") as f:
    f.write("\n".join(wav_scp_data))

# save the gt.jsonl file
with open(gt_jsonl, "w", encoding="utf-8") as f:
    f.write("\n".join(gt_data))

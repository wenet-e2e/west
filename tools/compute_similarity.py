# Copyright 2025 Hao Yin(1049755192@qq.com)

import json
import math
import sys

import wespeaker

# https://wenet.org.cn/downloads?models=wespeaker&version=campplus_cn_common_200k.tar.gz # noqa
model = wespeaker.load_model_pt("campplus")

prompts = {}
with open(sys.argv[1]) as f:
    for line in f:
        item = json.loads(line)
        prompts[item["key"]] = item["wav"]


total = 0.0
count = 0
with open(sys.argv[2]) as f, open(sys.argv[3], "w") as fout:
    for line in f:
        key, wav_gen = line.strip().split()
        if key in prompts:
            try:
                wav_prompt = prompts[key]
                sim = model.compute_similarity(wav_prompt, wav_gen)
                print(wav_prompt, wav_gen, sim)
                fout.write(f"{sim} {wav_prompt} {wav_gen}\n")
                if not math.isnan(sim):
                    total += sim
                    count += 1
            except Exception as e:
                print(e)
                continue
    fout.write("AVG speaker similarity {:.3f}\n".format(total / count))

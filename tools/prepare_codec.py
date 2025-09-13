#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025 Hao Yin(1049755192@qq.com)

# prepare the codec.jsonl file for the tts flow inference

import json
import re
import sys

result_jsonl = sys.argv[1]
test_jsonl = sys.argv[2]
codec_jsonl = sys.argv[3]

# extract llm_token from the result.jsonl
llm_token_list = []
with open(result_jsonl) as f:
    for line in f:
        arr = line.strip()
        numbers = re.findall(r"speech_(\d+)", arr)
        result = " ".join(numbers)
        llm_token_list.append(result)

# load the test jsonl file
json_lines = open(test_jsonl).readlines()
assert len(llm_token_list) == len(
    json_lines
), "llm_token_list and json_lines length mismatch"

data = []
for i, line in enumerate(json_lines):
    item = json.loads(line)
    # add the llm_token to the item
    item["llm_token"] = llm_token_list[i]
    data.append(item)

# save the data to the codec.jsonl file
with open(codec_jsonl, "w") as f:
    for x in data:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

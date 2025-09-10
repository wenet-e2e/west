#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025 Chengdong Liang(liangchengdongd@qq.com)

import json
import sys

test_json_path = sys.argv[1]
output_text_path = sys.argv[2]
output_hyp_ref_json = sys.argv[3]

text = []
with open(output_text_path, 'r') as fin:
    for line in fin:
        data = json.loads(line.strip())
        text.append(data['txt'])


with open(test_json_path, "r") as fref, open(output_hyp_ref_json, "w") as fjson:
    for idx, line in enumerate(fref):
        data = json.loads(line.strip())
        new_dict = {}
        new_dict['ref'] = data["ref"]
        new_dict['hyp'] = text[idx]
        fjson.write(json.dumps(new_dict, ensure_ascii=False))
        fjson.write("\n")

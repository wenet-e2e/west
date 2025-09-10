#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025 Chengdong Liang(liangchengdongd@qq.com)

import json
import re
import string
import sys

from word2number import w2n


def is_string_in_string(text, candidate):
    return all(x in text for x in candidate.split(" "))


def is_list_in_string(text, candidate):
    return any(
        [
            is_string_in_string(text, x)
            if isinstance(x, str) else is_list_in_string(text, x)
            for x in candidate
        ]
    )


def clean_punctuation(value):
    punctuation = string.punctuation
    punctuation = punctuation.replace("'", "")
    value = re.sub(f"[{punctuation}]", " ", value)
    return value


if __name__ == "__main__":

    pred_gt_json_file = sys.argv[1]

    acc = 0
    num = 0

    with open(pred_gt_json_file, "r") as f:
        for line in f:
            pred_gt = json.loads(line.strip())

            pred = pred_gt["hyp"]
            gt = pred_gt["ref"]

            pred = clean_punctuation(pred)
            pred = pred.lower()

            if not isinstance(gt, list):
                gt = [gt,]
            gt = [clean_punctuation(x) for x in gt]
            gt = [x.lower().strip() for x in gt]

            try:
                gt_number = [str(w2n.word_to_num(x.lower())) for x in gt]
            except Exception as e:
                print(e)
                gt_number = gt

            if is_list_in_string(pred, gt):
                acc += 1
            elif is_list_in_string(pred, gt_number):
                acc += 1
            else:
                print("======================================================")
                print("question: ", pred_gt["question"])
                print("ref=", pred_gt["ref"])
                print("hyp=", pred_gt["hyp"])
            num += 1

    print("======================================================")
    print(f"{acc=}")
    print(f"{num=}")
    print("======================================================")

    acc = acc / num * 100

    print("======================================================")
    print(f"{acc=}")
    print("======================================================")

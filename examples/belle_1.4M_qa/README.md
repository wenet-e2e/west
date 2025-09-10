## Dataset

- ASR data: AIShell2
- SpeechQA data: [Belle_1.4M-SLAM-Omni](https://huggingface.co/datasets/worstchan/Belle_1.4M-SLAM-Omni) dataset is prepared for the reproduction of SLAM-Omni. This is a multi-round Chinese spoken dialogue training dataset.

## Tutorial

First, prepare the train data `data/train.jsonl`, the data is like:
1. ASR data
```
{"wav": "/data_aishell/wav/train/S0002/BAC009S0002W0122.wav", "txt": "而对楼市成交抑制作用最大的限购"}
{"wav": "/data_aishell/wav/train/S0002/BAC009S0002W0123.wav", "txt": "也成为地方政府的眼中钉"}
```
2. QA data
```
{"key": "train-00000-of-01601-idx-1-train_3_5M_CN_ready4cosy_wo_code_switching-16012449-2", "wav": "train-00000-of-01601/train-00000-of-01601-idx-1-train_3.5M_CN_ready4cosy_wo_code_switching-16012449-2.wav", "messages": [{"role": "user", "content": "给定一段文本和关键词列表，删除文本中包含所有给定关键词的子字符串。\n文本：\"这是一个测试句子，目的是看看模型是否可以正确地从这个句子中删除关键词。\"\\n关键词列表：[‘测试’，‘模型’]"}, {"role": "assistant", "content": "删除包含所有给定关键词的子字符串后，文本变为：\"这是一个句子，目的是看看是否可以正确地从这个句子中删除关键词。\""}, {"role": "user", "content": {"type": "audio", "audio": "train-00000-of-01601/train-00000-of-01601-idx-1-train_3.5M_CN_ready4cosy_wo_code_switching-16012449-2.wav"}}, {"role": "assistant", "content": "好的，请稍等一下，现在我会将文本中的所有逗号替换为空格。处理后文本为：\"这是一个句子 目的是看看是否可以正确地从这个句子中删除关键词。\"。处理结果如何？"}]}
{"key": "train-00000-of-01601-idx-2-train_3_5M_CN_ready4cosy_wo_code_switching-28189110-1", "wav": "train-00000-of-01601/train-00000-of-01601-idx-2-train_3.5M_CN_ready4cosy_wo_code_switching-28189110-1.wav", "messages": [{"role": "user", "content": {"type": "audio", "audio": "train-00000-of-01601/train-00000-of-01601-idx-2-train_3.5M_CN_ready4cosy_wo_code_switching-28189110-1.wav"}}, {"role": "assistant", "content": "在绿野上，羚羊奔跑\n鸟语花香在心头荡漾\n涓涓小溪，蜿蜒、潺潺\n绿树成荫，凉雨淅淅沥沥\n自然的美景，如此神奇\n让我们沉迷，无法自拔\n在这美景之中，心灵得以宁静\n如此小小的悦动，细腻而清新"}]}
```

We train the QA model in two stages. In the first stage, we train the ASR model using AIShell2 dataset. In the second stage, we train the QA model using AIShell2 dataset and Belle_1.4M-SLAM-Omni dataset.

To train the ASR model, please refer to the [ASR tutorial](../aishell2/asr).

To train the QA model, just run
> NOTE: The train data is the combination of ASR data and QA data.
```shell
bash run.sh --stage train
```

To decode, just prepare the QA test data `data/test.jsonl` the same as train. the just run

```shell
bash run.sh --stage decode
```

## Result

|  audio encoder  |  llm model |            train data            | aishell2 test (WER) | speech qa test (ACC) | Details                              |
|:---------------:|:----------:|:--------------------------------:|:--------------------:|:-------------------:|:------------------------------------:|
| firered-asr-aed | Qwen3-1.7b |  aishell2 + Belle_1.4M-SLAM-Omni |     4.49 %           |      70.0%          | 4 A800 GPUS, pack 25000, 20000 steps |

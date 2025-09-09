# Performance Record

## Dataset

- ASR data: AIShell2
- SpeechQA data: [Belle_1.4M-SLAM-Omni](https://huggingface.co/datasets/worstchan/Belle_1.4M-SLAM-Omni) dataset is prepared for the reproduction of SLAM-Omni. This is a multi-round Chinese spoken dialogue training dataset.

## Result

|  audio encoder  |  llm model |            train data            | aishell2 test (WER) | speech qa test (ACC) |
|:---------------:|:----------:|:--------------------------------:|:--------------------:|:-------------------:|
| firered-asr-aed | Qwen3-1.7b |  aishell2 + Belle_1.4M-SLAM-Omni |     4.49 %           |      70.0%          |

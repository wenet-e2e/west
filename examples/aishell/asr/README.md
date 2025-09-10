## Tutorial

First, prepare the train data `data/train.jsonl`, the data is like:

```
{"wav": "/data_aishell/wav/train/S0002/BAC009S0002W0122.wav", "txt": "而对楼市成交抑制作用最大的限购"}
{"wav": "/data_aishell/wav/train/S0002/BAC009S0002W0123.wav", "txt": "也成为地方政府的眼中钉"}
```
where `wav` is the wav path, `txt` is the transcript.

To train the model, just run

``` shell
bash run.sh --stage train
```

To decode, just prepare the test data `data/test.jsonl` the same as train. then just run

``` shell
bash run.sh --stage decode
```

## Results

| LLM        | Speech Encoder | LoRA | test CER | Details                              |
|------------|----------------|------|----------|--------------------------------------|
| Qwen3-1.7B | firered        | Yes  | 4.17     | 8 A800 GPUs, pack 18000, 6000 steps  |
| Qwen2-7B   | firered        | No   | 4.01     | 4 A800 GPUs, pack 10000, 5000 steps  |
| Qwen2-7B   | paraformer     | Yes  | 3.51     | 4 A800 GPUs, pack 10000, 4000 steps  |

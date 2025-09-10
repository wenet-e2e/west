## Tutorial

First, prepare the train data `data/train.jsonl`, the data is like:

```
{"key": "IC0001W0001", "wav": "AISHELL-2/iOS/data/wav/C0001/IC0001W0001.wav", "txt": "厨房用具"}
{"key": "IC0001W0002", "wav": "AISHELL-2/iOS/data/wav/C0001/IC0001W0002.wav", "txt": "电压力锅"}
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

| LLM        | Speech Encoder | LoRA | test CER | Details                               |
|------------|----------------|------|----------|---------------------------------------|
| Qwen3-1.7B | firered        | No   | 5.41     | 4 A800 GPUs, pack 18000, 10000 steps  |

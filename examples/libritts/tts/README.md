## Tutorial

### TouchTTS
First, prepare the train data `data/train.jsonl`, the data is like:

```
{"wav": "/LibriTTS/train-clean-100/103/1241/103_1241_000000_000001.wav", "txt": "matthew Cuthbert is surprised"}
{"wav": "/LibriTTS/train-clean-100/1034/121119/1034_121119_000064_000001.wav", "txt": "What am I?--the law."}
```
where `wav` is the wav path, `txt` is the transcript.

To train the model, just run

``` shell
bash run.sh --stage train
```

To decode, prepare the test data `data/test.jsonl`, the data is like:

```
{"key": "1995_1837_000010_000000", "wav": "/LibriTTS/test-clean/1995/1837/1995_1837_000010_000000.wav", "txt": "Child? ", "syn_txt": "Good bye."}
{"key": "7127_75947_000057_000000", "wav": "/LibriTTS/test-clean/7127/75947/7127_75947_000057_000000.wav", "txt": "\"Yes.\" ", "syn_txt": "Go to her.\""}
```
where `key` is the unique id,
`wav` is the prompt wav path, `txt` is the prompt text corresponding to the `wav`,  
`syn_txt` is the text to be synthesized.


``` shell
bash run.sh --stage decode
```

### TouchFlow
First, prepare the train data `data/train.jsonl`, the data is like:

```
{"wav": "/LibriTTS/train-clean-100/103/1241/103_1241_000000_000001.wav", "txt": "matthew Cuthbert is surprised"}
{"wav": "/LibriTTS/train-clean-100/1034/121119/1034_121119_000064_000001.wav", "txt": "What am I?--the law."}
```
where `wav` is the wav path, `txt` is the transcript.

To train the model, just run

``` shell
bash run.sh --stage train
```

To decode, prepare the test data `data/test.flow.jsonl`, the data is like:

```
{"key": "3570_5694_000010_000007", "wav": "/LibriTTS/test-clean/3570/5694/3570_5694_000010_000007.wav", "syn_txt": "We wouldn't engineer long.\"", "llm_token": "2907 7 386 744 1470 2399 3914 2935 2323 3593 4056 2334 1147 3768 962 2680 2323 3489 409 489 2735 1784 1883 1449 2548 304 1223 2089 3438 1974 792 386 2103 393 393"}
```
where `key` is the unique id, `wav` is the prompt wav path,
`llm_token` is the speech token to be synthesized,
`syn_txt` is the text corresponding to the `llm_token`.


``` shell
bash run.sh --stage decode
```

## Results

| LLM        | WER (%) | #SUB | #INS + DEL | SS     | Details                                   |
|------------|---------|------|------------|--------|-------------------------------------------|
| Qwen2-0.5B | 5.15    | 217  | 92         |  0.847 |LLM: 8 A800 GPUs, pack 20000, 40000 steps<br>Flow: 8 3090 GPUs, batch 64, 50000 steps  |

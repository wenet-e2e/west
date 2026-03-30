## Tutorial

### TouchTTS
First, prepare the train data `data/train.jsonl`, the data is like:

```
{"wav": "/LibriTTS/train-clean-100/103/1241/103_1241_000000_000001.wav", "txt": "matthew Cuthbert is surprised"}
{"wav": "/LibriTTS/train-clean-100/1034/121119/1034_121119_000064_000001.wav", "txt": "What am I?--the law."}
```
where `wav` is the wav path, `txt` is the transcript.

Then, prepare the pretrained LLM model, such as Qwen/Qwen2.5-0.5B-Instruct,
and add speech tokens to model & tokenizer like:
```
# for speech_tokenizer_v1_25hz
python add_speech_tokens.py Qwen/Qwen2.5-0.5B-Instruct 4096 Qwen/Qwen2.5-0.5B-Audio-VQ

# for speech_tokenizer_v2_25hz; speech_tokenizer_v3_25hz
python add_speech_tokens.py Qwen/Qwen2.5-0.5B-Instruct 6561 Qwen/Qwen2.5-0.5B-Audio-FSQ
```

To train the model, just run

``` shell
bash run_llm.sh --stage train
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
bash run_llm.sh --stage decode
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
bash run_flow.sh --stage train
```

To decode, prepare the test data `data/test.flow.jsonl`, the data is like:

```
{"key": "3570_5694_000010_000007", "wav": "/LibriTTS/test-clean/3570/5694/3570_5694_000010_000007.wav", "syn_txt": "We wouldn't engineer long.\"", "llm_token": "2907 7 386 744 1470 2399 3914 2935 2323 3593 4056 2334 1147 3768 962 2680 2323 3489 409 489 2735 1784 1883 1449 2548 304 1223 2089 3438 1974 792 386 2103 393 393"}
```
where `key` is the unique id, `wav` is the prompt wav path,
`llm_token` is the speech token to be synthesized,
`syn_txt` is the text corresponding to the `llm_token`.


``` shell
bash run_flow.sh --stage decode
```

## Results

### LibriTTS
Trained on LibriTTS (～585 hours, all train data merged). To support long-form speech synthesis, two utterances from the same speaker are randomly concatenated during training.

**Test set:** 500 utterances randomly sampled from LibriTTS test-clean.


| LLM        | Tokenizer                  | WER (%) | #N   | #SUB | #INS + DEL | SS     | Details                                                                 |
|------------|----------------------------|---------|------|------|------------|--------|-------------------------------------------------------------------------|
| Qwen2-0.5B | speech_tokenizer_v1_25hz   | 5.56    | 5894 | 269  | 59         | 0.834  | LLM: 8 A800 GPUs, pack 20000, 40000 steps<br>Flow: 8 A800 GPUs, batch 32, 18000 steps |
| Qwen2-0.5B | speech_tokenizer_v2_25hz   | 3.87    | 5894 | 172  | 56         | 0.824  | LLM: 8 A800 GPUs, pack 20000, 40000 steps<br>Flow: 8 A800 GPUs, batch 32, 18000 steps |
| Qwen2-0.5B | speech_tokenizer_v3_25hz   | 3.55    | 5894 | 161  | 48         | 0.837  | LLM: 8 A800 GPUs, pack 20000, 40000 steps<br>Flow: 8 A800 GPUs, batch 32, 18000 steps |

### LargeData

Trained on ~190k hours of data from [EMILIA](https://huggingface.co/datasets/amphion/Emilia-Dataset), [LibriTTS](https://www.openslr.org/60/), and in-house datasets.

**Test set:** [seed-tts-zh](https://github.com/BytedanceSpeech/seed-tts-eval) and [seed-tts-en](https://github.com/BytedanceSpeech/seed-tts-eval) from the [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) benchmark.


| LLM        | Tokenizer                  | mode     | spk            | testset | CER/WER (%) | #N    | #SUB | #INS + DEL | SS         |
|------------|----------------------------|----------| ---------------|---------|-------------|-------|------|------------|------------|
| Qwen2-0.5B | speech_tokenizer_v3_25hz   | pretrain |  -             | test-zh | 1.56        | 42241 | 562  | 96         | 0.812      |
| Qwen2-0.5B | speech_tokenizer_v3_25hz   | pretrain |  -             | test-en | 2.34        | 11820 | 212  | 65         | 0.822      |
| Qwen2-0.5B | speech_tokenizer_v3_25hz   | SFT      | biaobei female | test-zh | **1.25**    | 42241 | 503  | 24         | **0.869**  |
| Qwen2-0.5B | speech_tokenizer_v3_25hz   | SFT      | internal       | test-en | **1.77**    | 11820 | 175  | 34         | **0.911**  |

- **spk (SFT)**
  - `biaobei female`: [标贝中文标准女声音库](https://www.data-baker.com/open_source.html).
  - `internal`: In-house English speaker (female); **1056** utterances, **~1.5** hours in total.

- **Training details**
```
Pretrain:
LLM: 8 A800 GPUs, pack 20000, 264k steps, lr 3e-4
Flow: 8 A800 GPUs, batch 64, 100k steps, lr 3e-4

SFT:
LLM: 8 A800 GPUs, pack 20000, 7k steps, lr 4e-4
Flow: 8 A800 GPUs, batch 64, 7k steps, lr 3e-4
```

**CER/WER comparison with CosyVoice series (Seed-TTS test-zh / test-en)**

| Model                                 | test-zh CER (%) | test-en WER (%) |
|---------------------------------------|-----------------|-----------------|
| Qwen2-0.5B + speech_tokenizer_v3_25hz | *1.56*          | *2.34*          |
| CosyVoice                             | 3.63            | 4.29            |
| CosyVoice2                            | 1.45            | 2.57            |
| CosyVoice3-0.5B                       | 1.16            | 2.02            |

---
## Field reference

| Field | Meaning |
|------|---------|
| `txt` | Text aligned with the reference audio (prompt transcript). |
| `wav` | Path to the reference / prompt waveform. |
| `syn_txt` | Target text to synthesize (used at inference). |
| `spk` | Speaker id or short description (SFT). |
| `ins` | Speaker style instruction (SFT). |

**Special tokens**:

- `<|spk_eos|>` — end of speaker block
- `<|ins_eos|>` — end of instruction block
- `<|audio_bos|>` — start of discrete audio tokens

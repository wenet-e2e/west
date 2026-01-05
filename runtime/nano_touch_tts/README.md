# Nano TouchTTS

On-device, LLM based, streaming text to speech.


## How to Build?

``` bash
cmake -B build
cmake --build build
```

## How to Run?

1. Get pretrained model from (TODO)
2. run the following command

```
export GLOG_logtostderr=1
export GLOG_v=2

mdir=touch_tts_models
odir=output_wavs

mkdir -p $odir
./build/src/bin/touch_tts_gen \
  --llm_model_path=$mdir/llm.gguf \
  --flow_model_path=$mdir/flow.onnx \
  --hifigan_model_path=$mdir/vocoder.onnx \
  --s3_model_path=$mdir/speech_tokenizer_v2.onnx \
  --speaker_model_path=$mdir/campplus.onnx \
  --prompt_wav=test_data/BAC009S0764W0121.wav \
  --prompt_text="甚至出现交易几乎停滞的情况" \
  --syn_text_file=test_data/syn.txt \
  --output_dir=$odir \
  --sample_rate=22050
```

# Nano TouchASR

On-device, LLM based, streaming speech recognition solution.

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

dir=touch_asr_models
./build/src/bin/touch_asr_main \
    --speech_encoder_model=$dir/encoder.onnx \
    --ctc_model=$dir/ctc.onnx \
    --ctc_tokens_file=$dir/ctc_tokens.txt \
    --projector_model=$dir/projector.qwen2.onnx \
    --llm_model=$dir/Qwen2.5-0.5B-Instruct.gguf \
    --wav_file=test_data/BAC009S0764W0121.wav
```


## Performance benchmark

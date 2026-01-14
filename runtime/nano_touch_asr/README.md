# Nano TouchASR

On-device, LLM based, streaming speech recognition solution.

## How to Build?

``` bash
cmake -B build
cmake --build build


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

Benchmark models:
* LLM: Qwen2.5-0.5B-Instruct, quantize method: 4bits k-quant
* encoder+ctc+projector: 30M conformer, 8bits dynamic quant

| Device   | Chip        | Quantize | RTF       | chunk(320ms) compute latency(ms) | LLM compute latency(ms) |
|----------|-------------|----------|-----------|----------------------------------|-------------------------|
| Mac Mini | M4          | N        | 0.0589635 | 7.70896                          | 160.12                  |
| Mac Mini | M4          | Y        | 0.057156  | 10.0387                          | 117.39                  |
| Mi Pad   | Qualcomm 8+ | N        | 0.415896  | 25.1398                          | 1560.32                 |
| Mi Pad   | Qualcomm 8+ | Y        | 0.460557  | 16.3114                          | 1897.8                  |



## Cross Compilation

### For ARM64 Linux (on macOS)

```bash
brew tap messense/macos-cross-toolchains
brew install aarch64-unknown-linux-gnu
cmake -B build-aarch64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-aarch64-linux-macos.cmake
cmake --build build-aarch64
```

### For Android

1. Install Android NDK (via Android Studio or standalone)

2. Set NDK path and build:

```bash
# Set NDK path (adjust to your NDK version)
export ANDROID_NDK=$HOME/Library/Android/sdk/ndk/26.1.10909125

# Build for Android arm64-v8a
cmake -B build-android -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-android-arm64.cmake
cmake --build build-android
```

3. Push to Android device and run:

```bash
adb push build-android/src/bin/touch_asr_main /data/local/tmp/
adb push touch_asr_models /data/local/tmp/
adb shell "cd /data/local/tmp && ./touch_asr_main ..."
```

// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#ifndef MODEL_HIFI_GAN_H_
#define MODEL_HIFI_GAN_H_

#include <string>
#include <vector>

#include "model/onnx_model.h"

namespace wenet {

class HifiGan {
 public:
  explicit HifiGan(const std::string& model_path);

  // Convert mel spectrogram to audio waveform
  // Input: mel spectrogram (T, num_mels)
  // Output: audio waveform
  void Forward(const std::vector<std::vector<float>>& mel,
               std::vector<float>* audio);

 private:
  OnnxModel model_;
  const int kNumMels = 80;
};

}  // namespace wenet

#endif  // MODEL_HIFI_GAN_H_

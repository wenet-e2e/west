// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#include "model/hifi_gan.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "utils/log.h"

namespace wenet {

HifiGan::HifiGan(const std::string& model_path) : model_(model_path) {}

void HifiGan::Forward(const std::vector<std::vector<float>>& mel,
                      std::vector<float>* audio) {
  // Input: mel spectrogram (1, num_mels, T)
  // Output: audio waveform (1, 1, T * hop_size)
  int num_frames = mel.size();
  if (num_frames == 0) {
    LOG(WARNING) << "Empty mel spectrogram";
    return;
  }

  // Convert mel from (T, num_mels) to (1, num_mels, T)
  std::vector<float> mel_flattened(kNumMels * num_frames);
  for (int i = 0; i < num_frames; ++i) {
    for (int j = 0; j < kNumMels; ++j) {
      mel_flattened[j * num_frames + i] = mel[i][j];
    }
  }

  const int64_t mel_shape[] = {1, kNumMels, num_frames};
  auto mel_ort = Ort::Value::CreateTensor<float>(
      model_.memory_info(), mel_flattened.data(), mel_flattened.size(),
      mel_shape, 3);

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(mel_ort));

  auto outputs_ort = model_.Run(ort_inputs);

  // Get output audio
  const float* output_data = outputs_ort[0].GetTensorData<float>();
  auto output_shape = outputs_ort[0].GetTensorTypeAndShapeInfo().GetShape();
  int audio_length = 1;
  for (auto dim : output_shape) {
    audio_length *= dim;
  }
  audio->assign(output_data, output_data + audio_length);
  // Convert normalized audio [-1, 1] to int16 range [-32768, 32767]
  for (int i = 0; i < audio_length; i++) {
    (*audio)[i] = std::max(-1.0f, std::min(1.0f, (*audio)[i])) * 32767.0f;
  }
}

}  // namespace wenet

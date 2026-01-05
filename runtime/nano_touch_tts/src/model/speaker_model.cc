// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#include "model/speaker_model.h"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>


#include "frontend/fbank.h"
#include "frontend/wav.h"

namespace wenet {

SpeakerModel::SpeakerModel(const std::string& model_path)
    : model_(model_path) {}

void SpeakerModel::ApplyMean(std::vector<std::vector<float>>* feat,
                             unsigned int feat_dim) {
  std::vector<float> mean(feat_dim, 0);
  for (auto& i : *feat) {
    std::transform(i.begin(), i.end(), mean.begin(), mean.begin(),
                   std::plus<>{});
  }
  std::transform(mean.begin(), mean.end(), mean.begin(),
                 [&](const float d) { return d / feat->size(); });
  for (auto& i : *feat) {
    std::transform(i.begin(), i.end(), mean.begin(), i.begin(), std::minus<>{});
  }
}

void SpeakerModel::ExtractEmbedding(
    const std::vector<std::vector<float>>& feats, std::vector<float>* embed) {
  int num_feats = feats.size();
  std::vector<float> feat_flattened(kNumBins * num_feats);
  for (int i = 0; i < num_feats; i++) {
    for (int j = 0; j < kNumBins; j++) {
      feat_flattened[i * kNumBins + j] = feats[i][j];
    }
  }
  const int64_t inputs_shape[] = {1, num_feats, kNumBins};
  auto inputs_ort = Ort::Value::CreateTensor<float>(
      model_.memory_info(), feat_flattened.data(), feat_flattened.size(),
      inputs_shape, 3);
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(inputs_ort));
  auto outputs_ort = model_.Run(ort_inputs);
  const float* output_data = outputs_ort[0].GetTensorData<float>();
  embed->assign(
      output_data,
      output_data + outputs_ort[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
}

void SpeakerModel::ExtractEmbedding(const std::vector<float>& pcm,
                                    std::vector<float>* embed) {
  // Norm point to -1.0 to 1.0 by the last
  bool scale_input_to_unit = true;
  Fbank fbank(kNumBins, 16000, 400, 160, 20, true, scale_input_to_unit);
  std::vector<std::vector<float>> feats;
  fbank.Compute(pcm, &feats);
  ApplyMean(&feats, kNumBins);
  ExtractEmbedding(feats, embed);
}

void SpeakerModel::ExtractEmbedding(const std::string& wav_file,
                                    std::vector<float>* embed) {
  std::vector<float> pcm;
  WavReader wav_reader(wav_file);
  CHECK_EQ(wav_reader.num_channel(), 1) << "Only support mono audio";
  CHECK_EQ(wav_reader.bits_per_sample(), 16) << "Only support 16 bits audio";
  pcm.assign(wav_reader.data(), wav_reader.data() + wav_reader.num_samples());
  ExtractEmbedding(pcm, embed);
}

}  // namespace wenet

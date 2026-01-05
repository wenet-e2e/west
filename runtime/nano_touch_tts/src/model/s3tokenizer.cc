// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#include <utility>
#include <string>
#include <vector>

#include "model/s3tokenizer.h"
#include "frontend/fbank.h"
#include "frontend/wav.h"

namespace wenet {

S3Tokenizer::S3Tokenizer(const std::string& model_path)
    : s3_model_(model_path) {}

void S3Tokenizer::Tokenize(const std::vector<std::vector<float>>& feats,
                           std::vector<int32_t>* tokens) {
  int num_feats = feats.size();
  // Convert feat from vector<vector<float>>(T, 128) to vector<float>(128, T)
  std::vector<float> feat_flattened(kNumBins * num_feats);
  for (int i = 0; i < kNumBins; i++) {
    for (int j = 0; j < num_feats; j++) {
      feat_flattened[i * num_feats + j] = feats[j][i];
    }
  }
  const int64_t inputs_shape[] = {1, kNumBins, num_feats};
  auto inputs_ort = Ort::Value::CreateTensor<float>(
      s3_model_.memory_info(), feat_flattened.data(), feat_flattened.size(),
      inputs_shape, 3);
  std::vector<int32_t> inputs_len = {num_feats};
  const int64_t inputs_len_shape[] = {1};
  auto inputs_len_ort = Ort::Value::CreateTensor<int32_t>(
      s3_model_.memory_info(), inputs_len.data(), inputs_len.size(),
      inputs_len_shape, 1);
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(inputs_ort));
  ort_inputs.push_back(std::move(inputs_len_ort));
  auto outputs_ort = s3_model_.Run(ort_inputs);

  int len = outputs_ort[0].GetTensorTypeAndShapeInfo().GetShape()[1];
  const int32_t* output_data = outputs_ort[0].GetTensorData<int32_t>();
  tokens->assign(output_data, output_data + len);
}

void S3Tokenizer::Tokenize(const std::vector<float>& pcm,
                           std::vector<int32_t>* tokens) {
  LogMelSpectrogram fbank(kNumBins, 16000, 400, 160);
  std::vector<std::vector<float>> feats;
  fbank.Compute(pcm, &feats);
  Tokenize(feats, tokens);
}

void S3Tokenizer::Tokenize(const std::string& wav_file,
                           std::vector<int32_t>* tokens) {
  WavReader wav_reader(wav_file);
  CHECK_EQ(wav_reader.num_channel(), 1) << "Only support mono audio";
  CHECK_EQ(wav_reader.bits_per_sample(), 16) << "Only support 16 bits audio";
  std::vector<float> wave(wav_reader.data(),
                          wav_reader.data() + wav_reader.num_samples());
  Tokenize(wave, tokens);
}

}  // namespace wenet

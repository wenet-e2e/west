// Copyright (c) 2025 Personal (Binbin Zhang)

#include "model/ctc_model.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace wenet {

CtcModel::CtcModel(const std::string& model_path) : model_(model_path) {}

void CtcModel::Forward(const std::vector<std::vector<float>>& encoder_outs,
                       std::vector<int>* out_ids) {
  int num_frames = encoder_outs.size();
  int hidden_size = encoder_outs[0].size();
  std::vector<float> encoder_outs_data;
  for (int i = 0; i < num_frames; ++i) {
    encoder_outs_data.insert(encoder_outs_data.end(), encoder_outs[i].begin(),
                             encoder_outs[i].end());
  }
  const int64_t encoder_outs_shape[3] = {1, num_frames, hidden_size};
  Ort::Value encoder_outs_ort = Ort::Value::CreateTensor<float>(
      model_.memory_info(), encoder_outs_data.data(), encoder_outs_data.size(),
      encoder_outs_shape, 3);

  std::vector<Ort::Value> inputs;
  inputs.emplace_back(std::move(encoder_outs_ort));
  std::vector<Ort::Value> ort_outputs = model_.Run(inputs);

  float* out_prob_data = ort_outputs[0].GetTensorMutableData<float>();
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();

  int num_outputs = type_info.GetShape()[1];
  int output_dim = type_info.GetShape()[2];
  for (int i = 0; i < num_outputs; i++) {
    std::vector<float> prob(out_prob_data + i * output_dim,
                            out_prob_data + i * output_dim + output_dim);
    int max_index = std::max_element(prob.begin(), prob.end()) - prob.begin();
    if (max_index != 0 && max_index != last_idx_) {  // 0 is blank
      cached_idx_.push_back(max_index);
    }
    last_idx_ = max_index;
  }
  out_ids->assign(cached_idx_.begin(), cached_idx_.end());
}

}  // namespace wenet

// Copyright (c) 2025 Personal (Binbin Zhang)

#include "model/projector.h"

#include <string>
#include <utility>
#include <vector>

namespace wenet {

Projector::Projector(const std::string& model_path) : model_(model_path) {}

void Projector::Forward(const std::vector<std::vector<float>>& input,
                        std::vector<std::vector<float>>* output) {
  int num_frames = input.size();
  int input_dim = input[0].size();

  std::vector<float> input_data;
  for (int i = 0; i < num_frames; ++i) {
    input_data.insert(input_data.end(), input[i].begin(), input[i].end());
  }

  const int64_t input_shape[3] = {1, num_frames, input_dim};
  Ort::Value input_ort =
      Ort::Value::CreateTensor<float>(model_.memory_info(), input_data.data(),
                                      input_data.size(), input_shape, 3);

  std::vector<Ort::Value> inputs;
  inputs.emplace_back(std::move(input_ort));
  std::vector<Ort::Value> ort_outputs = model_.Run(inputs);

  float* out_data = ort_outputs[0].GetTensorMutableData<float>();
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();

  int num_outputs = type_info.GetShape()[1];
  int output_dim = type_info.GetShape()[2];

  output->resize(num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    (*output)[i].resize(output_dim);
    memcpy((*output)[i].data(), out_data + i * output_dim,
           sizeof(float) * output_dim);
  }
}

}  // namespace wenet

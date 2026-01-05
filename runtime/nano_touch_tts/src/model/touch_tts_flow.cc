// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#include "model/touch_tts_flow.h"

#include <string>
#include <utility>
#include <vector>

#include "utils/log.h"

namespace wenet {

TouchTtsFlow::TouchTtsFlow(const std::string& flow_model_path)
    : model_(flow_model_path) {}

void TouchTtsFlow::Forward(const std::vector<std::vector<float>>& prompt_feats,
                           const std::vector<float>& prompt_spk_emb,
                           const std::vector<int32_t>& prompt_tokens,
                           const std::vector<int32_t>& llm_tokens,
                           std::vector<std::vector<float>>* gen_feats) {
  // inputs:
  //    prompt_feats (1, T, 80)
  //    prompt_spk_emb (1, 192)
  //    prompt_tokens (1, L1)
  //    llm_tokens (1, L2)
  // outputs:
  //    gen_feats (T, 80) -> (80, T)
  int num_feats = prompt_feats.size();
  // Convert feat from vector<vector<float>>(T, 80) to vector<float>(80, T)
  std::vector<float> feat_flattened(kNumBins * num_feats);
  for (int i = 0; i < num_feats; i++) {
    for (int j = 0; j < kNumBins; j++) {
      feat_flattened[i * kNumBins + j] = prompt_feats[i][j];
    }
  }
  const int64_t inputs_shape[] = {1, num_feats, kNumBins};
  auto prompt_feats_ort = Ort::Value::CreateTensor<float>(
      model_.memory_info(), feat_flattened.data(), feat_flattened.size(),
      inputs_shape, 3);
  const int64_t prompt_spk_emb_shape[] = {1, 192};
  auto prompt_spk_emb_ort = Ort::Value::CreateTensor<float>(
      model_.memory_info(), const_cast<float*>(prompt_spk_emb.data()),
      prompt_spk_emb.size(), prompt_spk_emb_shape, 2);
  const int64_t prompt_tokens_shape[] = {
      1, static_cast<int64_t>(prompt_tokens.size())};
  auto prompt_tokens_ort = Ort::Value::CreateTensor<int32_t>(
      model_.memory_info(), const_cast<int32_t*>(prompt_tokens.data()),
      prompt_tokens.size(), prompt_tokens_shape, 2);
  const int64_t llm_tokens_shape[] = {1,
                                      static_cast<int64_t>(llm_tokens.size())};
  auto llm_tokens_ort = Ort::Value::CreateTensor<int32_t>(
      model_.memory_info(), const_cast<int32_t*>(llm_tokens.data()),
      llm_tokens.size(), llm_tokens_shape, 2);
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(prompt_feats_ort));
  ort_inputs.push_back(std::move(prompt_spk_emb_ort));
  ort_inputs.push_back(std::move(prompt_tokens_ort));
  ort_inputs.push_back(std::move(llm_tokens_ort));
  auto outputs_ort = model_.Run(ort_inputs);
  const float* output_data = outputs_ort[0].GetTensorData<float>();
  int num_gen_feats = outputs_ort[0].GetTensorTypeAndShapeInfo().GetShape()[1];
  gen_feats->resize(num_gen_feats);
  for (int i = 0; i < num_gen_feats; i++) {
    (*gen_feats)[i].resize(kNumBins);
    for (int j = 0; j < kNumBins; j++) {
      (*gen_feats)[i][j] = output_data[i * kNumBins + j];
    }
  }
}

}  // namespace wenet

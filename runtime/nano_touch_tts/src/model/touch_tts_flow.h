// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#ifndef MODEL_TOUCH_TTS_FLOW_H_
#define MODEL_TOUCH_TTS_FLOW_H_

#include <memory>
#include <string>
#include <vector>

#include "model/onnx_model.h"

namespace wenet {

class TouchTtsFlow {
 public:
  explicit TouchTtsFlow(const std::string& flow_model_path);

  void Forward(const std::vector<std::vector<float>>& prompt_feats,
               const std::vector<float>& prompt_spk_emb,
               const std::vector<int32_t>& prompt_tokens,
               const std::vector<int32_t>& llm_tokens,
               std::vector<std::vector<float>>* gen_feats);

 private:
  OnnxModel model_;
  const int kNumBins = 80;
};

}  // namespace wenet

#endif  // MODEL_TOUCH_TTS_FLOW_H_

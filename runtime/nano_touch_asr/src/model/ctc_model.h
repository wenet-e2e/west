// Copyright (c) 2025 Personal (Binbin Zhang)

#ifndef MODEL_CTC_MODEL_H_
#define MODEL_CTC_MODEL_H_

#include <string>
#include <vector>

#include "model/onnx_model.h"

namespace wenet {

class CtcModel {
 public:
  explicit CtcModel(const std::string& model_path);
  void Forward(const std::vector<std::vector<float>>& encoder_outs,
               std::vector<int>* out_ids);
  void Reset() {
    cached_idx_.clear();
    last_idx_ = 0;
  }

 private:
  OnnxModel model_;

  std::vector<int> cached_idx_;
  int last_idx_ = 0;
};

}  // namespace wenet

#endif  // MODEL_CTC_MODEL_H_

// Copyright (c) 2025 Personal (Binbin Zhang)

#ifndef MODEL_PROJECTOR_H_
#define MODEL_PROJECTOR_H_

#include <string>
#include <vector>

#include "model/onnx_model.h"

namespace wenet {

class Projector {
 public:
  explicit Projector(const std::string& model_path);
  void Forward(const std::vector<std::vector<float>>& input,
               std::vector<std::vector<float>>* output);

 private:
  OnnxModel model_;
};

}  // namespace wenet

#endif  // MODEL_PROJECTOR_H_

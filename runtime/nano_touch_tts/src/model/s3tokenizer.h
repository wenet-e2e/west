// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#ifndef MODEL_S3TOKENIZER_H_
#define MODEL_S3TOKENIZER_H_

#include <string>
#include <vector>

#include "model/onnx_model.h"

namespace wenet {

class S3Tokenizer {
 public:
  explicit S3Tokenizer(const std::string& model_path);

  void Tokenize(const std::vector<std::vector<float>>& feats,
                std::vector<int32_t>* tokens);
  void Tokenize(const std::vector<float>& pcm, std::vector<int32_t>* tokens);
  void Tokenize(const std::string& wav_file, std::vector<int32_t>* tokens);

 private:
  OnnxModel s3_model_;

  const int kNumBins = 128;
};

}  // namespace wenet

#endif  // MODEL_S3TOKENIZER_H_

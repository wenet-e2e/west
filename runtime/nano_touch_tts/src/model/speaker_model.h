// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#ifndef MODEL_SPEAKER_MODEL_H_
#define MODEL_SPEAKER_MODEL_H_

#include <string>
#include <vector>

#include "model/onnx_model.h"

namespace wenet {

class SpeakerModel {
 public:
  explicit SpeakerModel(const std::string& model_path);
  void ApplyMean(std::vector<std::vector<float>>* feat, unsigned int feat_dim);
  void ExtractEmbedding(const std::vector<std::vector<float>>& feats,
                        std::vector<float>* embed);
  void ExtractEmbedding(const std::vector<float>& pcm,
                        std::vector<float>* embed);
  void ExtractEmbedding(const std::string& wav_file, std::vector<float>* embed);

 private:
  OnnxModel model_;
  const int kNumBins = 80;
};

}  // namespace wenet
#endif  // MODEL_SPEAKER_MODEL_H_

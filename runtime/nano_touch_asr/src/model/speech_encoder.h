// Copyright (c) 2025 Personal (Binbin Zhang)

#ifndef MODEL_SPEECH_ENCODER_H_
#define MODEL_SPEECH_ENCODER_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "model/onnx_model.h"
#include "utils/log.h"

namespace wenet {

class SpeechEncoder {
 public:
  explicit SpeechEncoder(const std::string& model_path);
  void Reset();
  void Forward(const std::vector<std::vector<float>>& chunk_feats,
               std::vector<std::vector<float>>* out);

  int NumFramesForThisChunk(int chunk_idx) const {
    if (chunk_idx == 0) {
      int context = right_context_ + 1;  // Add current frame
      return (chunk_size_ - 1) * subsampling_rate_ + context;
    } else {
      return chunk_size_ * subsampling_rate_;
    }
  }

 private:
  OnnxModel model_;

  int encoder_output_size_ = 0;
  int num_blocks_ = 0;
  int cnn_module_kernel_ = 0;
  int head_ = 0;
  int subsampling_rate_ = 0;
  int right_context_ = 0;
  int sos_ = 0;
  int eos_ = 0;
  int chunk_size_ = 0;
  int num_left_chunks_ = 0;

  // caches
  Ort::Value att_cache_ort_{nullptr};
  Ort::Value cnn_cache_ort_{nullptr};
  std::vector<Ort::Value> encoder_outs_;
  // NOTE: Instead of making a copy of the xx_cache, ONNX only maintains
  //  its data pointer when initializing xx_cache_ort (see https://github.com/
  //  microsoft/onnxruntime/blob/master/onnxruntime/core/framework
  //  /tensor.cc#L102-L129), so we need the following variables to keep
  //  our data "alive" during the lifetime of decoder.
  std::vector<float> att_cache_;
  std::vector<float> cnn_cache_;

  std::vector<std::vector<float>> cached_feature_;
  int offset_ = 0;
};

}  // namespace wenet

#endif  // MODEL_SPEECH_ENCODER_H_

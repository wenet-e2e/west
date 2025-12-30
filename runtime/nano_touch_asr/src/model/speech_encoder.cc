// Copyright (c) 2025 Personal (Binbin Zhang)

#include "model/speech_encoder.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wenet {

SpeechEncoder::SpeechEncoder(const std::string& model_path)
    : model_(model_path) {
  // Read metadata
  auto model_metadata = model_.session()->GetModelMetadata();
  Ort::AllocatorWithDefaultOptions allocator;
  encoder_output_size_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("output_size", allocator)
          .get());
  num_blocks_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("num_blocks", allocator)
          .get());
  head_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("head", allocator).get());
  cnn_module_kernel_ =
      atoi(model_metadata
               .LookupCustomMetadataMapAllocated("cnn_module_kernel", allocator)
               .get());
  subsampling_rate_ =
      atoi(model_metadata
               .LookupCustomMetadataMapAllocated("subsampling_rate", allocator)
               .get());
  right_context_ =
      atoi(model_metadata
               .LookupCustomMetadataMapAllocated("right_context", allocator)
               .get());
  sos_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("sos_symbol", allocator)
          .get());
  eos_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("eos_symbol", allocator)
          .get());
  chunk_size_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("chunk_size", allocator)
          .get());
  num_left_chunks_ = atoi(
      model_metadata.LookupCustomMetadataMapAllocated("left_chunks", allocator)
          .get());

  LOG(INFO) << "Onnx Model Info:";
  LOG(INFO) << "\tencoder_output_size " << encoder_output_size_;
  LOG(INFO) << "\tnum_blocks " << num_blocks_;
  LOG(INFO) << "\thead " << head_;
  LOG(INFO) << "\tcnn_module_kernel " << cnn_module_kernel_;
  LOG(INFO) << "\tsubsampling_rate " << subsampling_rate_;
  LOG(INFO) << "\tright_context " << right_context_;
  LOG(INFO) << "\tsos " << sos_;
  LOG(INFO) << "\teos " << eos_;
  LOG(INFO) << "\tchunk_size " << chunk_size_;
  LOG(INFO) << "\tnum_left_chunks " << num_left_chunks_;

  Reset();
}

void SpeechEncoder::Reset() {
  offset_ = 0;
  encoder_outs_.clear();
  cached_feature_.clear();
  // Reset att_cache
  if (num_left_chunks_ > 0) {
    int required_cache_size = chunk_size_ * num_left_chunks_;
    offset_ = required_cache_size;
    att_cache_.resize(num_blocks_ * head_ * required_cache_size *
                          encoder_output_size_ / head_ * 2,
                      0.0);
    const int64_t att_cache_shape[] = {num_blocks_, head_, required_cache_size,
                                       encoder_output_size_ / head_ * 2};
    att_cache_ort_ =
        Ort::Value::CreateTensor<float>(model_.memory_info(), att_cache_.data(),
                                        att_cache_.size(), att_cache_shape, 4);
  } else {
    att_cache_.resize(0, 0.0);
    const int64_t att_cache_shape[] = {num_blocks_, head_, 0,
                                       encoder_output_size_ / head_ * 2};
    att_cache_ort_ =
        Ort::Value::CreateTensor<float>(model_.memory_info(), att_cache_.data(),
                                        att_cache_.size(), att_cache_shape, 4);
  }

  // Reset cnn_cache
  cnn_cache_.resize(
      num_blocks_ * encoder_output_size_ * (cnn_module_kernel_ - 1), 0.0);
  const int64_t cnn_cache_shape[] = {num_blocks_, 1, encoder_output_size_,
                                     cnn_module_kernel_ - 1};
  cnn_cache_ort_ =
      Ort::Value::CreateTensor<float>(model_.memory_info(), cnn_cache_.data(),
                                      cnn_cache_.size(), cnn_cache_shape, 4);
}

void SpeechEncoder::Forward(const std::vector<std::vector<float>>& chunk_feats,
                            std::vector<std::vector<float>>* out) {
  // 1. Prepare onnx required data, splice cached_feature_ and chunk_feats
  // chunk
  int num_frames = cached_feature_.size() + chunk_feats.size();
  const int feature_dim = chunk_feats[0].size();
  std::vector<float> feats;
  for (size_t i = 0; i < cached_feature_.size(); ++i) {
    feats.insert(feats.end(), cached_feature_[i].begin(),
                 cached_feature_[i].end());
  }
  for (size_t i = 0; i < chunk_feats.size(); ++i) {
    feats.insert(feats.end(), chunk_feats[i].begin(), chunk_feats[i].end());
  }
  const int64_t feats_shape[3] = {1, num_frames, feature_dim};
  Ort::Value feats_ort = Ort::Value::CreateTensor<float>(
      model_.memory_info(), feats.data(), feats.size(), feats_shape, 3);
  // offset
  int64_t offset_int64 = static_cast<int64_t>(offset_);
  Ort::Value offset_ort = Ort::Value::CreateTensor<int64_t>(
      model_.memory_info(), &offset_int64, 1, std::vector<int64_t>{}.data(), 0);
  // required_cache_size
  int64_t required_cache_size = chunk_size_ * num_left_chunks_;
  Ort::Value required_cache_size_ort = Ort::Value::CreateTensor<int64_t>(
      model_.memory_info(), &required_cache_size, 1,
      std::vector<int64_t>{}.data(), 0);
  // att_mask
  Ort::Value att_mask_ort{nullptr};
  std::vector<uint8_t> att_mask(required_cache_size + chunk_size_, 1);
  if (num_left_chunks_ > 0) {
    int chunk_idx = offset_ / chunk_size_ - num_left_chunks_;
    if (chunk_idx < num_left_chunks_) {
      for (int i = 0; i < (num_left_chunks_ - chunk_idx) * chunk_size_; ++i) {
        att_mask[i] = 0;
      }
    }
    const int64_t att_mask_shape[] = {1, 1, required_cache_size + chunk_size_};
    att_mask_ort = Ort::Value::CreateTensor<bool>(
        model_.memory_info(), reinterpret_cast<bool*>(att_mask.data()),
        att_mask.size(), att_mask_shape, 3);
  }

  std::vector<Ort::Value> inputs;
  inputs.emplace_back(std::move(feats_ort));
  inputs.emplace_back(std::move(required_cache_size_ort));
  inputs.emplace_back(std::move(att_cache_ort_));
  inputs.emplace_back(std::move(cnn_cache_ort_));
  inputs.emplace_back(std::move(att_mask_ort));
  std::vector<Ort::Value> ort_outputs = model_.Run(inputs);
  offset_ += static_cast<int>(
      ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1]);

  att_cache_ort_ = std::move(ort_outputs[1]);
  cnn_cache_ort_ = std::move(ort_outputs[2]);

  float* out_data = ort_outputs[0].GetTensorMutableData<float>();
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
  int num_outputs = type_info.GetShape()[1];
  int output_dim = type_info.GetShape()[2];
  out->resize(num_outputs);
  for (int i = 0; i < num_outputs; i++) {
    (*out)[i].resize(output_dim);
    memcpy((*out)[i].data(), out_data + i * output_dim,
           sizeof(float) * output_dim);
  }

  // Cache feature for next chunk
  const int cached_feature_size = 1 + right_context_ - subsampling_rate_;
  if (chunk_feats.size() >= cached_feature_size) {
    // TODO(Binbin Zhang): Only deal the case when
    // chunk_feats.size() > cached_feature_size here, and it's consistent
    // with our current model, refine it later if we have new model or
    // new requirements
    cached_feature_.resize(cached_feature_size);
    for (int i = 0; i < cached_feature_size; ++i) {
      cached_feature_[i] =
          chunk_feats[chunk_feats.size() - cached_feature_size + i];
    }
  }
}

}  // namespace wenet

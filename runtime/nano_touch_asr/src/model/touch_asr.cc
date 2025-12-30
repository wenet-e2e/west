// Copyright (c) 2025 Personal (Binbin Zhang)

#include "model/touch_asr.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "utils/log.h"

namespace wenet {

TouchASR::TouchASR(const TouchASRConfig& config)
    : config_(config), feature_config_(config.num_bins, config.sample_rate) {
  // Initialize feature pipeline
  feature_pipeline_ = std::make_unique<FeaturePipeline>(feature_config_);
  LOG(INFO) << "Initialized FeaturePipeline";

  // Initialize speech encoder
  speech_encoder_ =
      std::make_unique<SpeechEncoder>(config.speech_encoder_model);
  LOG(INFO) << "Initialized SpeechEncoder";

  // Initialize CTC model
  ctc_model_ = std::make_unique<CtcModel>(config.ctc_model);
  LOG(INFO) << "Initialized CtcModel";

  // Initialize CTC tokens
  ctc_tokens_ = std::make_unique<CtcTokens>(config.ctc_tokens_file);
  LOG(INFO) << "Initialized CtcTokens";

  // Initialize Projector
  projector_ = std::make_unique<Projector>(config.projector_model);
  LOG(INFO) << "Initialized Projector";

  // Initialize LLM
  llm_ = std::make_unique<LLM>(config.llm_model);
  LOG(INFO) << "Initialized LLM";
}

void TouchASR::Reset() {
  chunk_idx_ = 0;
  all_encoder_outs_.clear();
  feature_pipeline_->Reset();
  speech_encoder_->Reset();
}

bool TouchASR::DecodeStreaming(std::string* result) {
  bool finish = false;
  int num_required_frames = speech_encoder_->NumFramesForThisChunk(chunk_idx_);
  std::vector<std::vector<float>> chunk_feats;
  // If not okay, that means we reach the end of the input
  if (!feature_pipeline_->Read(num_required_frames, &chunk_feats)) {
    finish = true;
  }
  // Pad to chunk size if necessary (for end chunk)
  if (chunk_idx_ > 0 &&
      static_cast<int>(chunk_feats.size()) < num_required_frames) {
    int pad_frames = num_required_frames - chunk_feats.size();
    std::vector<float> zero_frame(feature_pipeline_->feature_dim(), 0.0f);
    for (int p = 0; p < pad_frames; ++p) {
      chunk_feats.push_back(zero_frame);
    }
  }
  // Speech encoder forward
  std::vector<std::vector<float>> encoder_outs;
  speech_encoder_->Forward(chunk_feats, &encoder_outs);
  // Accumulate encoder outputs for projector
  for (const auto& out : encoder_outs) {
    all_encoder_outs_.push_back(out);
  }
  // CTC decoding
  std::vector<int> ctc_ids;
  ctc_model_->Forward(encoder_outs, &ctc_ids);
  std::stringstream ss;
  for (int id : ctc_ids) {
    ss << ctc_tokens_->GetToken(id);
  }
  *result = ss.str();
  chunk_idx_++;
  return finish;
}

void TouchASR::DecodeNonStreaming(std::string* result) {
  CHECK(!all_encoder_outs_.empty()) << "No encoder outputs";
  std::vector<std::vector<float>> projected_outs;
  projector_->Forward(all_encoder_outs_, &projected_outs);
  // // Read projected outputs from file test_data/speech_embd.txt
  // std::ifstream file("test_data/speech_embd.qwen2.txt");
  // std::string line;
  // while (std::getline(file, line)) {
  //   std::istringstream iss(line);
  //   std::vector<float> emb;
  //   float f;
  //   while (iss >> f) {
  //     emb.push_back(f);
  //   }
  //   projected_outs.push_back(emb);
  // }
  // file.close();
  LOG(INFO) << "Projected outputs: " << projected_outs.size() << " frames";
  result->clear();
  llm_->Generate(projected_outs, result);
}

}  // namespace wenet

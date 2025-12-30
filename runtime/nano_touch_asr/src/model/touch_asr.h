// Copyright (c) 2025 Personal (Binbin Zhang)

#ifndef MODEL_TOUCH_ASR_H_
#define MODEL_TOUCH_ASR_H_

#include <memory>
#include <string>
#include <vector>

#include "frontend/feature_pipeline.h"
#include "model/ctc_model.h"
#include "model/ctc_tokens.h"
#include "model/llm.h"
#include "model/projector.h"
#include "model/speech_encoder.h"

namespace wenet {

struct TouchASRConfig {
  std::string speech_encoder_model;
  std::string ctc_model;
  std::string ctc_tokens_file;
  std::string projector_model;
  std::string llm_model;
  int num_bins = 80;
  int sample_rate = 16000;
};

class TouchASR {
 public:
  explicit TouchASR(const TouchASRConfig& config);

  // return true if all wav are decoded
  bool DecodeStreaming(std::string* result);
  void DecodeNonStreaming(std::string* result);

  void AcceptWaveform(const std::vector<float>& audio_samples) {
    feature_pipeline_->AcceptWaveform(audio_samples.data(),
                                      audio_samples.size());
  }

  void SetInputFinished() { feature_pipeline_->set_input_finished(); }

  // Reset the decoder state for new utterance
  void Reset();

 private:
  TouchASRConfig config_;
  FeaturePipelineConfig feature_config_;

  std::unique_ptr<FeaturePipeline> feature_pipeline_;
  std::unique_ptr<SpeechEncoder> speech_encoder_;
  std::unique_ptr<CtcModel> ctc_model_;
  std::unique_ptr<CtcTokens> ctc_tokens_;
  std::unique_ptr<Projector> projector_;
  std::unique_ptr<LLM> llm_;

  int chunk_idx_ = 0;
  std::vector<std::vector<float>> all_speech_embd_;
};

}  // namespace wenet

#endif  // MODEL_TOUCH_ASR_H_

// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#ifndef MODEL_TOUCH_TTS_LLM_H_
#define MODEL_TOUCH_TTS_LLM_H_

#include <string>
#include <vector>

#include "llama.h"  // NOLINT

namespace wenet {

class TouchTtsLlm {
 public:
  explicit TouchTtsLlm(const std::string& model_path);
  ~TouchTtsLlm();

  void Generate(const std::vector<llama_token>& prompt_tokens,
                std::string* response);
  void Generate(const std::string& prompt, std::string* response);
  void Generate(const std::vector<int32_t>& prompt_speech,
                const std::string& prompt_text,
                const std::string& syn_text,  // text to synthesize
                std::string* response_speech);

 private:
  std::string model_path_;

  llama_model_params model_params_;
  llama_model* model_;
  const llama_vocab* vocab_;
  llama_context_params ctx_params_;
  llama_context* ctx_;
  llama_sampler_chain_params smpl_params_;
  llama_sampler* smpl_;
};

}  // namespace wenet
#endif  // MODEL_TOUCH_TTS_LLM_H_

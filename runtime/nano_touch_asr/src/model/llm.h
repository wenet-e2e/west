// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#ifndef MODEL_LLM_H_
#define MODEL_LLM_H_

#include <string>
#include <vector>

#include "llama.h"  // NOLINT

namespace wenet {

class LLM {
 public:
  explicit LLM(const std::string& model_path);
  ~LLM();

  void Generate(const llama_batch& prompt_batch, std::string* response);
  void Generate(const std::vector<llama_token>& prompt_tokens,
                std::string* response);
  void Generate(const std::string& prompt, std::string* response);
  void Generate(const std::vector<std::vector<float>>& audio_embd,
                std::string* response);
  void ResetContext();
  void FreeContext();

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
#endif  // MODEL_LLM_H_

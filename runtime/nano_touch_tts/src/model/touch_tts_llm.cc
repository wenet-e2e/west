// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#include "model/touch_tts_llm.h"

#include <string>
#include <vector>

#include "utils/log.h"

namespace wenet {

TouchTtsLlm::TouchTtsLlm(const std::string& model_path)
    : model_path_(model_path) {
  // load dynamic backends
  ggml_backend_load_all();
  const int ngl = 99;
  // initialize the model
  model_params_ = llama_model_default_params();
  model_params_.n_gpu_layers = ngl;
  model_ = llama_model_load_from_file(model_path_.c_str(), model_params_);
  vocab_ = llama_model_get_vocab(model_);

  smpl_params_ = llama_sampler_chain_default_params();
  smpl_params_.no_perf = false;
  smpl_ = llama_sampler_chain_init(smpl_params_);
  if (smpl_ == NULL) {
    LOG(ERROR) << "Failed to create the llama_sampler";
    return;
  }
  // llama_sampler_chain_add(smpl_, llama_sampler_init_temp(0.8f));
  llama_sampler_chain_add(smpl_, llama_sampler_init_top_k(10));
  llama_sampler_chain_add(smpl_, llama_sampler_init_top_p(0.8, 1));
  llama_sampler_chain_add(
      smpl_, llama_sampler_init_penalties(100,     // repeat penalty window
                                          1.4,    // repeat penalty 1.1
                                          0.0,    // disable frequency penalty
                                          0.0));  // disable presence penalty
  llama_sampler_chain_add(smpl_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
//   LOG(INFO) << "bos "<< llama_vocab_bos(vocab_);
//   LOG(INFO) << "eos "<< llama_vocab_eos(vocab_);
//   LOG(INFO) << "eot "<< llama_vocab_eot(vocab_);
}

TouchTtsLlm::~TouchTtsLlm() {
  llama_sampler_free(smpl_);
  llama_model_free(model_);
}

void TouchTtsLlm::Generate(const std::vector<llama_token>& prompt_tokens,
                           std::string* response) {
  const int n_ctx = 256;
  ctx_params_ = llama_context_default_params();
  ctx_params_.n_ctx = n_ctx;
  ctx_params_.n_batch = n_ctx;
  ctx_ = llama_init_from_model(model_, ctx_params_);
  if (ctx_ == NULL) {
    LOG(ERROR) << "Failed to create the llama_context";
    return;
  }
  // prepare a batch for the prompt
  llama_batch batch = llama_batch_get_one(
      const_cast<llama_token*>(prompt_tokens.data()), prompt_tokens.size());
  llama_token new_token_id;
  while (true) {
    // evaluate the current batch with the transformer model
    if (llama_decode(ctx_, batch)) {
      LOG(ERROR) << "Failed to eval the prompt";
      return;
    }
    new_token_id = llama_sampler_sample(smpl_, ctx_, -1);
    // is it an end of generation?
    if (llama_vocab_is_eog(vocab_, new_token_id)) {
      break;
    }
    char buf[128];
    int n =
        llama_token_to_piece(vocab_, new_token_id, buf, sizeof(buf), 0, true);
    if (n < 0) {
      LOG(ERROR) << "Failed to convert token to piece";
      return;
    }
    std::string s(buf, n);
    *response += s;
    // prepare the next batch with the sampled token
    batch = llama_batch_get_one(&new_token_id, 1);
  }
  llama_free(ctx_);
}

void TouchTtsLlm::Generate(const std::string& prompt, std::string* response) {
  std::vector<llama_chat_message> messages;
  std::vector<char> formatted(llama_n_ctx(ctx_));
  const char* tmpl = llama_model_chat_template(model_, /* name */ nullptr);
  messages.push_back({"user", strdup(prompt.c_str())});
  int new_len =
      llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                formatted.data(), formatted.size());
  if (new_len < 0) {
    LOG(ERROR) << "Failed to apply the chat template";
    return;
  }
  std::string prompt_formatted(formatted.begin(), formatted.begin() + new_len);
  const int n_prompt =
      -llama_tokenize(vocab_, prompt_formatted.c_str(), prompt_formatted.size(),
                      NULL, 0, true, true);
  std::vector<llama_token> prompt_tokens(n_prompt);
  if (llama_tokenize(vocab_, prompt_formatted.c_str(), prompt_formatted.size(),
                     prompt_tokens.data(), prompt_tokens.size(), true,
                     true) < 0) {
    LOG(ERROR) << "Failed to tokenize the prompt_formatted";
    return;
  }
  Generate(prompt_tokens, response);
}

void TouchTtsLlm::Generate(const std::vector<int32_t>& prompt_speech,
                           const std::string& prompt_text,
                           const std::string& syn_text,  // text to synthesize
                           std::string* response_speech) {
  std::string prompt = prompt_text + syn_text + "<|audio_bos|>";
  LOG(INFO) << "Prompt: " << prompt;
  const int n_prompt = -llama_tokenize(vocab_, prompt.c_str(), prompt.size(),
                                       NULL, 0, true, true);
  std::vector<llama_token> prompt_tokens(n_prompt);
  if (llama_tokenize(vocab_, prompt.c_str(), prompt.size(),
                     prompt_tokens.data(), prompt_tokens.size(), true,
                     true) < 0) {
    LOG(ERROR) << "Failed to tokenize the prompt";
    return;
  }
  prompt_tokens.insert(prompt_tokens.begin(), 151644);
  const int kSpeechStartToken = 151670;
  prompt_tokens.reserve(prompt_tokens.size() + prompt_speech.size());
  for (int i = 0; i < prompt_speech.size(); i++) {
    prompt_tokens.push_back(prompt_speech[i] + kSpeechStartToken);
  }
  Generate(prompt_tokens, response_speech);
}

}  // namespace wenet

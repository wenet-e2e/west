// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#include "model/llm.h"

#include <string>
#include <vector>

#include "utils/log.h"
#include "utils/timer.h"

namespace wenet {

LLM::LLM(const std::string& model_path) : model_path_(model_path) {
  // load dynamic backends
  ggml_backend_load_all();
  const int ngl = 99;
  // initialize the model
  model_params_ = llama_model_default_params();
  model_params_.n_gpu_layers = ngl;
  model_ = llama_model_load_from_file(model_path_.c_str(), model_params_);
  vocab_ = llama_model_get_vocab(model_);

  smpl_params_ = llama_sampler_chain_default_params();
  smpl_ = llama_sampler_chain_init(smpl_params_);
  if (smpl_ == NULL) {
    LOG(ERROR) << "Failed to create the llama_sampler";
    return;
  }
  llama_sampler_chain_add(smpl_, llama_sampler_init_top_k(1));
  llama_sampler_chain_add(smpl_, llama_sampler_init_greedy());
}

LLM::~LLM() {
  llama_sampler_free(smpl_);
  llama_model_free(model_);
}

void LLM::ResetContext() {
  // TODO(Binbin Zhang): Fixme , currently it's big enough
  const int n_ctx = 4096;
  ctx_params_ = llama_context_default_params();
  ctx_params_.n_ctx = n_ctx;
  ctx_params_.n_batch = 256;
  ctx_ = llama_init_from_model(model_, ctx_params_);
  if (ctx_ == NULL) {
    LOG(ERROR) << "Failed to create the llama_context";
    return;
  }
}

void LLM::FreeContext() { llama_free(ctx_); }

void LLM::Generate(const llama_batch& prompt_batch, std::string* response) {
  response->clear();
  llama_batch batch = prompt_batch;
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
}

void LLM::Generate(const std::string& prompt, std::string* response) {
  std::string prompt_formatted = "<|im_start|>user\n" + prompt +
                                 "/no_think"
                                 "<|im_end|>\n" +
                                 "<|im_start|>assistant\n";
  LOG(INFO) << "Prompt formatted: " << prompt_formatted;
  const int n_prompt =
      -llama_tokenize(vocab_, prompt_formatted.c_str(), prompt_formatted.size(),
                      NULL, 0, true, true);
  std::vector<llama_token> prompt_tokens(n_prompt);
  if (llama_tokenize(vocab_, prompt_formatted.c_str(), prompt_formatted.size(),
                     prompt_tokens.data(), prompt_tokens.size(), true,
                     true) < 0) {
    LOG(ERROR) << "Failed to tokenize the prompt_formatted";
    return;
  }  // prepare a batch for the prompt
  Generate(prompt_tokens, response);
}

void LLM::Generate(const std::vector<llama_token>& prompt_tokens,
                   std::string* response) {
  llama_batch batch = llama_batch_get_one(
      const_cast<llama_token*>(prompt_tokens.data()), prompt_tokens.size());
  ResetContext();
  Generate(batch, response);
  FreeContext();
}

void LLM::Generate(const std::vector<std::vector<float>>& audio_embd,
                   std::string* response) {
  std::string prompt_before =
      std::string("<|im_start|>user\n") + "Transcribe the Speech<|audio_bos|>";
  std::string prompt_after =
      "<|audio_eos|><|im_end|>\n<|im_start|>assistant\n";  // /no_think
  const int n_prompt_before = -llama_tokenize(
      vocab_, prompt_before.c_str(), prompt_before.size(), NULL, 0, true, true);
  std::vector<llama_token> prompt_tokens_before(n_prompt_before);
  if (llama_tokenize(vocab_, prompt_before.c_str(), prompt_before.size(),
                     prompt_tokens_before.data(), prompt_tokens_before.size(),
                     true, true) < 0) {
    LOG(ERROR) << "Failed to tokenize the prompt_before";
    return;
  }
  const int n_prompt_after = -llama_tokenize(
      vocab_, prompt_after.c_str(), prompt_after.size(), NULL, 0, true, true);
  std::vector<llama_token> prompt_tokens_after(n_prompt_after);
  if (llama_tokenize(vocab_, prompt_after.c_str(), prompt_after.size(),
                     prompt_tokens_after.data(), prompt_tokens_after.size(),
                     true, true) < 0) {
    LOG(ERROR) << "Failed to tokenize the prompt_after";
    return;
  }

  ResetContext();

  // int total_len = n_prompt_before + audio_embd.size() + n_prompt_after;
  llama_batch batch_before = llama_batch_init(n_prompt_before, 0, 1);
  batch_before.n_tokens = prompt_tokens_before.size();
  std::stringstream ss;
  for (int i = 0; i < prompt_tokens_before.size(); i++) {
    int pos = i;
    batch_before.token[i] = prompt_tokens_before[i];
    batch_before.pos[i] = pos;
    batch_before.n_seq_id[i] = 1;
    batch_before.seq_id[i][0] = 0;
    batch_before.logits[i] = 0;
    ss << prompt_tokens_before[i] << " ";
  }
  LOG(INFO) << "Prompt tokens before: " << ss.str();
  LOG(INFO) << "Decoding prompt tokens before";
  Timer timer_before;
  llama_decode(ctx_, batch_before);
  LOG(INFO) << "Decode prompt_before took " << timer_before.Elapsed() << " ms";
  llama_batch_free(batch_before);

  int n_embd = llama_model_n_embd(model_);
  LOG(INFO) << "n_embd: " << n_embd
            << " audio_embd.size(): " << audio_embd.size();
  llama_batch batch_audio = llama_batch_init(audio_embd.size(), n_embd, 1);
  batch_audio.n_tokens = audio_embd.size();
  for (int i = 0; i < audio_embd.size(); i++) {
    int pos = n_prompt_before + i;
    const float* emb = audio_embd[i].data();
    float* dest = batch_audio.embd + (i * n_embd);
    memcpy(dest, emb, n_embd * sizeof(float));
    batch_audio.pos[i] = pos;
    batch_audio.n_seq_id[i] = 1;
    batch_audio.seq_id[i][0] = 0;
    batch_audio.logits[i] = 0;
  }
  LOG(INFO) << "Decoding audio embeddings";
  Timer timer_audio;
  llama_decode(ctx_, batch_audio);
  LOG(INFO) << "Decode audio_embd took " << timer_audio.Elapsed() << " ms";
  llama_batch_free(batch_audio);

  llama_batch batch_after = llama_batch_init(n_prompt_after, 0, 1);
  batch_after.n_tokens = prompt_tokens_after.size();
  ss.str("");
  ss.clear();
  for (int i = 0; i < prompt_tokens_after.size(); i++) {
    int pos = n_prompt_before + audio_embd.size() + i;
    batch_after.token[i] = prompt_tokens_after[i];
    batch_after.pos[i] = pos;
    batch_after.n_seq_id[i] = 1;
    batch_after.seq_id[i][0] = 0;
    batch_after.logits[i] = 0;
    ss << prompt_tokens_after[i] << " ";
  }
  batch_after.logits[prompt_tokens_after.size() - 1] = 1;
  LOG(INFO) << "Decoding prompt tokens after";
  LOG(INFO) << "Prompt tokens after: " << ss.str();
  Timer timer_after;
  Generate(batch_after, response);
  LOG(INFO) << "Generate after took " << timer_after.Elapsed() << " ms";
  llama_batch_free(batch_after);
  FreeContext();
}

}  // namespace wenet

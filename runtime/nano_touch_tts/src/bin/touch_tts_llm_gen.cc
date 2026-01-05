// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#include <iostream>
#include <string>
#include <vector>

#include "gflags/gflags.h"

#include "model/s3tokenizer.h"
#include "model/touch_tts_llm.h"
#include "utils/log.h"

DEFINE_string(llm_model_path, "", "Model path");
DEFINE_string(wav_file, "", "WAV file to compute features");
DEFINE_string(prompt_text, "", "Prompt text");
DEFINE_string(syn_text, "", "Text to synthesize");
DEFINE_string(s3_model_path, "", "S3 model path");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  wenet::TouchTtsLlm touch_tts_llm(FLAGS_llm_model_path);
  std::string response;
  // llm_model_path & prompt text are required
  CHECK(!FLAGS_llm_model_path.empty()) << "LLM model path is required";
  CHECK(!FLAGS_prompt_text.empty()) << "Prompt text is required";
  if (FLAGS_wav_file.empty() || FLAGS_syn_text.empty()) {
    // Norm text LLM generation
    touch_tts_llm.Generate(FLAGS_prompt_text, &response);
  } else {
    // Speech LLM zero-shot generation
    std::vector<int32_t> prompt_speech;
    wenet::S3Tokenizer s3_tokenizer(FLAGS_s3_model_path);
    s3_tokenizer.Tokenize(FLAGS_wav_file, &prompt_speech);
    touch_tts_llm.Generate(prompt_speech, FLAGS_prompt_text, FLAGS_syn_text,
                           &response);
  }
  LOG(INFO) << "Response: " << response;
  return 0;
}

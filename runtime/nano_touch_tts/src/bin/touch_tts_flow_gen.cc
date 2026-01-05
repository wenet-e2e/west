// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gflags/gflags.h"

#include "model/s3tokenizer.h"
#include "model/touch_tts_flow.h"
#include "model/speaker_model.h"
#include "frontend/fbank.h"
#include "frontend/resample.h"
#include "frontend/wav.h"
#include "utils/log.h"

DEFINE_string(flow_model_path, "", "Model path");
DEFINE_string(speaker_model_path, "", "Speaker model path");
DEFINE_string(s3_model_path, "", "S3 model path");
DEFINE_string(prompt_wav_file, "", "Prompt WAV file");
DEFINE_string(llm_tokens_file, "", "LLM tokens file");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  wenet::TouchTtsFlow touch_tts_flow(FLAGS_flow_model_path);

  wenet::WavReader wav_reader(FLAGS_prompt_wav_file);
  CHECK_EQ(wav_reader.num_channel(), 1) << "Only support mono audio";
  CHECK_EQ(wav_reader.bits_per_sample(), 16) << "Only support 16 bits audio";
  std::vector<float> wave(wav_reader.data(),
                          wav_reader.data() + wav_reader.num_samples());
  if (wav_reader.sample_rate() != 22050) {
    std::vector<float> wave_resampled;
    wenet::Resample(wave, wav_reader.sample_rate(), 22050, &wave_resampled);
    wave = wave_resampled;
  }
  std::vector<std::vector<float>> prompt_feats;  // YES
  wenet::LogMelSpectrogramVocoder fbank;
  fbank.Compute(wave, &prompt_feats);
  wenet::SpeakerModel speaker_model(FLAGS_speaker_model_path);
  std::vector<float> prompt_spk_emb;  // YES
  speaker_model.ExtractEmbedding(FLAGS_prompt_wav_file, &prompt_spk_emb);
  std::vector<int32_t> prompt_tokens;
  wenet::S3Tokenizer s3_tokenizer(FLAGS_s3_model_path);
  s3_tokenizer.Tokenize(FLAGS_prompt_wav_file, &prompt_tokens);  // YES
  std::vector<int32_t> llm_tokens;  // YES
  std::ifstream llm_tokens_file(FLAGS_llm_tokens_file);
  std::string line;
  std::getline(llm_tokens_file, line);
  std::istringstream iss(line);
  int32_t token;
  while (iss >> token) {
    llm_tokens.push_back(token);
  }
  std::vector<std::vector<float>> gen_feats;
  touch_tts_flow.Forward(prompt_feats, prompt_spk_emb, prompt_tokens,
                         llm_tokens, &gen_feats);
  LOG(INFO) << "Gen feats: " << gen_feats.size();
  for (const auto& feat : gen_feats) {
    for (const auto& f : feat) {
      std::cout << f << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}

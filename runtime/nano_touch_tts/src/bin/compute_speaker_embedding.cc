// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#include <iostream>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "model/speaker_model.h"

DEFINE_string(model_path, "", "Model path");
DEFINE_string(wav_file, "", "WAV file to compute features");


int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  wenet::SpeakerModel speaker_model(FLAGS_model_path);
  std::vector<float> embed;
  speaker_model.ExtractEmbedding(FLAGS_wav_file, &embed);
  for (const auto& e : embed) {
    std::cout << e << " ";
  }
  std::cout << std::endl;
  return 0;
}

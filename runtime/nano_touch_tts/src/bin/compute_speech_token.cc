// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#include <iostream>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "model/s3tokenizer.h"

DEFINE_string(model_path, "", "Model path");
DEFINE_string(wav_file, "", "WAV file to compute features");


int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  wenet::S3Tokenizer s3_tokenizer(FLAGS_model_path);
  std::vector<int32_t> tokens;
  s3_tokenizer.Tokenize(FLAGS_wav_file, &tokens);
  for (const auto& token : tokens) {
    std::cout << token << " ";
  }
  std::cout << std::endl;
  return 0;
}

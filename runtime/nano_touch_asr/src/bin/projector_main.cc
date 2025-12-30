// Copyright (c) 2025 Personal (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "model/projector.h"

DEFINE_string(projector_model, "", "Projector model path");
DEFINE_int32(num_frames, 10, "Number of input frames");
DEFINE_int32(input_dim, 256, "Input dimension");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(!FLAGS_projector_model.empty()) << "Projector model is required";

  // 1. Load projector model
  wenet::Projector projector(FLAGS_projector_model);
  LOG(INFO) << "Loaded projector model from " << FLAGS_projector_model;

  // 2. Construct random test input
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<std::vector<float>> input(FLAGS_num_frames,
                                        std::vector<float>(FLAGS_input_dim));
  for (int i = 0; i < FLAGS_num_frames; ++i) {
    for (int j = 0; j < FLAGS_input_dim; ++j) {
      input[i][j] = dist(gen);
    }
  }
  LOG(INFO) << "Constructed random input: " << FLAGS_num_frames << " frames, "
            << FLAGS_input_dim << " dim";

  // 3. Run forward
  std::vector<std::vector<float>> output;
  projector.Forward(input, &output);

  // 4. Print output info
  LOG(INFO) << "Output: " << output.size() << " frames, "
            << (output.empty() ? 0 : output[0].size()) << " dim";

  // Print first few values of first frame
  if (!output.empty() && !output[0].empty()) {
    std::cout << "First frame output (first 10 values): ";
    int print_size = std::min(10, static_cast<int>(output[0].size()));
    for (int i = 0; i < print_size; ++i) {
      std::cout << output[0][i] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}

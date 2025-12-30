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

#include <iostream>
#include <string>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "model/llm.h"

DEFINE_string(llm_model, "", "LLM model path");
DEFINE_string(prompt, "", "Input prompt for generation");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(!FLAGS_llm_model.empty()) << "LLM model is required";
  CHECK(!FLAGS_prompt.empty()) << "Prompt is required";

  // 1. Load LLM model
  wenet::LLM llm(FLAGS_llm_model);
  LOG(INFO) << "Loaded LLM model from " << FLAGS_llm_model;

  // 2. Generate response
  std::string response;
  LOG(INFO) << "Generating response for prompt: " << FLAGS_prompt;
  llm.Generate(FLAGS_prompt, &response);

  // 3. Output result
  std::cout << "Prompt: " << FLAGS_prompt << std::endl;
  std::cout << "Response: " << response << std::endl;

  return 0;
}

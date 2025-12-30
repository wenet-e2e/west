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

#ifndef MODEL_CTC_TOKENS_H_
#define MODEL_CTC_TOKENS_H_

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "utils/log.h"

namespace wenet {

class CtcTokens {
 public:
  explicit CtcTokens(const std::string& tokens_file) {
    std::ifstream fin(tokens_file);
    CHECK(fin.is_open()) << "Failed to open tokens file: " << tokens_file;

    std::string line;
    while (std::getline(fin, line)) {
      std::istringstream iss(line);
      std::string token;
      int index;
      if (iss >> token >> index) {
        idx2token_[index] = token;
      }
    }
    LOG(INFO) << "Loaded " << idx2token_.size() << " tokens from "
              << tokens_file;
  }

  std::string GetToken(int idx) const {
    auto it = idx2token_.find(idx);
    if (it != idx2token_.end()) {
      return it->second;
    }
    return "<unk>";
  }

  size_t size() const { return idx2token_.size(); }

 private:
  std::unordered_map<int, std::string> idx2token_;
};

}  // namespace wenet

#endif  // MODEL_CTC_TOKENS_H_

// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)
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

#ifndef FRONTEND_RESAMPLE_H_
#define FRONTEND_RESAMPLE_H_

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace wenet {

// Linear interpolation resampling
// Simple but fast, suitable for most use cases
inline void LinearResample(const float* input, int input_size,
                           int input_sample_rate, int output_sample_rate,
                           std::vector<float>* output) {
  if (input_sample_rate == output_sample_rate) {
    output->assign(input, input + input_size);
    return;
  }

  double ratio = static_cast<double>(input_sample_rate) / output_sample_rate;
  int output_size = static_cast<int>(std::ceil(input_size / ratio));
  output->resize(output_size);

  for (int i = 0; i < output_size; ++i) {
    double src_idx = i * ratio;
    int idx0 = static_cast<int>(src_idx);
    int idx1 = idx0 + 1;
    double frac = src_idx - idx0;

    if (idx1 >= input_size) {
      (*output)[i] = input[input_size - 1];
    } else {
      (*output)[i] =
          static_cast<float>((1.0 - frac) * input[idx0] + frac * input[idx1]);
    }
  }
}

// Sinc function for high-quality resampling
inline double Sinc(double x) {
  if (std::abs(x) < 1e-10) {
    return 1.0;
  }
  return std::sin(M_PI * x) / (M_PI * x);
}

// Lanczos window function
inline double LanczosWindow(double x, int a) {
  if (std::abs(x) < 1e-10) {
    return 1.0;
  }
  if (std::abs(x) >= a) {
    return 0.0;
  }
  return Sinc(x) * Sinc(x / a);
}

// High-quality Lanczos resampling
// Better quality but slower, use when audio quality is critical
inline void LanczosResample(const float* input, int input_size,
                            int input_sample_rate, int output_sample_rate,
                            std::vector<float>* output, int filter_size = 3) {
  if (input_sample_rate == output_sample_rate) {
    output->assign(input, input + input_size);
    return;
  }

  double ratio = static_cast<double>(input_sample_rate) / output_sample_rate;
  int output_size = static_cast<int>(std::ceil(input_size / ratio));
  output->resize(output_size);

  for (int i = 0; i < output_size; ++i) {
    double src_idx = i * ratio;
    int center = static_cast<int>(std::round(src_idx));

    double sum = 0.0;
    double weight_sum = 0.0;

    for (int j = center - filter_size + 1; j <= center + filter_size; ++j) {
      if (j >= 0 && j < input_size) {
        double weight = LanczosWindow(src_idx - j, filter_size);
        sum += input[j] * weight;
        weight_sum += weight;
      }
    }

    if (weight_sum > 0.0) {
      (*output)[i] = static_cast<float>(sum / weight_sum);
    } else {
      (*output)[i] = 0.0f;
    }
  }
}

// Convenient wrapper function using std::vector
inline void Resample(const std::vector<float>& input, int input_sample_rate,
                     int output_sample_rate, std::vector<float>* output,
                     bool high_quality = false) {
  if (high_quality) {
    LanczosResample(input.data(), input.size(), input_sample_rate,
                    output_sample_rate, output);
  } else {
    LinearResample(input.data(), input.size(), input_sample_rate,
                   output_sample_rate, output);
  }
}

}  // namespace wenet

#endif  // FRONTEND_RESAMPLE_H_

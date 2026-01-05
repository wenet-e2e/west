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
#include <vector>

#include "frontend/fbank.h"
#include "frontend/resample.h"
#include "frontend/wav.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(wav_file, "", "WAV file to compute features");
DEFINE_int32(num_bins, 128, "Number of bins");
DEFINE_int32(sample_rate, 16000, "Sample rate");
DEFINE_int32(frame_length, 400, "Frame length");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  wenet::LogMelSpectrogramVocoder fbank;
  wenet::WavReader wav_reader(FLAGS_wav_file);
  CHECK_EQ(wav_reader.num_channel(), 1) << "Only support mono audio";
  CHECK_EQ(wav_reader.bits_per_sample(), 16) << "Only support 16 bits audio";
  std::vector<float> wave(wav_reader.data(),
                          wav_reader.data() + wav_reader.num_samples());
  if (wav_reader.sample_rate() != 22050) {
    std::vector<float> wave_resampled;
    wenet::Resample(wave, wav_reader.sample_rate(), 22050, &wave_resampled);
    wave = wave_resampled;
  }
  std::vector<std::vector<float>> feat;
  fbank.Compute(wave, &feat);
  std::cout << "feat size: " << feat.size() << " " << feat[0].size()
            << std::endl;
  for (const auto& frame : feat) {
    for (int i = 0; i < 3; i++) {
      std::cout << frame[i] << " ";
    }
    std::cout << " ... ";
    for (int i = 3; i > 0; i--) {
      std::cout << frame[frame.size() - i] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}

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
#include <vector>

#include "frontend/resample.h"
#include "frontend/wav.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "model/touch_asr.h"

DEFINE_string(speech_encoder_model, "", "Speech encoder model path");
DEFINE_string(ctc_model, "", "CTC model path");
DEFINE_string(ctc_tokens_file, "", "CTC tokens file path");
DEFINE_string(projector_model, "", "Projector model path");
DEFINE_string(llm_model, "", "LLM model path");
DEFINE_string(wav_file, "", "WAV file to decode");
DEFINE_int32(num_bins, 80, "Number of mel bins");
DEFINE_int32(sample_rate, 16000, "Sample rate");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(!FLAGS_speech_encoder_model.empty())
      << "Speech encoder model is required";
  CHECK(!FLAGS_ctc_model.empty()) << "CTC model is required";
  CHECK(!FLAGS_ctc_tokens_file.empty()) << "CTC tokens file is required";
  CHECK(!FLAGS_projector_model.empty()) << "Projector model is required";
  CHECK(!FLAGS_llm_model.empty()) << "LLM model is required";
  CHECK(!FLAGS_wav_file.empty()) << "WAV file is required";

  // 1. Read WAV file
  wenet::WavReader wav_reader(FLAGS_wav_file);
  CHECK_EQ(wav_reader.num_channel(), 1) << "Only support mono audio";
  CHECK_EQ(wav_reader.bits_per_sample(), 16) << "Only support 16 bits audio";
  std::vector<float> wave(wav_reader.data(),
                          wav_reader.data() + wav_reader.num_samples());
  LOG(INFO) << "Read WAV file: " << FLAGS_wav_file
            << ", samples: " << wave.size()
            << ", sample_rate: " << wav_reader.sample_rate();

  // Resample if necessary
  if (wav_reader.sample_rate() != FLAGS_sample_rate) {
    std::vector<float> wave_resampled;
    wenet::Resample(wave, wav_reader.sample_rate(), FLAGS_sample_rate,
                    &wave_resampled);
    wave = wave_resampled;
    LOG(INFO) << "Resampled from " << wav_reader.sample_rate() << " to "
              << FLAGS_sample_rate;
  }

  // 2. Initialize TouchASR
  wenet::TouchASRConfig config;
  config.speech_encoder_model = FLAGS_speech_encoder_model;
  config.ctc_model = FLAGS_ctc_model;
  config.ctc_tokens_file = FLAGS_ctc_tokens_file;
  config.projector_model = FLAGS_projector_model;
  config.llm_model = FLAGS_llm_model;
  config.num_bins = FLAGS_num_bins;
  config.sample_rate = FLAGS_sample_rate;

  wenet::TouchASR asr(config);
  LOG(INFO) << "Initialized TouchASR";

  // 3. Decode
  asr.AcceptWaveform(wave);
  asr.SetInputFinished();
  std::string result;
  while (true) {
    bool finish = asr.DecodeStreaming(&result);
    LOG(INFO) << "Stream(CTC) result: " << result;
    if (finish) {
      break;
    }
  }
  asr.DecodeNonStreaming(&result);
  LOG(INFO) << "Non-streaming(LLM) result: " << result;
  return 0;
}

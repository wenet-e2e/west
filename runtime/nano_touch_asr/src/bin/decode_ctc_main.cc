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
#include <vector>

#include "frontend/feature_pipeline.h"
#include "frontend/resample.h"
#include "frontend/wav.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "model/ctc_model.h"
#include "model/ctc_tokens.h"
#include "model/speech_encoder.h"

DEFINE_string(speech_encoder_model, "", "Speech encoder model path");
DEFINE_string(ctc_model, "", "CTC model path");
DEFINE_string(ctc_tokens_file, "", "CTC tokens file path");
DEFINE_string(wav_file, "", "WAV file to decode");
DEFINE_int32(num_bins, 80, "Number of mel bins");
DEFINE_int32(sample_rate, 16000, "Sample rate");
DEFINE_int32(chunk_size, 4, "Chunk size in frames");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(!FLAGS_speech_encoder_model.empty())
      << "Speech encoder model is required";
  CHECK(!FLAGS_ctc_model.empty()) << "CTC model is required";
  CHECK(!FLAGS_ctc_tokens_file.empty()) << "CTC tokens file is required";
  CHECK(!FLAGS_wav_file.empty()) << "WAV file is required";

  // 1. Read WAV file
  wenet::WavReader wav_reader(FLAGS_wav_file);
  CHECK_EQ(wav_reader.num_channel(), 1) << "Only support mono audio";
  CHECK_EQ(wav_reader.bits_per_sample(), 16) << "Only support 16 bits audio";
  std::vector<float> wave(wav_reader.data(),
                          wav_reader.data() + wav_reader.num_samples());

  // Resample if necessary
  if (wav_reader.sample_rate() != FLAGS_sample_rate) {
    std::vector<float> wave_resampled;
    wenet::Resample(wave, wav_reader.sample_rate(), FLAGS_sample_rate,
                    &wave_resampled);
    wave = wave_resampled;
    LOG(INFO) << "Resampled from " << wav_reader.sample_rate() << " to "
              << FLAGS_sample_rate;
  }

  // 2. Compute fbank features using FeaturePipeline
  wenet::FeaturePipelineConfig config(FLAGS_num_bins, FLAGS_sample_rate);
  config.Info();
  wenet::FeaturePipeline feature_pipeline(config);
  feature_pipeline.AcceptWaveform(wave.data(), wave.size());
  feature_pipeline.set_input_finished();

  std::vector<std::vector<float>> feats;
  feature_pipeline.Read(feature_pipeline.num_frames(), &feats);
  LOG(INFO) << "Computed " << feats.size()
            << " frames, dim = " << (feats.empty() ? 0 : feats[0].size());

  // 3. Initialize models
  wenet::SpeechEncoder speech_encoder(FLAGS_speech_encoder_model);
  LOG(INFO) << "Initialized SpeechEncoder";
  wenet::CtcModel ctc_model(FLAGS_ctc_model);
  LOG(INFO) << "Initialized CtcModel";
  wenet::CtcTokens ctc_tokens(FLAGS_ctc_tokens_file);
  LOG(INFO) << "Initialized CtcTokens";

  // 4. Process in chunks
  speech_encoder.Reset();
  std::vector<std::vector<float>> all_encoder_outs;

  int num_frames = feats.size();
  int feature_dim = feats.empty() ? 0 : feats[0].size();

  int i = 0;
  int chunk_idx = 0;
  while (i < num_frames) {
    int current_chunk_size = speech_encoder.NumFramesForThisChunk(chunk_idx);
    int chunk_end = std::min(i + current_chunk_size, num_frames);
    std::vector<std::vector<float>> chunk_feats(feats.begin() + i,
                                                feats.begin() + chunk_end);

    // Pad to chunk size if necessary (for non-first chunks)
    if (chunk_idx > 0 && chunk_feats.size() < current_chunk_size) {
      int pad_frames = current_chunk_size - chunk_feats.size();
      std::vector<float> zero_frame(feature_dim, 0.0f);
      for (int p = 0; p < pad_frames; ++p) {
        chunk_feats.push_back(zero_frame);
      }
      LOG(INFO) << "Padded " << pad_frames << " zero frames to last chunk";
    }

    std::vector<std::vector<float>> encoder_outs;
    speech_encoder.Forward(chunk_feats, &encoder_outs);
    std::vector<int> ctc_ids;
    std::stringstream ss;
    ctc_model.Forward(encoder_outs, &ctc_ids);
    for (int id : ctc_ids) {
      ss << ctc_tokens.GetToken(id);
    }
    LOG(INFO) << "Forwarding chunk " << chunk_idx << " with size "
              << chunk_feats.size() << " result: " << ss.str();
    i += current_chunk_size;
    chunk_idx++;
  }
  return 0;
}

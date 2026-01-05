// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "frontend/fbank.h"
#include "frontend/resample.h"
#include "frontend/wav.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "model/hifi_gan.h"

DEFINE_string(hifigan_model, "", "HiFi-GAN model path");
DEFINE_string(mel_file, "", "Input mel spectrogram text file");
DEFINE_string(input_wav, "", "Input WAV file to extract mel from");
DEFINE_string(output_wav, "", "Output WAV file");
DEFINE_int32(sample_rate, 22050, "Output sample rate");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(!FLAGS_hifigan_model.empty()) << "HiFi-GAN model path is required";
  CHECK(!FLAGS_output_wav.empty()) << "Output WAV file is required";
  CHECK(!FLAGS_mel_file.empty() || !FLAGS_input_wav.empty())
      << "Either mel_file or input_wav is required";

  std::vector<std::vector<float>> mel;

  if (!FLAGS_input_wav.empty()) {
    // Extract mel from WAV file
    wenet::WavReader wav_reader(FLAGS_input_wav);
    CHECK_EQ(wav_reader.num_channel(), 1) << "Only support mono audio";
    CHECK_EQ(wav_reader.bits_per_sample(), 16) << "Only support 16 bits audio";
    std::vector<float> wave(wav_reader.data(),
                            wav_reader.data() + wav_reader.num_samples());
    if (wav_reader.sample_rate() != 22050) {
      std::vector<float> wave_resampled;
      wenet::Resample(wave, wav_reader.sample_rate(), 22050, &wave_resampled);
      wave = wave_resampled;
    }
    wenet::LogMelSpectrogramVocoder fbank;
    fbank.Compute(wave, &mel);
    LOG(INFO) << "Extracted mel from WAV: " << mel.size() << " frames, "
              << (mel.empty() ? 0 : mel[0].size()) << " bins";
  } else {
    // Load mel spectrogram from text file
    // Format: each line is a frame, values separated by spaces
    std::ifstream mel_file(FLAGS_mel_file);
    CHECK(mel_file.is_open()) << "Failed to open mel file: " << FLAGS_mel_file;

    std::string line;
    while (std::getline(mel_file, line)) {
      if (line.empty()) continue;
      std::vector<float> frame;
      std::istringstream iss(line);
      float value;
      while (iss >> value) {
        frame.push_back(value);
      }
      if (!frame.empty()) {
        mel.push_back(frame);
      }
    }
    mel_file.close();
    LOG(INFO) << "Loaded mel from file: " << mel.size() << " frames, "
              << (mel.empty() ? 0 : mel[0].size()) << " bins";
  }

  // Initialize HiFi-GAN model
  wenet::HifiGan hifigan(FLAGS_hifigan_model);

  // Generate audio
  std::vector<float> audio;
  hifigan.Forward(mel, &audio);
  LOG(INFO) << "Generated audio: " << audio.size() << " samples";
  // Write output WAV file
  wenet::WavWriter wav_writer(audio.data(), audio.size(), 1, FLAGS_sample_rate,
                              16);
  wav_writer.Write(FLAGS_output_wav);
  LOG(INFO) << "Saved audio to: " << FLAGS_output_wav;

  return 0;
}

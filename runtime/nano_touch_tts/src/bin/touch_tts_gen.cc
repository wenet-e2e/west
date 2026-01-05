// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "frontend/fbank.h"
#include "frontend/resample.h"
#include "frontend/wav.h"
#include "gflags/gflags.h"
#include "model/hifi_gan.h"
#include "model/s3tokenizer.h"
#include "model/speaker_model.h"
#include "model/touch_tts_flow.h"
#include "model/touch_tts_llm.h"
#include "utils/log.h"
#include "utils/timer.h"

DEFINE_string(llm_model_path, "", "LLM model path");
DEFINE_string(flow_model_path, "", "Flow model path");
DEFINE_string(hifigan_model_path, "", "HiFi-GAN model path");
DEFINE_string(s3_model_path, "", "S3 tokenizer model path");
DEFINE_string(speaker_model_path, "", "Speaker model path");
DEFINE_string(prompt_wav, "", "Prompt WAV file");
DEFINE_string(prompt_text, "", "Prompt text (transcription of prompt wav)");
DEFINE_string(syn_text_file, "",
              "File containing texts to synthesize (one per line)");
DEFINE_string(output_dir, "", "Output directory for WAV files");
DEFINE_int32(sample_rate, 22050, "Output sample rate");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(!FLAGS_llm_model_path.empty()) << "LLM model path is required";
  CHECK(!FLAGS_flow_model_path.empty()) << "Flow model path is required";
  CHECK(!FLAGS_hifigan_model_path.empty()) << "HiFi-GAN model path is required";
  CHECK(!FLAGS_s3_model_path.empty()) << "S3 model path is required";
  CHECK(!FLAGS_speaker_model_path.empty()) << "Speaker model path is required";
  CHECK(!FLAGS_prompt_wav.empty()) << "Prompt WAV is required";
  CHECK(!FLAGS_prompt_text.empty()) << "Prompt text is required";
  CHECK(!FLAGS_syn_text_file.empty()) << "Synthesis text file is required";
  CHECK(!FLAGS_output_dir.empty()) << "Output directory is required";

  // Read synthesis texts from file
  std::vector<std::string> syn_texts;
  std::ifstream syn_file(FLAGS_syn_text_file);
  CHECK(syn_file.is_open())
      << "Failed to open syn_text_file: " << FLAGS_syn_text_file;
  std::string line;
  while (std::getline(syn_file, line)) {
    if (!line.empty()) {
      syn_texts.push_back(line);
    }
  }
  syn_file.close();
  LOG(INFO) << "Loaded " << syn_texts.size() << " texts to synthesize";

  wenet::S3Tokenizer s3_tokenizer(FLAGS_s3_model_path);
  wenet::TouchTtsLlm llm(FLAGS_llm_model_path);
  wenet::TouchTtsFlow flow(FLAGS_flow_model_path);
  wenet::HifiGan hifigan(FLAGS_hifigan_model_path);
  wenet::SpeakerModel speaker_model(FLAGS_speaker_model_path);
  LOG(INFO) << "Initialized models okay";

  // Timing statistics (in milliseconds)
  int total_s3_time = 0;
  int total_llm_time = 0;
  int total_flow_time = 0;
  int total_hifigan_time = 0;
  int total_speaker_time = 0;
  int total_preprocess_time = 0;
  int num_texts = syn_texts.size();

  for (size_t i = 0; i < syn_texts.size(); ++i) {
    const std::string& syn_text = syn_texts[i];
    LOG(INFO) << "=== Text " << (i + 1) << "/" << syn_texts.size() << " ===";
    LOG(INFO) << "Synthesizing: " << syn_text;
    wenet::Timer timer;

    // Step 1: Extract speech tokens from prompt wav using S3Tokenizer
    timer.Reset();
    std::vector<int32_t> prompt_speech_tokens;
    s3_tokenizer.Tokenize(FLAGS_prompt_wav, &prompt_speech_tokens);
    total_s3_time += timer.Elapsed();
    LOG(INFO) << "Step 1: S3Tokenizer - " << timer.Elapsed()
              << " ms, tokens: " << prompt_speech_tokens.size();

    // Step 2: Generate LLM tokens using TouchTtsLlm
    timer.Reset();
    std::string llm_response;
    llm.Generate(prompt_speech_tokens, FLAGS_prompt_text, syn_text,
                 &llm_response);
    total_llm_time += timer.Elapsed();
    LOG(INFO) << "Step 2: LLM - " << timer.Elapsed() << " ms";

    // Parse LLM response to tokens
    // Format: <|speech_4218|><|speech_4299|><|speech_2112|>...
    std::vector<int32_t> llm_tokens;
    size_t pos = 0;
    const std::string prefix = "<|speech_";
    const std::string suffix = "|>";
    while ((pos = llm_response.find(prefix, pos)) != std::string::npos) {
      size_t start = pos + prefix.length();
      size_t end = llm_response.find(suffix, start);
      if (end != std::string::npos) {
        std::string num_str = llm_response.substr(start, end - start);
        int32_t token = std::stoi(num_str);
        llm_tokens.push_back(token);
        pos = end + suffix.length();
      } else {
        break;
      }
    }
    LOG(INFO) << "  LLM tokens: " << llm_tokens.size();

    // Step 3: Preprocess - read wav, resample, extract mel
    timer.Reset();
    wenet::WavReader wav_reader(FLAGS_prompt_wav);
    CHECK_EQ(wav_reader.num_channel(), 1) << "Only support mono audio";
    CHECK_EQ(wav_reader.bits_per_sample(), 16) << "Only support 16 bits audio";
    std::vector<float> wave(wav_reader.data(),
                            wav_reader.data() + wav_reader.num_samples());
    if (wav_reader.sample_rate() != 22050) {
      std::vector<float> wave_resampled;
      wenet::Resample(wave, wav_reader.sample_rate(), 22050, &wave_resampled);
      wave = wave_resampled;
    }
    std::vector<std::vector<float>> prompt_feats;
    wenet::LogMelSpectrogramVocoder fbank;
    fbank.Compute(wave, &prompt_feats);
    total_preprocess_time += timer.Elapsed();
    LOG(INFO) << "Step 3: Preprocess - " << timer.Elapsed()
              << " ms, mel frames: " << prompt_feats.size();

    // Step 4: Extract speaker embedding
    timer.Reset();
    std::vector<float> prompt_spk_emb;
    speaker_model.ExtractEmbedding(FLAGS_prompt_wav, &prompt_spk_emb);
    total_speaker_time += timer.Elapsed();
    LOG(INFO) << "Step 4: Speaker - " << timer.Elapsed() << " ms";

    // Step 5: Generate mel features using flow model
    timer.Reset();
    std::vector<std::vector<float>> gen_feats;
    flow.Forward(prompt_feats, prompt_spk_emb, prompt_speech_tokens, llm_tokens,
                 &gen_feats);
    total_flow_time += timer.Elapsed();
    LOG(INFO) << "Step 5: Flow - " << timer.Elapsed()
              << " ms, gen frames: " << gen_feats.size();

    // Step 6: Generate audio using HiFi-GAN
    timer.Reset();
    std::vector<float> audio;
    hifigan.Forward(gen_feats, &audio);
    total_hifigan_time += timer.Elapsed();
    LOG(INFO) << "Step 6: HiFi-GAN - " << timer.Elapsed()
              << " ms, samples: " << audio.size();

    // Write output WAV file
    std::string output_wav =
        FLAGS_output_dir + "/" + std::to_string(i) + ".wav";
    wenet::WavWriter wav_writer(audio.data(), audio.size(), 1,
                                FLAGS_sample_rate, 16);
    wav_writer.Write(output_wav);
    LOG(INFO) << "Saved audio to: " << output_wav;
  }

  // Print timing statistics
  LOG(INFO) << "";
  LOG(INFO) << "========== Timing Statistics (" << num_texts
            << " texts) ==========";
  LOG(INFO) << "S3Tokenizer:  " << total_s3_time << " ms total, "
            << total_s3_time / num_texts << " ms avg";
  LOG(INFO) << "LLM:          " << total_llm_time << " ms total, "
            << total_llm_time / num_texts << " ms avg";
  LOG(INFO) << "Preprocess:   " << total_preprocess_time << " ms total, "
            << total_preprocess_time / num_texts << " ms avg";
  LOG(INFO) << "Speaker:      " << total_speaker_time << " ms total, "
            << total_speaker_time / num_texts << " ms avg";
  LOG(INFO) << "Flow:         " << total_flow_time << " ms total, "
            << total_flow_time / num_texts << " ms avg";
  LOG(INFO) << "HiFi-GAN:     " << total_hifigan_time << " ms total, "
            << total_hifigan_time / num_texts << " ms avg";
  int total_time = total_s3_time + total_llm_time + total_preprocess_time +
                   total_speaker_time + total_flow_time + total_hifigan_time;
  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "Total:        " << total_time << " ms total, "
            << total_time / num_texts << " ms avg";
  LOG(INFO) << "==============================================";

  return 0;
}

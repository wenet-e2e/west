// Copyright (c) 2017 Personal (Binbin Zhang)
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

#ifndef FRONTEND_FBANK_H_
#define FRONTEND_FBANK_H_

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "frontend/fft.h"
#include "frontend/fft2.h"
#include "utils/log.h"

namespace wenet {

// This code is based on kaldi Fbank implementation, please see
// https://github.com/kaldi-asr/kaldi/blob/master/src/feat/feature-fbank.cc

static const int kS16AbsMax = 1 << 15;

enum class WindowType {
  kPovey = 0,
  kHanning,
};

enum class MelType {
  kHTK = 0,
  kSlaney,
};

enum class NormalizationType {
  kKaldi = 0,
  kWhisper,
};

enum class LogBase {
  kBaseE = 0,
  kBase10,
};

class Fbank {
 public:
  Fbank(int num_bins, int sample_rate, int frame_length, int frame_shift,
        float low_freq = 20, bool pre_emphasis = true,
        bool scale_input_to_unit = false,
        float log_floor = std::numeric_limits<float>::epsilon(),
        LogBase log_base = LogBase::kBaseE,
        WindowType window_type = WindowType::kPovey,
        MelType mel_type = MelType::kHTK,
        NormalizationType norm_type = NormalizationType::kKaldi,
        bool power_of_two = true, bool center_pad = false)
      : num_bins_(num_bins),
        sample_rate_(sample_rate),
        frame_length_(frame_length),
        frame_shift_(frame_shift),
        use_log_(true),
        remove_dc_offset_(true),
        generator_(0),
        distribution_(0, 1.0),
        dither_(0.0),
        low_freq_(low_freq),
        high_freq_(sample_rate / 2),
        pre_emphasis_(pre_emphasis),
        scale_input_to_unit_(scale_input_to_unit),
        log_floor_(log_floor),
        log_base_(log_base),
        norm_type_(norm_type),
        power_of_two_(power_of_two),
        center_pad_(center_pad) {
    if (power_of_two) {
      fft_points_ = UpperPowerOfTwo(frame_length_);
      // generate bit reversal table and trigonometric function table
      const int fft_points_4 = fft_points_ / 4;
      bitrev_.resize(fft_points_);
      sintbl_.resize(fft_points_ + fft_points_4);
      make_sintbl(fft_points_, sintbl_.data());
      make_bitrev(fft_points_, bitrev_.data());
      fft_real_.resize(fft_points_);
      fft_img_.resize(fft_points_);
    } else {
      fft_points_ = frame_length_;
      fft2_in_.resize(fft_points_ * 2);
      fft2_out_.resize(fft_points_ * 2 * 2 * 2);
    }

    InitMelFilters(mel_type);
    InitWindow(window_type);
  }

  void InitMelFilters(MelType mel_type) {
    int num_fft_bins = fft_points_ / 2;
    float fft_bin_width = static_cast<float>(sample_rate_) / fft_points_;
    float mel_low_freq = MelScale(low_freq_, mel_type);
    float mel_high_freq = MelScale(high_freq_, mel_type);
    float mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins_ + 1);
    bins_.resize(num_bins_);
    center_freqs_.resize(num_bins_);

    for (int bin = 0; bin < num_bins_; ++bin) {
      float left_mel = mel_low_freq + bin * mel_freq_delta,
            center_mel = mel_low_freq + (bin + 1) * mel_freq_delta,
            right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;
      center_freqs_[bin] = InverseMelScale(center_mel, mel_type);
      std::vector<float> this_bin(num_fft_bins);
      int first_index = -1, last_index = -1;
      for (int i = 0; i < num_fft_bins; ++i) {
        float freq = (fft_bin_width * i);  // Center frequency of this fft
        // bin.
        float mel = MelScale(freq, mel_type);
        if (mel > left_mel && mel < right_mel) {
          float weight;
          if (mel_type == MelType::kHTK) {
            if (mel <= center_mel)
              weight = (mel - left_mel) / (center_mel - left_mel);
            else if (mel > center_mel)
              weight = (right_mel - mel) / (right_mel - center_mel);
          } else if (mel_type == MelType::kSlaney) {
            if (mel <= center_mel) {
              weight = (InverseMelScale(mel, mel_type) -
                        InverseMelScale(left_mel, mel_type)) /
                       (InverseMelScale(center_mel, mel_type) -
                        InverseMelScale(left_mel, mel_type));
              weight *= 2.0 / (InverseMelScale(right_mel, mel_type) -
                               InverseMelScale(left_mel, mel_type));
            } else if (mel > center_mel) {
              weight = (InverseMelScale(right_mel, mel_type) -
                        InverseMelScale(mel, mel_type)) /
                       (InverseMelScale(right_mel, mel_type) -
                        InverseMelScale(center_mel, mel_type));
              weight *= 2.0 / (InverseMelScale(right_mel, mel_type) -
                               InverseMelScale(left_mel, mel_type));
            }
          }
          this_bin[i] = weight;
          if (first_index == -1) first_index = i;
          last_index = i;
        }
      }
      CHECK(first_index != -1 && last_index >= first_index);
      bins_[bin].first = first_index;
      int size = last_index + 1 - first_index;
      bins_[bin].second.resize(size);
      for (int i = 0; i < size; ++i) {
        bins_[bin].second[i] = this_bin[first_index + i];
      }
    }
  }

  void InitWindow(WindowType window_type) {
    window_.resize(frame_length_);
    if (window_type == WindowType::kPovey) {
      // povey window
      double a = M_2PI / (frame_length_ - 1);
      for (int i = 0; i < frame_length_; ++i)
        window_[i] = pow(0.5 - 0.5 * cos(a * i), 0.85);
    } else if (window_type == WindowType::kHanning) {
      // periodic hanning window
      double a = M_2PI / (frame_length_);
      for (int i = 0; i < frame_length_; ++i)
        window_[i] = 0.5 * (1.0 - cos(i * a));
    }
  }

  void set_use_log(bool use_log) { use_log_ = use_log; }

  void set_remove_dc_offset(bool remove_dc_offset) {
    remove_dc_offset_ = remove_dc_offset;
  }

  void set_dither(float dither) { dither_ = dither; }

  int num_bins() const { return num_bins_; }

  static inline float InverseMelScale(float mel_freq,
                                      MelType mel_type = MelType::kHTK) {
    if (mel_type == MelType::kHTK) {
      return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
    } else if (mel_type == MelType::kSlaney) {
      float f_min = 0.0;
      float f_sp = 200.0f / 3.0f;
      float min_log_hz = 1000.0;
      float freq = f_min + f_sp * mel_freq;
      float min_log_mel = (min_log_hz - f_min) / f_sp;
      float logstep = logf(6.4) / 27.0f;
      if (mel_freq >= min_log_mel) {
        return min_log_hz * expf(logstep * (mel_freq - min_log_mel));
      } else {
        return freq;
      }
    } else {
      throw std::invalid_argument("Unsupported mel type!");
    }
  }

  static inline float MelScale(float freq, MelType mel_type = MelType::kHTK) {
    if (mel_type == MelType::kHTK) {
      return 1127.0f * logf(1.0f + freq / 700.0f);
    } else if (mel_type == MelType::kSlaney) {
      float f_min = 0.0;
      float f_sp = 200.0f / 3.0f;
      float min_log_hz = 1000.0;
      float mel = (freq - f_min) / f_sp;
      float min_log_mel = (min_log_hz - f_min) / f_sp;
      float logstep = logf(6.4) / 27.0f;
      if (freq >= min_log_hz) {
        return min_log_mel + logf(freq / min_log_hz) / logstep;
      } else {
        return mel;
      }
    } else {
      throw std::invalid_argument("Unsupported mel type!");
    }
  }

  static int UpperPowerOfTwo(int n) {
    return static_cast<int>(pow(2, ceil(log(n) / log(2))));
  }

  // pre emphasis
  void PreEmphasis(float coeff, std::vector<float>* data) const {
    if (coeff == 0.0) return;
    for (int i = data->size() - 1; i > 0; i--)
      (*data)[i] -= coeff * (*data)[i - 1];
    (*data)[0] -= coeff * (*data)[0];
  }

  // Apply window on data in place
  void ApplyWindow(std::vector<float>* data) const {
    CHECK_GE(data->size(), window_.size());
    for (size_t i = 0; i < window_.size(); ++i) {
      (*data)[i] *= window_[i];
    }
  }

  void WhisperNorm(std::vector<std::vector<float>>* feat,
                   float max_mel_engery) {
    int num_frames = feat->size();
    for (int i = 0; i < num_frames; ++i) {
      for (int j = 0; j < num_bins_; ++j) {
        float energy = (*feat)[i][j];
        if (energy < max_mel_engery - 8) energy = max_mel_engery - 8;
        energy = (energy + 4.0) / 4.0;
        (*feat)[i][j] = energy;
      }
    }
  }

  // Compute fbank feat, return num frames
  int Compute(const std::vector<float>& wave,
              std::vector<std::vector<float>>* feat) {
    int num_samples = wave.size();
    if (num_samples < frame_length_) return 0;
    const float* wave_data = wave.data();
    std::vector<float> padded_wave;
    if (center_pad_) {  // reflect padding, keep consistent with torchaudio
      int pad_length = frame_length_ / 2;
      std::vector<float> pad_left(
          wave.begin() + 1, wave.begin() + pad_length + 1);  // [1, pad_length]
      std::vector<float> pad_right(
          wave.end() - pad_length - 1,
          wave.end() - 1);  // [num_samples - pad_length - 1, num_samples - 1]
      std::reverse(pad_right.begin(), pad_right.end());
      std::reverse(pad_left.begin(), pad_left.end());
      padded_wave.reserve(num_samples + 2 * pad_length);
      padded_wave.insert(padded_wave.end(), pad_left.begin(), pad_left.end());
      padded_wave.insert(padded_wave.end(), wave.begin(), wave.end());
      padded_wave.insert(padded_wave.end(), pad_right.begin(), pad_right.end());
      num_samples = padded_wave.size();
      wave_data = padded_wave.data();
    }
    int num_frames = ((num_samples - frame_length_) / frame_shift_);
    feat->resize(num_frames);
    std::vector<float> power(fft_points_ / 2);

    float max_mel_engery = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < num_frames; ++i) {
      std::vector<float> data(wave_data + i * frame_shift_,
                              wave_data + i * frame_shift_ + frame_length_);

      if (scale_input_to_unit_) {
        for (int j = 0; j < frame_length_; ++j) {
          data[j] = data[j] / kS16AbsMax;
        }
      }

      // optional add noise
      if (dither_ != 0.0) {
        for (size_t j = 0; j < data.size(); ++j)
          data[j] += dither_ * distribution_(generator_);
      }
      // optinal remove dc offset
      if (remove_dc_offset_) {
        float mean = 0.0;
        for (size_t j = 0; j < data.size(); ++j) mean += data[j];
        mean /= data.size();
        for (size_t j = 0; j < data.size(); ++j) data[j] -= mean;
      }

      if (pre_emphasis_) {
        PreEmphasis(0.97, &data);
      }
      ApplyWindow(&data);

      if (power_of_two_) {
        // copy data to fft_real
        memset(fft_img_.data(), 0, sizeof(float) * fft_points_);
        memset(fft_real_.data() + frame_length_, 0,
               sizeof(float) * (fft_points_ - frame_length_));
        memcpy(fft_real_.data(), data.data(), sizeof(float) * frame_length_);
        fft(bitrev_.data(), sintbl_.data(), fft_real_.data(), fft_img_.data(),
            fft_points_);
        // power
        for (int j = 0; j < fft_points_ / 2; ++j) {
          power[j] = fft_real_[j] * fft_real_[j] + fft_img_[j] * fft_img_[j];
        }
      } else {
        memcpy(fft2_in_.data(), data.data(), sizeof(float) * frame_length_);
        fft2(fft2_in_.data(), fft_points_, fft2_out_.data());
        // power
        for (int j = 0; j < fft_points_ / 2; ++j) {
          power[j] = fft2_out_[j * 2 + 0] * fft2_out_[j * 2 + 0] +
                     fft2_out_[j * 2 + 1] * fft2_out_[j * 2 + 1];
        }
      }

      if (use_magnitude_) {
        // https://github.com/wenet-e2e/west/blob/main/west/utils/audio.py#L92
        for (int j = 0; j < fft_points_ / 2; ++j) {
          power[j] = std::sqrt(power[j] + 1e-9);
        }
      }

      (*feat)[i].resize(num_bins_);
      // cepstral coefficients, triangle filter array

      for (int j = 0; j < num_bins_; ++j) {
        float mel_energy = 0.0;
        int s = bins_[j].first;
        for (size_t k = 0; k < bins_[j].second.size(); ++k) {
          mel_energy += bins_[j].second[k] * power[s + k];
        }
        // optional use log
        if (use_log_) {
          if (mel_energy < log_floor_) mel_energy = log_floor_;

          if (log_base_ == LogBase::kBaseE)
            mel_energy = logf(mel_energy);
          else if (log_base_ == LogBase::kBase10)
            mel_energy = log10(mel_energy);
        }
        if (max_mel_engery < mel_energy) max_mel_engery = mel_energy;
        (*feat)[i][j] = mel_energy;
      }
    }
    if (norm_type_ == NormalizationType::kWhisper)
      WhisperNorm(feat, max_mel_engery);

    return num_frames;
  }

 protected:
  int num_bins_;
  int sample_rate_;
  int frame_length_, frame_shift_;
  int fft_points_;
  bool use_log_;
  bool remove_dc_offset_;
  bool pre_emphasis_;
  bool scale_input_to_unit_;
  float low_freq_;
  float log_floor_;
  float high_freq_;
  LogBase log_base_;
  NormalizationType norm_type_;

  std::vector<float> center_freqs_;
  std::vector<std::pair<int, std::vector<float>>> bins_;
  std::vector<float> window_;
  std::default_random_engine generator_;
  std::normal_distribution<float> distribution_;
  float dither_;

  // bit reversal table
  std::vector<int> bitrev_;
  // trigonometric function table
  std::vector<float> sintbl_;
  std::vector<float> fft_real_, fft_img_;

  bool power_of_two_;
  std::vector<float> fft2_in_;
  std::vector<float> fft2_out_;

  bool center_pad_;
  bool use_magnitude_ = false;
};

// LogMelSpectrogram feature extractor with specific configuration
// Configuration:
//   low_freq = 0.0
//   pre_emphasis = false
//   log_floor = 1e-10
//   log_base = LogBase::kBase10
//   window_type = WindowType::kHanning
//   mel_type = MelType::kSlaney
//   scale_input_to_unit = true
//   norm_type = NormalizationType::kWhisper
//   power_of_two = false
class LogMelSpectrogram : public Fbank {
 public:
  LogMelSpectrogram(int num_bins, int sample_rate, int frame_length,
                    int frame_shift)
      : Fbank(num_bins, sample_rate, frame_length, frame_shift,
              0.0f,                         // low_freq
              false,                        // pre_emphasis
              true,                         // scale_input_to_unit
              1e-10f,                       // log_floor
              LogBase::kBase10,             // log_base
              WindowType::kHanning,         // window_type
              MelType::kSlaney,             // mel_type
              NormalizationType::kWhisper,  // norm_type
              false,                        // power_of_two
              true) {                       // center_pad
    this->set_dither(0.0);
    this->set_remove_dc_offset(false);
  }
};

class LogMelSpectrogramVocoder : public Fbank {
 public:
  LogMelSpectrogramVocoder()
      : Fbank(80, 22050, 1024, 256,
              0.0f,                       // low_freq
              false,                      // pre_emphasis
              true,                       // scale_input_to_unit
              1e-5f,                      // log_floor
              LogBase::kBaseE,            // log_base
              WindowType::kHanning,       // window_type
              MelType::kSlaney,           // mel_type
              NormalizationType::kKaldi,  // norm_type
              true,                       // power_of_two
              true) {                     // center_pad
    high_freq_ = 8000;
    use_magnitude_ = true;
    this->set_dither(0.0);
    this->set_remove_dc_offset(false);
  }
};

}  // namespace wenet

#endif  // FRONTEND_FBANK_H_

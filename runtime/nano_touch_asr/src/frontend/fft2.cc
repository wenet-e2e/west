// Copyright (c) 2025 Binbin Zhang(binzha@qq.com)

#include <math.h>

#include <iostream>
#include <vector>

#include "frontend/fft2.h"

// Copy from whisper.cpp
// https://github.com/ggml-org/whisper.cpp/blob/master/src/whisper.cpp#L2997

#define WHISPER_N_FFT 400
#define SIN_COS_N_COUNT WHISPER_N_FFT

namespace {
struct whisper_global_cache {
  // In FFT, we frequently use sine and cosine operations with the same values.
  // We can use precalculated values to speed up the process.
  float sin_vals[SIN_COS_N_COUNT];
  float cos_vals[SIN_COS_N_COUNT];

  // Hann window (Use cosf to eliminate difference)
  // ref: https://pytorch.org/docs/stable/generated/torch.hann_window.html
  // ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L147
  float hann_window[WHISPER_N_FFT];

  whisper_global_cache() {
    fill_sin_cos_table();
    fill_hann_window(sizeof(hann_window) / sizeof(hann_window[0]), true,
                     hann_window);
  }

  void fill_sin_cos_table() {
    for (int i = 0; i < SIN_COS_N_COUNT; i++) {
      double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
      sin_vals[i] = sinf(theta);
      cos_vals[i] = cosf(theta);
    }
  }

  void fill_hann_window(int length, bool periodic, float* output) {
    int offset = -1;
    if (periodic) {
      offset = 0;
    }
    for (int i = 0; i < length; i++) {
      output[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
  }
} global_cache;
}  // namespace

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
static void dft(const float* in, int N, float* out) {
  const int sin_cos_step = SIN_COS_N_COUNT / N;

  for (int k = 0; k < N; k++) {
    float re = 0;
    float im = 0;

    for (int n = 0; n < N; n++) {
      int idx = (k * n * sin_cos_step) % (SIN_COS_N_COUNT);  // t = 2*M_PI*k*n/N
      re += in[n] * global_cache.cos_vals[idx];              // cos(t)
      im -= in[n] * global_cache.sin_vals[idx];              // sin(t)
    }

    out[k * 2 + 0] = re;
    out[k * 2 + 1] = im;
  }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
void fft2(float* in, int N, float* out) {
  if (N == 1) {
    out[0] = in[0];
    out[1] = 0;
    return;
  }

  const int half_N = N / 2;
  if (N - half_N * 2 == 1) {
    dft(in, N, out);
    return;
  }

  float* even = in + N;
  for (int i = 0; i < half_N; ++i) {
    even[i] = in[2 * i];
  }
  float* even_fft = out + 2 * N;
  fft2(even, half_N, even_fft);

  float* odd = even;
  for (int i = 0; i < half_N; ++i) {
    odd[i] = in[2 * i + 1];
  }
  float* odd_fft = even_fft + N;
  fft2(odd, half_N, odd_fft);

  const int sin_cos_step = SIN_COS_N_COUNT / N;
  for (int k = 0; k < half_N; k++) {
    int idx = k * sin_cos_step;              // t = 2*M_PI*k/N
    float re = global_cache.cos_vals[idx];   // cos(t)
    float im = -global_cache.sin_vals[idx];  // sin(t)

    float re_odd = odd_fft[2 * k + 0];
    float im_odd = odd_fft[2 * k + 1];

    out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
    out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

    out[2 * (k + half_N) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
    out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
  }
}

// int main() {
//   int kNum = 400;
//   std::vector<float> fft_in(2 * kNum, 0.0);
//   for (int i = 0; i < kNum; i++) {
//     fft_in[i] = i;
//   }
//   std::vector<float> fft_out(2 * 2 * 2 * kNum, 0.0);

//   fft2(fft_in.data(), kNum, fft_out.data());

//   for (int i = 0; i < fft_out.size(); i+=2) {
//     std::cout << fft_out[i] << " " << fft_out[i+1] << "\n";
//   }

//   return 0;
// }

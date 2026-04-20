// CPU FFT baselines: iterative radix-2 and radix-4 Cooley-Tukey.
// g++ -O2 -std=c++17 fft_cpu.cpp -o fft_cpu

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

using cpx = std::complex<double>;
const double PI = std::acos(-1.0);

// O(N^2) DFT, used as the reference for correctness.
std::vector<cpx> naive_dft(const std::vector<cpx> &x) {
  int n = x.size();
  std::vector<cpx> X(n);
  for (int k = 0; k < n; k++) {
    cpx sum = 0;
    for (int j = 0; j < n; j++) {
      double ang = -2.0 * PI * k * j / n;
      sum += x[j] * cpx(std::cos(ang), std::sin(ang));
    }
    X[k] = sum;
  }
  return X;
}

// In-place bit-reversal permutation.
void bitrev(std::vector<cpx> &data) {
  int n = data.size();
  int j = 0;
  for (int i = 1; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      std::swap(data[i], data[j]);
  }
}

// Iterative radix-2 FFT. N must be a power of 2.
void fft2(std::vector<cpx> &data, bool inv = false) {
  int n = data.size();
  bitrev(data);
  for (int len = 2; len <= n; len *= 2) {
    double ang = (inv ? 2.0 : -2.0) * PI / len;
    cpx wlen(std::cos(ang), std::sin(ang));
    int half = len / 2;
    for (int base = 0; base < n; base += len) {
      cpx w = 1;
      for (int k = 0; k < half; k++) {
        cpx u = data[base + k];
        cpx v = data[base + k + half] * w;
        data[base + k] = u + v;
        data[base + k + half] = u - v;
        w *= wlen;
      }
    }
  }
  if (inv) {
    for (int i = 0; i < n; i++)
      data[i] /= n;
  }
}

// Base-4 digit-reversal, used by radix-4 FFT.
void digitrev4(std::vector<cpx> &data) {
  int n = data.size();
  int log4 = 0;
  while ((1 << (2 * log4)) < n)
    log4++;
  for (int i = 0; i < n; i++) {
    int src = i, rev = 0;
    for (int k = 0; k < log4; k++) {
      rev = (rev << 2) | (src & 3);
      src >>= 2;
    }
    if (i < rev)
      std::swap(data[i], data[rev]);
  }
}

// Iterative radix-4 FFT. N must be a power of 4.
void fft4(std::vector<cpx> &data, bool inv = false) {
  int n = data.size();
  digitrev4(data);
  double sign = inv ? 1.0 : -1.0;

  for (int len = 4; len <= n; len *= 4) {
    int quarter = len / 4;
    double ang = sign * 2.0 * PI / len;
    cpx w1(std::cos(ang), std::sin(ang));
    cpx w2(std::cos(2 * ang), std::sin(2 * ang));
    cpx w3(std::cos(3 * ang), std::sin(3 * ang));

    for (int base = 0; base < n; base += len) {
      cpx wa = 1, wb = 1, wc = 1;
      for (int k = 0; k < quarter; k++) {
        cpx a0 = data[base + k];
        cpx a1 = data[base + k + quarter] * wa;
        cpx a2 = data[base + k + 2 * quarter] * wb;
        cpx a3 = data[base + k + 3 * quarter] * wc;

        cpx sum02 = a0 + a2;
        cpx dif02 = a0 - a2;
        cpx sum13 = a1 + a3;
        cpx dif13 = a1 - a3;
        // forward multiplies dif13 by -j; inverse by +j.
        cpx dif13_rot;
        if (inv)
          dif13_rot = cpx(-dif13.imag(), dif13.real());
        else
          dif13_rot = cpx(dif13.imag(), -dif13.real());

        data[base + k] = sum02 + sum13;
        data[base + k + quarter] = dif02 + dif13_rot;
        data[base + k + 2 * quarter] = sum02 - sum13;
        data[base + k + 3 * quarter] = dif02 - dif13_rot;

        wa *= w1;
        wb *= w2;
        wc *= w3;
      }
    }
  }
  if (inv) {
    for (int i = 0; i < n; i++)
      data[i] /= n;
  }
}

double max_err(const std::vector<cpx> &A, const std::vector<cpx> &B) {
  double e = 0;
  for (size_t i = 0; i < A.size(); i++)
    e = std::max(e, std::abs(A[i] - B[i]));
  return e;
}

int main() {
  const int N = 1 << 12; // 4096 = 4^6, works for both radix-2 and radix-4

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1, 1);

  std::vector<cpx> x(N);
  for (int i = 0; i < N; i++)
    x[i] = cpx(dist(rng), dist(rng));

  // radix-2 timing
  std::vector<cpx> x_r2 = x;
  auto t_start = std::chrono::high_resolution_clock::now();
  fft2(x_r2);
  auto t_end = std::chrono::high_resolution_clock::now();
  long us_r2 =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // radix-4 timing
  std::vector<cpx> x_r4 = x;
  t_start = std::chrono::high_resolution_clock::now();
  fft4(x_r4);
  t_end = std::chrono::high_resolution_clock::now();
  long us_r4 =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  printf("N=%d\n", N);
  printf("  radix-2: %ld us\n", us_r2);
  printf("  radix-4: %ld us\n", us_r4);
  printf("  radix-2 vs radix-4 err: %.3e\n", max_err(x_r2, x_r4));

  // correctness check vs naive DFT on a small N
  const int Ns = 64;
  std::vector<cpx> y(Ns);
  for (int i = 0; i < Ns; i++)
    y[i] = cpx(dist(rng), dist(rng));

  std::vector<cpx> ref = naive_dft(y);
  std::vector<cpx> y_r2 = y;
  fft2(y_r2);
  std::vector<cpx> y_r4 = y;
  fft4(y_r4);
  printf("  (N=%d) radix-2 vs DFT: %.3e\n", Ns, max_err(y_r2, ref));
  printf("  (N=%d) radix-4 vs DFT: %.3e\n", Ns, max_err(y_r4, ref));

  // forward/inverse round-trip check
  std::vector<cpx> rt = x;
  fft2(rt);
  fft2(rt, true);
  printf("  forward/inverse round-trip: %.3e\n", max_err(rt, x));

  return 0;
}

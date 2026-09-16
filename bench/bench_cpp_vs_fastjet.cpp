// flashjet vs FastJet, both sides pure C++ on identical events.
//
// Benchmarking through the scikit-hep `fastjet` Python bindings understates
// FastJet (per-event PseudoJet construction crosses the binding) and mixes in
// awkward-array overhead, so the honest bar is FastJet's own C++ library with
// its default `Best` strategy -- which for anti-kt R=0.4 means N2Plain below
// N~39, N2Tiled, then N2MinHeapTiled, then N2MHTLazy9 as N grows.  The
// strategy actually chosen is printed for each point.
//
// Two things this harness is careful about:
//
//   * FastJet is timed BOTH ways.  `cached` builds the PseudoJets up front,
//     and PseudoJet fills its rapidity/phi cache lazily, so from the second
//     repeat on FastJet gets the initial atan2/log for free while we recompute
//     from raw components every run.  `from-raw` starts from px,py,pz,E --
//     the same job flashjet is measured doing, and the honest comparison.
//   * The sides are INTERLEAVED round by round and reported as
//     distributions, not a single best-of.  This box drifts under thermal and
//     background load; measuring all of one side then all of the other lets
//     that drift masquerade as a difference.  The ratio is formed within each
//     round, then quantiled, so drift cancels.
//
// Build with scripts/bench_cpp.sh (it locates libfastjet from the installed
// python package).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include "fastjet/ClusterSequence.hh"

extern "C" void flashjet_cluster_f64(const double*, const uint8_t*, int64_t, int64_t,
                                     double, double, int64_t*, int64_t*, int64_t*, double*, int);

static std::vector<std::vector<double>> gen(int64_t B, int64_t N, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<double> U(0, 1);
  std::vector<std::vector<double>> out(B);
  for (int64_t b = 0; b < B; ++b) {
    out[b].resize(N * 4);
    for (int64_t k = 0; k < N; ++k) {
      const double pt = 0.5 + 79.5 * U(g), y = -3 + 6 * U(g);
      const double ph = 6.28318530718 * U(g), m = U(g);
      const double mt = std::sqrt(m * m + pt * pt);
      out[b][4 * k + 0] = pt * std::cos(ph);
      out[b][4 * k + 1] = pt * std::sin(ph);
      out[b][4 * k + 2] = mt * std::sinh(y);
      out[b][4 * k + 3] = mt * std::cosh(y);
    }
  }
  return out;
}

static double quantile(std::vector<double> v, double f) {
  std::sort(v.begin(), v.end());
  const double x = f * (v.size() - 1);
  const size_t i = static_cast<size_t>(x);
  const double frac = x - i;
  return (i + 1 < v.size()) ? v[i] * (1 - frac) + v[i + 1] * frac : v[i];
}

template <typename F>
static double tick(F &&fn) {
  const auto t0 = std::chrono::steady_clock::now();
  fn();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

int main(int argc, char **argv) {
  const int64_t B = atoi(argv[1]), N = atoi(argv[2]);
  const int rounds = argc > 3 ? atoi(argv[3]) : 15;
  const double R = argc > 4 ? atof(argv[4]) : 0.4;
  const int threads = argc > 5 ? atoi(argv[5]) : 1;

  const auto evs = gen(B, N, 1);
  fastjet::JetDefinition jd(fastjet::antikt_algorithm, R);

  std::vector<std::vector<fastjet::PseudoJet>> pjs(B);
  for (int64_t b = 0; b < B; ++b) {
    pjs[b].reserve(N);
    for (int64_t k = 0; k < N; ++k)
      pjs[b].push_back(fastjet::PseudoJet(evs[b][4 * k], evs[b][4 * k + 1],
                                          evs[b][4 * k + 2], evs[b][4 * k + 3]));
  }

  std::vector<double> flat(B * N * 4);
  std::vector<uint8_t> mask(B * N, 1);
  for (int64_t b = 0; b < B; ++b)
    for (int64_t k = 0; k < N * 4; ++k) flat[b * N * 4 + k] = evs[b][k];
  std::vector<int64_t> h1(B * N), h2(B * N), h3(B * N);
  std::vector<double> hd(B * N);

  auto fj_cached = [&]() {
    for (int64_t b = 0; b < B; ++b) {
      fastjet::ClusterSequence cs(pjs[b], jd);
      cs.inclusive_jets();
    }
  };
  // Each worker gets its own ClusterSequence over its own events.  This build
  // defines FASTJET_HAVE_THREAD_SAFETY, so that is a supported thing to do,
  // and quoting our threaded numbers against a single-threaded FastJet would
  // not be honest.
  auto fj_raw = [&]() {
    auto work = [&](int64_t lo, int64_t hi) {
      for (int64_t b = lo; b < hi; ++b) {
        std::vector<fastjet::PseudoJet> in;
        in.reserve(N);
        for (int64_t k = 0; k < N; ++k)
          in.push_back(fastjet::PseudoJet(evs[b][4 * k], evs[b][4 * k + 1],
                                          evs[b][4 * k + 2], evs[b][4 * k + 3]));
        fastjet::ClusterSequence cs(in, jd);
        cs.inclusive_jets();
      }
    };
    if (threads <= 1) { work(0, B); return; }
    std::vector<std::thread> ts;
    const int64_t chunk = (B + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
      const int64_t lo = std::min<int64_t>(t * chunk, B), hi = std::min<int64_t>(lo + chunk, B);
      if (lo < hi) ts.emplace_back(work, lo, hi);
    }
    for (auto &x : ts) x.join();
  };
  auto ours = [&]() {
    flashjet_cluster_f64(flat.data(), mask.data(), B, N, R, -1.0, h1.data(), h2.data(),
                         h3.data(), hd.data(), threads);
  };

  {
    fastjet::ClusterSequence cs(pjs[0], jd);
    printf("  fastjet strategy: %s\n", cs.strategy_string().c_str());
  }

  std::vector<double> v_cached, v_raw, v_ours, v_ratio;
  for (int r = 0; r <= rounds; ++r) {  // round 0 is warm-up and discarded
    const double a = tick(fj_cached);
    const double b = tick(fj_raw);
    const double c = tick(ours);
    if (r == 0) continue;
    v_cached.push_back(a);
    v_raw.push_back(b);
    v_ours.push_back(c);
    v_ratio.push_back(b / c);
  }

  const double us = 1e6 / B;
  printf("N=%5ld B=%-5ld %dt | fj-cached %7.1f | fj-raw %7.1f [%7.1f %7.1f] | "
         "flashjet %7.1f [%7.1f %7.1f] us/ev | ratio %.3f [%.3f %.3f] n=%d\n",
         N, B, threads, quantile(v_cached, 0.5) * us,
         quantile(v_raw, 0.5) * us, quantile(v_raw, 0.16) * us, quantile(v_raw, 0.84) * us,
         quantile(v_ours, 0.5) * us, quantile(v_ours, 0.16) * us, quantile(v_ours, 0.84) * us,
         quantile(v_ratio, 0.5), quantile(v_ratio, 0.16), quantile(v_ratio, 0.84),
         static_cast<int>(v_ratio.size()));
  return 0;
}

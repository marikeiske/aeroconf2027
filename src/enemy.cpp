// enemy.cpp - memory-bandwidth co-runner used to exercise the shared-memory
// interference channel (CAST-32A / AMC 20-193 "interference channels").
// Streams read-modify-write over a buffer larger than the last-level cache.
// Usage: enemy <MiB> <seconds> [threads]
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  size_t mib = argc > 1 ? std::atoi(argv[1]) : 64;
  double secs = argc > 2 ? std::atof(argv[2]) : 60;
  int th = argc > 3 ? std::atoi(argv[3]) : 1;
  auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(secs);
  std::atomic<unsigned long long> sweeps{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < th; ++t)
    ts.emplace_back([&, t]() {
      size_t n = mib * 1024 * 1024 / sizeof(long long);
      std::vector<long long> buf(n, t);
      volatile long long sink = 0;
      while (std::chrono::steady_clock::now() < end) {
        for (size_t i = 0; i < n; i += 8) buf[i] += 1;  // one touch per 64 B line
        sink += buf[n / 2];
        sweeps++;
      }
    });
  for (auto& t : ts) t.join();
  std::printf("enemy: %llu sweeps\n", (unsigned long long)sweeps.load());
  return 0;
}

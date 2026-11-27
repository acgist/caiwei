#ifndef CAIWEI_RUNTIME_PERF_HPP
#define CAIWEI_RUNTIME_PERF_HPP

#include <cstdint>

namespace caiwei {
namespace perf {

struct ContextPerf {
    // clang-format off
    const int type = 0;
    int64_t  time               = 0;
    uint64_t audio_time         = 0;
    uint64_t audio_n_frames     = 0;
    uint64_t image_time         = 0;
    uint64_t image_n_frames     = 0;
    uint64_t prefill_time       = 0;
    uint64_t prefill_n_tokens   = 0; // 手动统计
    uint64_t prefill_n_tokens_  = 0; // 官方统计
    uint64_t generate_time      = 0;
    uint64_t generate_n_tokens  = 0; // 手动统计
    uint64_t generate_n_tokens_ = 0; // 官方统计
    // clang-format on

    void reset();
    void merge(ContextPerf& other);

    void set_audio_n_frames(uint64_t n_frames);
    void set_image_n_frames(uint64_t n_frames);
    void set_prefill_n_tokens(uint64_t n_tokens, uint64_t n_tokens_, bool last = false);
    void set_generate_n_tokens(uint64_t n_tokens, uint64_t n_tokens_, bool last = false);

    ContextPerf(int type);
    ~ContextPerf();
};

} // namespace perf
} // namespace caiwei

#endif // CAIWEI_RUNTIME_PERF_HPP

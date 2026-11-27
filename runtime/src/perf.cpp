#include "caiwei/perf.hpp"
#include "caiwei/context.hpp"
#include "caiwei/env.hpp"

#include <cinttypes>

static void context_pref_print(const caiwei::perf::ContextPerf& perf);

caiwei::perf::ContextPerf::ContextPerf(int type) : type(type), time(caiwei::env::steady_timestamp()) {
}

caiwei::perf::ContextPerf::~ContextPerf() {
    context_pref_print(*this);
}

void caiwei::perf::ContextPerf::reset() {
    this->time = caiwei::env::steady_timestamp();
}

void caiwei::perf::ContextPerf::merge(ContextPerf& other) {
    // clang-format off
    this->audio_time         += std::exchange(other.audio_time,         0);
    this->audio_n_frames     += std::exchange(other.audio_n_frames,     0);
    this->image_time         += std::exchange(other.image_time,         0);
    this->image_n_frames     += std::exchange(other.image_n_frames,     0);
    this->prefill_time       += std::exchange(other.prefill_time,       0);
    this->prefill_n_tokens   += std::exchange(other.prefill_n_tokens,   0);
    this->prefill_n_tokens_  += std::exchange(other.prefill_n_tokens_,  0);
    this->generate_time      += std::exchange(other.generate_time,      0);
    this->generate_n_tokens  += std::exchange(other.generate_n_tokens,  0);
    this->generate_n_tokens_ += std::exchange(other.generate_n_tokens_, 0);
    // clang-format on
}

void caiwei::perf::ContextPerf::set_audio_n_frames(uint64_t n_frames) {
    this->audio_n_frames += n_frames;
    auto time = caiwei::env::steady_timestamp();
    this->audio_time += time - this->time;
    this->time = time;
}

void caiwei::perf::ContextPerf::set_image_n_frames(uint64_t n_frames) {
    this->image_n_frames += n_frames;
    auto time = caiwei::env::steady_timestamp();
    this->image_time += time - this->time;
    this->time = time;
}

void caiwei::perf::ContextPerf::set_prefill_n_tokens(uint64_t n_tokens, uint64_t n_tokens_, bool last) {
    // clang-format off
    this->prefill_n_tokens  += n_tokens;
    this->prefill_n_tokens_ += n_tokens_;
    // clang-format on
    if (last) {
        auto time = caiwei::env::steady_timestamp();
        this->prefill_time += time - this->time;
        this->time = time;
    }
}

void caiwei::perf::ContextPerf::set_generate_n_tokens(uint64_t n_tokens, uint64_t n_tokens_, bool last) {
    // clang-format off
    this->generate_n_tokens  += n_tokens;
    this->generate_n_tokens_ += n_tokens_;
    // clang-format on
    if (last) {
        auto time = caiwei::env::steady_timestamp();
        this->generate_time += time - this->time;
        this->time = time;
    }
}

static void context_pref_print(const caiwei::perf::ContextPerf& perf) {
    // clang-format off
    const uint64_t prefill_time      = perf.prefill_time;
    const uint64_t prefill_n_tokens  = std::max(perf.prefill_n_tokens, perf.prefill_n_tokens_);
    const uint64_t generate_time     = perf.generate_time;
    const uint64_t generate_n_tokens = std::max(perf.generate_n_tokens, perf.generate_n_tokens_);
    const uint64_t audio_time        = perf.audio_time;
    const uint64_t audio_n_frames    = perf.audio_n_frames;
    const uint64_t image_time        = perf.image_time;
    const uint64_t image_n_frames    = perf.image_n_frames;
    if (
        prefill_time  == 0 &&
        generate_time == 0 &&
        audio_time    == 0 &&
        image_time    == 0 &&
        prefill_n_tokens  == 0 &&
        generate_n_tokens == 0 &&
        audio_n_frames    == 0 &&
        image_n_frames    == 0
    ) {
        return;
    }
    // clang-format on
    std::printf("\n");
    std::printf("--------------------------------------------------------------------------------------\n");
    std::printf("Context type = %s\n", caiwei::context::get_type_name(static_cast<caiwei::context::Type>(perf.type)));
    std::printf("--------------------------------------------------------------------------------------\n");
    std::printf(
        " %-12s  %-15s  %-8s  %-23s  %-23s\n", "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second"
    );
    std::printf("--------------------------------------------------------------------------------------\n");
    if (prefill_time > 0 && prefill_n_tokens > 0) {
        const float prefill_tpt = 1.0F * prefill_time / prefill_n_tokens;
        const float prefill_tps = 1e3F / prefill_time * prefill_n_tokens;
        std::printf(
            " %-12s  %-15" PRIu64 "  %-8" PRIu64 "  %-23.2f  %-23.2f\n",
            "Prefill",
            prefill_time,
            prefill_n_tokens,
            prefill_tpt,
            prefill_tps
        );
    }
    if (generate_time > 0 && generate_n_tokens > 0) {
        const float generate_tpt = 1.0F * generate_time / generate_n_tokens;
        const float generate_tps = 1e3F / generate_time * generate_n_tokens;
        std::printf(
            " %-12s  %-15" PRIu64 "  %-8" PRIu64 "  %-23.2f  %-23.2f\n",
            "Generate",
            generate_time,
            generate_n_tokens,
            generate_tpt,
            generate_tps
        );
    }
    if (audio_time > 0 && audio_n_frames > 0) {
        std::printf("--------------------------------------------------------------------------------------\n");
        const float fps = 1e3F / audio_time * audio_n_frames;
        std::printf(" Audio frames = %" PRIu64 " / %" PRIu64 " ms = %.2f FPS\n", audio_n_frames, audio_time, fps);
    }
    if (image_time > 0 && image_n_frames > 0) {
        std::printf("--------------------------------------------------------------------------------------\n");
        const float fps = 1e3F / image_time * image_n_frames;
        std::printf(" Image frames = %" PRIu64 " / %" PRIu64 " ms = %.2f FPS\n", image_n_frames, image_time, fps);
    }
    std::printf("--------------------------------------------------------------------------------------\n");
    std::printf("\n");
}

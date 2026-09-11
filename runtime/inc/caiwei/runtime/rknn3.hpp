#ifndef CAIWEI_RUNTIME_RKNN3_HPP
#define CAIWEI_RUNTIME_RKNN3_HPP

#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
#include "caiwei/context.hpp"
#include "caiwei/text_tool.hpp"

#include <mutex>
#include <chrono>
#include <generator>
#include <filesystem>
#include <condition_variable>

#include <sys/stat.h>

#include "rknn3/rknn3_api.h"
#include "caiwei/runtime/tokenizer.hpp"

namespace caiwei  {
namespace context {

struct ContextSession {
std::mutex mutex;
std::condition_variable cv;
Tokenizer* tokenizer = nullptr;
int      embedding_dim;
float16* embedding_data = nullptr;
bool first = false;
bool end   = false;
bool thinking = false;
bool toolcall = false;
int32_t b_thinking;
int32_t e_thinking;
int32_t b_toolcall;
int32_t e_toolcall;
int n_decode_tokens;
int n_prefill_tokens;
caiwei::text::ResultToolcall result_toolcall;
std::vector<caiwei::text::Result> token;
std::chrono::system_clock::time_point llm_begin_time;
std::chrono::system_clock::time_point llm_first_time;
std::chrono::system_clock::time_point llm_end_time;
};

int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len);
int result_callback(void* userdata, RKLLMResult* result, LLMCallState state);
int tokenizer_callback(void* userdata, const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max);
void printf_session_perf(caiwei::context::ContextSession* session);

class RKNN3Context {
protected:
    int32_t max_token_length;
    std::string model_path;
    std::string weight_path;
    std::string tokenizer_path;
    std::string embedding_path;
    rknn3_context context = 0;
    Tokenizer* tokenizer = nullptr;
    int      vocab_size;
    int      embedding_fd;
    int      embedding_dim;
    float16* embedding_data;
    struct stat emb_st{};
    caiwei::text::ChatTemplate chat_template;
    caiwei::text::SpecialToken special_token;
public:
    bool load_model();
    rknn3_session* get_session(const caiwei::text::CompletionsRequest& request, ContextSession* context_session);
    std::generator<caiwei::text::Result> generate(const caiwei::text::CompletionsRequest& request);
public:
    RKNN3Context(std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path, int32_t max_token_length, caiwei::text::SpecialToken special_token);
    ~RKNN3Context();
};

class ClsRKNN3Context  : public ClsContext,  public RKNN3Context {};
class DetRKNN3Context  : public DetContext,  public RKNN3Context {};
class SegRKNN3Context  : public SegContext,  public RKNN3Context {};
class PoseRKNN3Context : public PoseContext, public RKNN3Context {};
class ASRRKNN3Context  : public ASRContext,  public RKNN3Context {};

class LLMRKNN3Context : public LLMContext, public RKNN3Context {
public:
    LLMRKNN3Context(std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime);
    ~LLMRKNN3Context();
public:
public:
    bool load() override;
    std::generator<std::string> run(const caiwei::text::CompletionsRequest& request) override;
};

class VLMRKNN3Context  : public VLMContext,  public RKNN3Context {};
class EmbeddingRKNN3Context : public EmbeddingContext, public RKNN3Context {};
class RerankingRKNN3Context : public RerankingContext, public RKNN3Context {};

}
}

#endif // CAIWEI_RUNTIME_RKNN3_HPP

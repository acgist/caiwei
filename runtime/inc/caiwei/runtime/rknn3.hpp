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

#ifdef ENABLE_SPEEDUP
#include "speedup.h"
#endif

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
int vision_latency   = 0;
int n_decode_tokens  = 0;
int n_prefill_tokens = 0;
caiwei::text::ResultToolcall result_toolcall;
std::vector<caiwei::text::Result> token;
std::chrono::system_clock::time_point llm_begin_time;
std::chrono::system_clock::time_point llm_first_time;
std::chrono::system_clock::time_point llm_end_time;
std::vector<rknn3_tensor> output_tensors{};
std::vector<std::vector<float>> model_output{};
};

int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len);
int result_callback(void* userdata, RKLLMResult* result, LLMCallState state);
int tokenizer_callback(void* userdata, const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max);
int output_callback(void* userdata, rknn3_tensor* output_tensors, uint32_t n_output_tensors, LLMOutputCallbackState state);
void printf_session_perf(caiwei::context::ContextSession* session);

bool init_output(rknn3_context context, caiwei::context::ContextSession* session, int n_output_tensors);
void free_output(rknn3_context context, caiwei::context::ContextSession* session);

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
    virtual std::vector<rknn3_llm_input> get_inputs(rknn3_session* session, const caiwei::text::CompletionsRequest& request);
    rknn3_session* get_session(int max_tokens, rknn3_sampling_params sampling_params, ContextSession* context_session);
    std::generator<caiwei::text::Result> generate(const caiwei::text::CompletionsRequest& request);
public:
    RKNN3Context(std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path, int32_t max_token_length, caiwei::text::SpecialToken special_token);
    ~RKNN3Context();
};

// class ASRRKNN3Context  : public ASRContext,  public RKNN3Context {};

class LLMRKNN3Context : public LLMContext, public RKNN3Context {
public:
    LLMRKNN3Context(std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime);
    ~LLMRKNN3Context();
public:
    bool load() override;
    std::generator<caiwei::text::Result> run(const caiwei::text::CompletionsRequest& request) override;
};

// class VLMRKNN3Context : public VLMContext,  public RKNN3Context {
// public:
//     int n_internal_mems;
//     std::vector<rknn3_tensor_mem*> internal_mems;
// protected:
//     std::string vlm_model_path;
//     rknn3_context vlm_context = 0;
//     int input_size;
//     int output_size;
//     int model_channel;
//     int model_height;
//     int model_width;
//     uint32_t* embeds_shape;
//     uint32_t embeds_ndims;
//     std::vector<rknn3_tensor> inputs;
//     std::vector<rknn3_tensor> outputs;
//     std::vector<rknn3_tensor_attr> input_attrs;
//     std::vector<rknn3_tensor_attr> output_attrs;
//     int deepstack_aligned_size;
//     int pruned_version_flag;
//     rknn3_tensor_attr deepstack_attrs[3];
//     std::vector<rknn3_aux_tensor> deepstack_tensor;
// public:
//     bool load_vlm_model();
//     bool vlm_run(float16* img_embeds, float16* deepstack_data0, float16* deepstack_data1, float16* deepstack_data2);
//     std::vector<rknn3_llm_input> get_inputs(rknn3_session* session, const caiwei::text::CompletionsRequest& request) override;
//     std::generator<caiwei::text::Result> run(const caiwei::text::CompletionsRequest& request) override;
// public:
//     VLMRKNN3Context();
//     ~VLMRKNN3Context();
// };

class EmbeddingRKNN3Context : public EmbeddingContext, public RKNN3Context {
public:
    bool load() override;
    caiwei::text::EmbeddingResult run(const caiwei::text::EmbeddingsRequest& request) override;
public:
    EmbeddingRKNN3Context(std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime);
    ~EmbeddingRKNN3Context();
};

class RerankingRKNN3Context : public RerankingContext, public RKNN3Context {
    private:
    std::string bos_key;
    std::string eos_key;
    std::string system_prompt;
    std::string instruction_prompt;
    std::string instruction_key;
    std::string query_key;
    std::string document_key;
public:
    bool load() override;
    caiwei::text::RerankingResult run(const caiwei::text::RerankingsRequest& request) override;
public:
    RerankingRKNN3Context(
        std::string model_path,
        std::string weight_path,
        std::string embedding_path,
        std::string tokenizer_path,
        int32_t max_token_length,
        std::string bos_key,
        std::string eos_key,
        std::string system_prompt,
        std::string instruction_prompt,
        std::string instruction_key,
        std::string query_key,
        std::string document_key,
        caiwei::text::SpecialToken special_token,
        caiwei::runtime::Runtime* runtime
    );
    ~RerankingRKNN3Context();
};

}
}

#endif // CAIWEI_RUNTIME_RKNN3_HPP

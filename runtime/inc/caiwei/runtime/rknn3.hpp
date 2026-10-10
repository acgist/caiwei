#ifndef CAIWEI_RUNTIME_RKNN3_HPP
#define CAIWEI_RUNTIME_RKNN3_HPP

#include "caiwei/context.hpp"
#include "caiwei/text_tool.hpp"
#include "caiwei/tokenizer.hpp"

#include <mutex>
#include <chrono>
#include <generator>
#include <condition_variable>

#include <sys/stat.h>

#include "rknn3/rknn3_api.h"

namespace caiwei  {
namespace context {

inline auto rknn3_session_deleter = [](rknn3_session* session) {
    if (session) {
        rknn3_session_destroy(session);
    }
};

using rknn3_session_ptr = std::unique_ptr<rknn3_session, decltype(rknn3_session_deleter)>;

struct ContextSession {
std::mutex mutex;
std::condition_variable cv;
Tokenizer* tokenizer{ nullptr };
int      embedding_dim;
float16* embedding_data{ nullptr };
bool stop     = false;
bool thinking = false;
bool toolcall = false;
int32_t b_thinking;
int32_t e_thinking;
int32_t b_toolcall;
int32_t e_toolcall;
int audio_frames     = 0;
int image_frames     = 0;
int media_latency    = 0;
int n_decode_tokens  = 0;
int n_prefill_tokens = 0;
float prefill_ms = 0;
float decode_ms  = 0;
std::vector<caiwei::text::Result> token{};
std::vector<std::vector<float>> model_output{};
caiwei::text::ResultToolcall result_toolcall;
};

int embed_callback    (void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len);
int output_callback   (void* userdata, rknn3_tensor* output_tensors, uint32_t n_output_tensors, LLMOutputCallbackState state);
int result_callback   (void* userdata, RKLLMResult* result, LLMCallState state);
int tokenizer_callback(void* userdata, const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max);

void printf_session_perf(caiwei::context::ContextSession* session);

class RKNN3Context {
protected:
    int32_t max_token_length;
    std::string model_path;
    std::string weight_path;
    std::string embedding_path;
    std::string tokenizer_path;
    std::string media_model_path;
    std::string media_weight_path;
    rknn3_context context = 0;
    rknn3_context media_context = 0;
    Tokenizer* tokenizer = nullptr;
    int      vocab_size;
    int      embedding_fd;
    int      embedding_dim;
    float16* embedding_data;
    struct stat emb_st{};
    caiwei::text::ChatTemplate chat_template;
    caiwei::text::SpecialToken special_token;
    std::vector<rknn3_tensor> media_input;
    std::vector<rknn3_tensor> media_output;
    std::vector<rknn3_tensor_mem*> internal_mems;
    std::vector<rknn3_tensor> output_tensors{};
public:
    bool load_model      (bool user_mem_internal = false);
    bool load_media_model(bool user_mem_internal = false);
    bool init_internal_mems(uint32_t core_mask_llm, uint32_t core_mask_media);
    bool init_output_tensors(int n_output_tensors);
    virtual std::vector<rknn3_llm_input> get_inputs(rknn3_session* session, ContextSession* context_session, caiwei::text::CompletionsRequest& request);
    rknn3_session* get_session(caiwei::text::CompletionsRequest* request, ContextSession* context_session);
    std::generator<caiwei::text::Result> generate(caiwei::text::CompletionsRequest& request);
public:
    RKNN3Context(std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path, int32_t max_token_length, caiwei::text::SpecialToken special_token);
    RKNN3Context(
        std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path,
        std::string media_model_path, std::string media_weight_path,
        int32_t max_token_length, caiwei::text::SpecialToken special_token
    );
    virtual ~RKNN3Context();
};

class ASRRKNN3Context : public ASRContext, public RKNN3Context {
private:
    std::vector<float16> embed;
    std::vector<float> mel_filters;
    std::vector<int32_t> prefix_token;
    std::vector<int32_t> suffix_token;
private:
    bool load_mel_filters();
    bool load_prompt_token();
public:
    bool load() override;
    std::vector<rknn3_llm_input> get_inputs(rknn3_session* session, ContextSession* context_session, caiwei::text::CompletionsRequest& request) override;
    std::generator<caiwei::text::Result> run(caiwei::text::CompletionsRequest& request) override;
public:
    ASRRKNN3Context(
        std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path,
        std::string media_model_path, std::string media_weight_path,
        int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime
    );
    ~ASRRKNN3Context();
};

class LLMRKNN3Context : public LLMContext, public RKNN3Context {
private:
    std::string prompt;
public:
    LLMRKNN3Context(std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime);
    ~LLMRKNN3Context();
public:
    bool load() override;
    std::vector<rknn3_llm_input> get_inputs(rknn3_session* session, ContextSession* context_session, caiwei::text::CompletionsRequest& request) override;
    std::generator<caiwei::text::Result> run(caiwei::text::CompletionsRequest& request) override;
};

class VLMRKNN3Context : public VLMContext, public RKNN3Context {
private:
    std::string prompt;
    std::vector<float16> image_embeds;
    std::vector<float16> video_embeds;
    int deepstack_count = 0;
    std::vector<rknn3_aux_tensor> deepstack_tensors;
private:
    bool fill_media(int i, int media_i, std::vector<float16>& embeds, const std::vector<uint8_t>& data, std::vector<rknn3_aux_tensor>& deepstack_tensors);
    bool init_deepstack_tensors(int count);
    bool release_deepstack_tensors();
public:
    bool load() override;
    std::vector<rknn3_llm_input> get_inputs(rknn3_session* session, ContextSession* context_session, caiwei::text::CompletionsRequest& request) override;
    std::generator<caiwei::text::Result> run(caiwei::text::CompletionsRequest& request) override;
public:
    VLMRKNN3Context(
        std::string model_path, std::string weight_path, std::string embedding_path, std::string tokenizer_path,
        std::string media_model_path, std::string media_weight_path,
        int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime
    );
    ~VLMRKNN3Context();
};

class EmbeddingRKNN3Context : public EmbeddingContext, public RKNN3Context {
public:
    bool load() override;
    caiwei::text::EmbeddingResult run(caiwei::text::EmbeddingsRequest& request) override;
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
    caiwei::text::RerankingResult run(caiwei::text::RerankingsRequest& request) override;
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

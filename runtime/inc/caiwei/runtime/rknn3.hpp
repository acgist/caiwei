#ifndef CAIWEI_RUNTIME_RKNN3_HPP
#define CAIWEI_RUNTIME_RKNN3_HPP

#include <mutex>
#include <generator>
#include <condition_variable>

#include "llama.h"

#include <sys/stat.h>

#include "rknn3/rknn3_api.h"

#include "caiwei/context.hpp"
#include "caiwei/text_tool.hpp"
#include "caiwei/tokenizer.hpp"

#ifndef CAIWEI_RKNN3_CORE_MASK
#define CAIWEI_RKNN3_CORE_MASK 0xFF
#endif

namespace caiwei  {
namespace context {

inline auto rknn3_session_deleter = [](rknn3_session* session) {
    if (session) {
        rknn3_session_destroy(session);
    }
};

using rknn3_session_ptr = std::unique_ptr<rknn3_session, decltype(rknn3_session_deleter)>;

struct alignas(16) ContextSession {
    bool stop     = false;
    bool thinking = false;
    bool toolcall = false;
    bool add_special   = false;
    bool parse_special = true;
    int32_t b_thinking;
    int32_t e_thinking;
    int32_t b_toolcall;
    int32_t e_toolcall;
    int audio_frames           = 0;
    int image_frames           = 0;
    uint32_t n_prefill_tokens  = 0;
    uint32_t n_generate_tokens = 0;
    float feature_ms           = 0;
    float prefill_ms           = 0;
    float generate_ms          = 0;
    std::mutex mutex;
    std::condition_variable condition;
    int        embedding_dim;
    float16  * embedding_data{ nullptr };
    Tokenizer* tokenizer     { nullptr };
    TokenBuffer token_buffer;
    caiwei::text::ToolcallSession toolcall_session;
    std::vector<std::vector<float>> pool_result{};
    std::vector<caiwei::text::CompletionResult> token_result{};
};

class RKNN3Context {
protected:
    const std::string model_path;
    const std::string weight_path;
    const std::string embedding_path;
    const std::string tokenizer_path;
    const std::string media_model_path;
    const std::string media_weight_path;
          int32_t     max_token_length;
    rknn3_context context       = 0;
    rknn3_context media_context = 0;
           int      vocab_size;
           int      embedding_fd = -1;
           int      embedding_dim;
           float16* embedding_data{ nullptr };
    struct stat     embedding_stat{};
    Tokenizer* tokenizer{ nullptr };
    caiwei::text::ChatTemplate chat_template;
    caiwei::text::SpecialToken special_token;
    std::vector<rknn3_tensor> media_input   {};
    std::vector<rknn3_tensor> media_output  {};
    std::vector<rknn3_tensor> output_tensors{};
    std::vector<rknn3_tensor_mem*> internal_tensor_mems{};
protected:
    bool load_model(rknn3_context& context, const std::string& model_path, const std::string& weight_path, bool user_mem_internal = false, uint32_t core_mask = CAIWEI_RKNN3_CORE_MASK);
    bool load_model      (bool user_mem_internal = false, uint32_t core_mask = CAIWEI_RKNN3_CORE_MASK);
    bool load_media_model(bool user_mem_internal = false, uint32_t core_mask = CAIWEI_RKNN3_CORE_MASK);
    bool load_tokenizer();
    bool load_embedding();
    bool init_media_tensors();
    void release_media_tensors();
    bool init_output_tensors(int n_output_tensors = 1);
    void release_output_tensors();
    bool init_internal_tensor_mems();
    void release_internal_tensor_mems();
    rknn3_session* get_session(ContextSession* context_session, const caiwei::text::CompletionsRequest* request);
    virtual std::vector<rknn3_llm_input> get_inputs(std::string& prompt, ContextSession* context_session, const caiwei::text::CompletionsRequest& request);
    std::vector<std::vector<float>> pooling(uint32_t& prompt_tokens, std::vector<const std::string*> prompt_list);
    std::generator<caiwei::text::CompletionResult> generate(const caiwei::text::CompletionsRequest& request);
public:
    RKNN3Context(
        std::string model_path,
        std::string weight_path,
        std::string embedding_path,
        std::string tokenizer_path,
        int32_t max_token_length,
        caiwei::text::SpecialToken special_token
    );
    RKNN3Context(
        std::string model_path,
        std::string weight_path,
        std::string embedding_path,
        std::string tokenizer_path,
        std::string media_model_path,
        std::string media_weight_path,
        int32_t max_token_length,
        caiwei::text::SpecialToken special_token
    );
    virtual ~RKNN3Context();
};

class ASRRKNN3Context : public ASRContext, public RKNN3Context {
private:
    std::vector<float>   mel_filters {};
    std::vector<float16> audio_embeds{};
    std::vector<int32_t> prefix_token{};
    std::vector<int32_t> suffix_token{};
private:
    bool load_mel_filters();
    bool init_prompt_token();
protected:
    std::vector<rknn3_llm_input> get_inputs(std::string& prompt, ContextSession* context_session, const caiwei::text::CompletionsRequest& request) override;
public:
    bool load() override;
    std::generator<caiwei::text::CompletionResult> run(const caiwei::text::CompletionsRequest& request) override;
public:
    ASRRKNN3Context(
        std::string model_path,
        std::string weight_path,
        std::string embedding_path,
        std::string tokenizer_path,
        std::string media_model_path,
        std::string media_weight_path,
        int32_t max_token_length,
        caiwei::text::SpecialToken special_token,
        caiwei::runtime::Runtime* runtime
    );
    ~ASRRKNN3Context();
};

class LLMRKNN3Context : public LLMContext, public RKNN3Context {
protected:
    std::vector<rknn3_llm_input> get_inputs(std::string& prompt, ContextSession* context_session, const caiwei::text::CompletionsRequest& request) override;
public:
    bool load() override;
    std::generator<caiwei::text::CompletionResult> run(const caiwei::text::CompletionsRequest& request) override;
public:
    LLMRKNN3Context(
        std::string model_path,
        std::string weight_path,
        std::string embedding_path,
        std::string tokenizer_path,
        int32_t max_token_length,
        caiwei::text::SpecialToken special_token,
        caiwei::runtime::Runtime* runtime
    );
    ~LLMRKNN3Context();
};

class VLMRKNN3Context : public VLMContext, public RKNN3Context {
private:
    std::vector<float16> image_embeds{};
    std::vector<float16> video_embeds{};
    std::vector<rknn3_aux_tensor> deepstack_tensors{};
private:
    bool fill_media(int media_i, int deepstack_i, std::vector<float16>& embeds, int width, int height, int channels, const std::vector<uint8_t>& data, std::vector<rknn3_aux_tensor>& deepstack_tensors);
    bool init_deepstack_tensors(int length, int size = 3);
protected:
    std::vector<rknn3_llm_input> get_inputs(std::string& prompt, ContextSession* context_session, const caiwei::text::CompletionsRequest& request) override;
public:
    bool load() override;
    std::generator<caiwei::text::CompletionResult> run(const caiwei::text::CompletionsRequest& request) override;
public:
    VLMRKNN3Context(
        std::string model_path,
        std::string weight_path,
        std::string embedding_path,
        std::string tokenizer_path,
        std::string media_model_path,
        std::string media_weight_path,
        int32_t max_token_length,
        caiwei::text::SpecialToken special_token,
        caiwei::runtime::Runtime* runtime
    );
    ~VLMRKNN3Context();
};

class EmbeddingRKNN3Context : public EmbeddingContext, public RKNN3Context {
public:
    bool load() override;
    caiwei::text::EmbeddingResult run(const caiwei::text::EmbeddingsRequest& request) override;
public:
    EmbeddingRKNN3Context(
        std::string model_path,
        std::string weight_path,
        std::string embedding_path,
        std::string tokenizer_path,
        int32_t max_token_length,
        caiwei::text::SpecialToken special_token,
        caiwei::runtime::Runtime* runtime
    );
    ~EmbeddingRKNN3Context();
};

class RerankingRKNN3Context : public RerankingContext, public RKNN3Context {
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
        caiwei::text::SpecialToken special_token,
        caiwei::runtime::Runtime* runtime
    );
    ~RerankingRKNN3Context();
};

void rknn3_dump_tensor_attr(const char* name, rknn3_tensor_attr* attr);
void rknn3_perf_session_print(rknn3_session* session, caiwei::context::ContextSession* context_session);

} // context
} // caiwei

#endif // CAIWEI_RUNTIME_RKNN3_HPP

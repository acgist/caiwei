#ifndef CAIWEI_RUNTIME_LLAMACPP_HPP
#define CAIWEI_RUNTIME_LLAMACPP_HPP

#include <generator>

#include "caiwei/context.hpp"
#include "caiwei/text_tool.hpp"

#include "mtmd.h"

#include "llama.h"

namespace caiwei  {
namespace context {

struct caiwei_llama_batch {
    bool success   = true;
    bool embedding = false;
    uint32_t    n_prompt_tokens = 0;
    llama_pos   n_pos;
    llama_batch batch;
    std::string finish_reason;
    caiwei_llama_batch() noexcept = default;
    ~caiwei_llama_batch();
    caiwei_llama_batch(const caiwei_llama_batch& batch) noexcept = delete;
    caiwei_llama_batch& operator=(const caiwei_llama_batch& batch) noexcept = delete;
    caiwei_llama_batch(caiwei_llama_batch&& batch) noexcept = default;
    caiwei_llama_batch& operator=(caiwei_llama_batch&& batch) noexcept = default;
};

class LlamaCPPContext {
protected:
    const std::string path;
    const std::string mtmd_path;
          int32_t     max_token_length;
          caiwei::text::ChatTemplate chat_template;
          caiwei::text::SpecialToken special_token;
          llama_model* model{ nullptr };
    const llama_vocab* vocab{ nullptr };
          mtmd_context* mtmd_model{ nullptr };
protected:
    llama_context* get_context(bool embeddings = false) const;
    llama_sampler* get_sampler(const caiwei::text::CompletionsRequest& request) const;
    virtual caiwei_llama_batch get_inputs(std::string& prompt, llama_context* context, const caiwei::text::CompletionsRequest& request) const;
    virtual bool get_mtmd_inputs(caiwei_llama_batch& caiwei_batch, mtmd::bitmaps& bitmaps, const std::string& prompt, llama_context* context, const caiwei::text::CompletionsRequest& request) const;
    std::generator<caiwei::text::CompletionResult> generate(const caiwei::text::CompletionsRequest& request) const;
public:
    LlamaCPPContext(std::string path,                        int32_t max_token_length, caiwei::text::SpecialToken special_token);
    LlamaCPPContext(std::string path, std::string mtmd_path, int32_t max_token_length, caiwei::text::SpecialToken special_token);
    virtual ~LlamaCPPContext();
public:
    bool load_model();
    bool load_mtmd_model();
};

class ASRLlamaCPPContext : public ASRContext, public LlamaCPPContext {
public:
    ASRLlamaCPPContext(
        std::string path,
        std::string mtmd_path,
        int32_t max_token_length,
        caiwei::text::SpecialToken special_token,
        caiwei::runtime::Runtime* runtime
    );
    ~ASRLlamaCPPContext();
protected:
    caiwei_llama_batch get_inputs(std::string& prompt, llama_context* context, const caiwei::text::CompletionsRequest& request) const;
public:
    bool load() override;
    std::generator<caiwei::text::CompletionResult> run(const caiwei::text::CompletionsRequest& request) override;
};

class LLMLlamaCPPContext : public LLMContext, public LlamaCPPContext {
public:
    LLMLlamaCPPContext(
        std::string path,
        int32_t max_token_length,
        caiwei::text::SpecialToken special_token,
        caiwei::runtime::Runtime* runtime
    );
    ~LLMLlamaCPPContext();
protected:
    caiwei_llama_batch get_inputs(std::string& prompt, llama_context* context, const caiwei::text::CompletionsRequest& request) const;
public:
    bool load() override;
    std::generator<caiwei::text::CompletionResult> run(const caiwei::text::CompletionsRequest& request) override;
};

class VLMLlamaCPPContext : public VLMContext, public LlamaCPPContext {
public:
    VLMLlamaCPPContext(
        std::string path,
        std::string mtmd_path,
        int32_t max_token_length,
        caiwei::text::SpecialToken special_token,
        caiwei::runtime::Runtime* runtime
    );
    ~VLMLlamaCPPContext();
protected:
    caiwei_llama_batch get_inputs(std::string& prompt, llama_context* context, const caiwei::text::CompletionsRequest& request) const;
public:
    bool load() override;
    std::generator<caiwei::text::CompletionResult> run(const caiwei::text::CompletionsRequest& request) override;
};

class EmbeddingLlamaCPPContext : public EmbeddingContext, public LlamaCPPContext {
public:
    EmbeddingLlamaCPPContext(
        std::string path,
        int32_t max_token_length,
        caiwei::text::SpecialToken special_token,
        caiwei::runtime::Runtime* runtime
    );
    ~EmbeddingLlamaCPPContext();
public:
    bool load() override;
    caiwei::text::EmbeddingResult run(const caiwei::text::EmbeddingsRequest& request) override;
};

class RerankingLlamaCPPContext : public RerankingContext, public LlamaCPPContext {
public:
    RerankingLlamaCPPContext(
        std::string path,
        int32_t max_token_length,
        caiwei::text::SpecialToken special_token,
        caiwei::runtime::Runtime* runtime
    );
    ~RerankingLlamaCPPContext();
public:
    bool load() override;
    caiwei::text::RerankingResult run(const caiwei::text::RerankingsRequest& request) override;
};

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text, bool add_special = true, bool parse_special = true);
void batch_add(llama_batch& batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id>& seq_id, bool logits);
void batch_decode_pooling(llama_context* ctx, llama_batch& batch, float* output, int n_seq, int n_embd_out);

} // context
} // caiwei

#endif // CAIWEI_RUNTIME_LLAMACPP_HPP

#ifndef CAIWEI_RUNTIME_LLAMACPP_HPP
#define CAIWEI_RUNTIME_LLAMACPP_HPP

#include "caiwei/context.hpp"
#include "caiwei/text_tool.hpp"

#include <generator>

struct llama_batch;
struct llama_model;
struct llama_vocab;
struct llama_context;
struct llama_sampler;

struct mtmd_context;

typedef int32_t llama_pos;
typedef int32_t llama_token;
typedef int32_t llama_seq_id;

namespace mtmd {

struct bitmaps;

}; // mtmd

namespace caiwei  {
namespace context {

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
    std::generator<caiwei::text::Result> generate     (const caiwei::text::CompletionsRequest& request) const;
    std::generator<caiwei::text::Result> generate_mtmd(const caiwei::text::CompletionsRequest& request) const;
    virtual bool build_mtmd(std::string& prompt, const caiwei::text::CompletionsRequest& request, mtmd::bitmaps& bitmaps) const;
public:
    LlamaCPPContext(std::string path,                        int32_t max_token_length, caiwei::text::SpecialToken special_token);
    LlamaCPPContext(std::string path, std::string mtmd_path, int32_t max_token_length, caiwei::text::SpecialToken special_token);
    virtual ~LlamaCPPContext();
public:
    bool load_model();
    bool load_mtmd();
};

class ASRLlamaCPPContext : public ASRContext, public LlamaCPPContext {
public:
    ASRLlamaCPPContext(std::string path, std::string mtmd_path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime);
    ~ASRLlamaCPPContext();
public:
    bool load() override;
    std::generator<caiwei::text::Result> run(const caiwei::text::CompletionsRequest& request) override;
    bool build_mtmd(std::string& prompt, const caiwei::text::CompletionsRequest& request, mtmd::bitmaps& bitmaps) const override;
};

class LLMLlamaCPPContext : public LLMContext, public LlamaCPPContext {
public:
    LLMLlamaCPPContext(std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime);
    ~LLMLlamaCPPContext();
public:
    bool load() override;
    std::generator<caiwei::text::Result> run(const caiwei::text::CompletionsRequest& request) override;
};

class VLMLlamaCPPContext : public VLMContext, public LlamaCPPContext {
public:
    VLMLlamaCPPContext(std::string path, std::string mtmd_path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime);
    ~VLMLlamaCPPContext();
public:
    bool load() override;
    std::generator<caiwei::text::Result> run(const caiwei::text::CompletionsRequest& request) override;
    bool build_mtmd(std::string& prompt, const caiwei::text::CompletionsRequest& request, mtmd::bitmaps& bitmaps) const override;
};

class EmbeddingLlamaCPPContext : public EmbeddingContext, public LlamaCPPContext {
public:
    EmbeddingLlamaCPPContext(std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime);
    ~EmbeddingLlamaCPPContext();
public:
    bool load() override;
    caiwei::text::EmbeddingResult run(const caiwei::text::EmbeddingsRequest& request) override;
};

class RerankingLlamaCPPContext : public RerankingContext, public LlamaCPPContext {
public:
    RerankingLlamaCPPContext(std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token, caiwei::runtime::Runtime* runtime);
    ~RerankingLlamaCPPContext();
public:
    bool load() override;
    caiwei::text::RerankingResult run(const caiwei::text::RerankingsRequest& request) override;
};

void batch_add(llama_batch& batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id>& seq_id, bool logits);
void batch_decode_pooling(llama_context* ctx, llama_batch& batch, float* output, int n_seq, int n_embd_out);

} // context
} // caiwei

#endif //CAIWEI_RUNTIME_LLAMACPP_HPP

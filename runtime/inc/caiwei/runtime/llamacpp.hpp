#ifndef CAIWEI_RUNTIME_LLAMACPP_HPP
#define CAIWEI_RUNTIME_LLAMACPP_HPP

#include "caiwei/log.hpp"
#include "caiwei/env.hpp"
#include "caiwei/context.hpp"
#include "caiwei/text_tool.hpp"

#include "mtmd.h"
#include "llama-cpp.h"

#include <generator>
#include <filesystem>

namespace caiwei  {
namespace context {

extern llama_token piece_to_token(const llama_vocab* vocab, const std::string& token);
extern std::string token_to_piece(const llama_vocab* vocab, llama_token token, std::string default_value = "");

class LlamaCPPContext {
protected:
    int32_t max_token_length;
    std::string path;
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    caiwei::text::ChatTemplate chat_template;
    caiwei::text::SpecialToken special_token;
protected:
    llama_context* get_context(bool embeddings = false);
    llama_sampler* get_sampler(const caiwei::text::CompletionsRequest& request);
    std::vector<llama_token> tokenize(const std::string& prompt, llama_context* context, bool add_special = true, bool parse_special = true);
    std::generator<caiwei::text::Result> generate(const caiwei::text::CompletionsRequest& request);
public:
    LlamaCPPContext(std::string path, int32_t max_token_length, caiwei::text::SpecialToken special_token);
    ~LlamaCPPContext();
public:
    bool load_model();
};

class ASRLlamaCPPContext : public ASRContext, public LlamaCPPContext {
public:
    bool load() override;
    std::generator<caiwei::text::Result> run(const caiwei::text::CompletionsRequest& request) override;
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
    bool load() override;
    std::generator<caiwei::text::Result> run(const caiwei::text::CompletionsRequest& request) override;
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
private:
    std::string bos_key;
    std::string eos_key;
    std::string system_prompt;
    std::string instruction_prompt;
    std::string instruction_key;
    std::string query_key;
    std::string document_key;
public:
    RerankingLlamaCPPContext(
        std::string path,
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
    ~RerankingLlamaCPPContext();
public:
    bool load() override;
    caiwei::text::RerankingResult run(const caiwei::text::RerankingsRequest& request) override;
};

} // context
} // caiwei

#endif //CAIWEI_RUNTIME_LLAMACPP_HPP

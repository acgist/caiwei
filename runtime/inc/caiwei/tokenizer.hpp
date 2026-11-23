#ifndef CAIWEI_RUNTIME_TOKENIZER_HPP
#define CAIWEI_RUNTIME_TOKENIZER_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "llama.h"

namespace caiwei {
namespace context {

// clang-format off
llama_token piece_to_token(const llama_vocab* vocab, const std::string& piece, bool add_special = false, bool parse_special = true);
std::string token_to_piece(const llama_vocab* vocab, llama_token token, bool special = true);
int32_t token_to_piece(const llama_vocab* vocab, llama_token token, char* buf, int32_t len, bool special = true);
int32_t tokenize(const llama_vocab* vocab, const char* text, int32_t length, llama_token* tokens, int32_t n_tokens, bool add_special = false, bool parse_special = true);
std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text, bool add_special = false, bool parse_special = true);
std::string detokenize(const llama_vocab* vocab, const llama_token* tokens, int32_t n_tokens, bool remove_special = false, bool unparse_special = false);
// clang-format on

class TokenBuffer {
private:
    int pos = 0;
    int len = 0;
    std::string buffer;
public:
    TokenBuffer();
    TokenBuffer(const TokenBuffer&) = delete;
    TokenBuffer& operator=(const TokenBuffer&) = delete;
    TokenBuffer(TokenBuffer&&) = delete;
    TokenBuffer& operator=(TokenBuffer&&) = delete;
public:
    int get_length();
    char* get_buffer();
    int resize(int length);
    std::string get_token(int length);
};

class Tokenizer {
private:
    // clang-format off
          llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    // clang-format on
public:
    Tokenizer();
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) = delete;
    Tokenizer& operator=(Tokenizer&&) = delete;
private:
    void release();
public:
    bool load(const std::string& path);
    bool is_eog(llama_token token_id) const;
    int32_t get_size() const;
    llama_token get_nl() const;
    llama_token get_bos() const;
    llama_token get_eos() const;
    // clang-format off
    llama_token piece_to_token(const std::string& piece, bool add_special = false, bool parse_special = true);
    std::string token_to_piece(llama_token token, bool special = true);
    int32_t token_to_piece(llama_token token, char* buf, int32_t len, bool special = true);
    int32_t tokenize(const char* text, int32_t length, llama_token* tokens, int32_t n_tokens, bool add_special = false, bool parse_special = true);
    std::vector<llama_token> tokenize(const std::string& text, bool add_special = false, bool parse_special = true);
    std::string detokenize(const llama_token* tokens, int32_t n_tokens, bool remove_special = false, bool unparse_special = false);
    // clang-format on
};

} // namespace context
} // namespace caiwei

#endif // CAIWEI_RUNTIME_TOKENIZER_HPP

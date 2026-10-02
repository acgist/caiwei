#ifndef CAIWEI_RUNTIME_TOKENIZER_HPP
#define CAIWEI_RUNTIME_TOKENIZER_HPP

#include <string>
#include <vector>
#include <cstdint>

#include "llama-cpp.h"

namespace caiwei  {
namespace context {

llama_token piece_to_token(const llama_vocab* vocab, const std::string& token, bool add_special = false, bool parse_special = true);
std::string token_to_piece(const llama_vocab* vocab, llama_token token, std::string default_value = "", bool special = true);
std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text, bool add_special = true, bool parse_special = true);
std::string detokenize(const llama_vocab* vocab, llama_token* tokens, int32_t n_tokens, bool remove_special = false, bool unparse_special = false);

class Tokenizer {
private:
          llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
public:
    Tokenizer(const std::string& path);
    ~Tokenizer();
public:
    bool    is_eog(int32_t token_id);
    int32_t get_size();
    int32_t get_bos();
    int32_t get_eos();
    int32_t get_eot();
    int32_t get_pad();
    int32_t get_nl();
    int32_t piece_to_token(const std::string& piece, bool add_special = false, bool parse_special = true);
    std::string token_to_piece(int32_t token, std::string default_value = "", bool special = true);
    int tokenize(const char* text, int32_t length, int32_t* tokens, int32_t n_tokens, bool add_special = true, bool parse_special = true);
    std::string detokenize(int32_t* tokens, int32_t n_tokens, bool remove_special = false, bool unparse_special = false);
};

} // context
} // caiwei

#endif // CAIWEI_RUNTIME_TOKENIZER_HPP

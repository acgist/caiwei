#ifndef CAIWEI_RUNTIME_TOKENIZER_HPP
#define CAIWEI_RUNTIME_TOKENIZER_HPP

#include <string>
#include <cstdint>

#include "llama-cpp.h"

namespace caiwei  {
namespace context {

class Tokenizer {
private:
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
public:
    Tokenizer(const std::string& path);
    ~Tokenizer();
public:
    int32_t get_size();
    int32_t get_bos();
    int32_t get_eos();
    int32_t get_pad();
    int32_t piece_to_token(const std::string& piece);
    std::string token_to_piece(int32_t token);
    int tokenize(const char* text, int32_t length, int32_t* tokens, int32_t n_tokens);
    std::string detokenize(int32_t* tokens, int32_t n_tokens);
};

}
}

#endif

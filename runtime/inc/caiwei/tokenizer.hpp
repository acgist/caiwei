#ifndef CAIWEI_RUNTIME_TOKENIZER_HPP
#define CAIWEI_RUNTIME_TOKENIZER_HPP

#include <string>
#include <vector>
#include <cstdint>

typedef int32_t llama_pos;
typedef int32_t llama_token;
typedef int32_t llama_seq_id;

struct llama_model;
struct llama_vocab;

namespace caiwei  {
namespace context {

llama_token piece_to_token(const llama_vocab* vocab, const std::string& piece, bool add_special = false, bool parse_special = true);
std::string token_to_piece(const llama_vocab* vocab, llama_token token, std::string default_value = "", bool special = true);
int tokenize(const llama_vocab* vocab, const char* text, int32_t length, llama_token* tokens, int32_t n_tokens, bool add_special = true, bool parse_special = true);
std::string detokenize(const llama_vocab* vocab, llama_token* tokens, int32_t n_tokens, bool remove_special = false, bool unparse_special = false);

class Tokenizer {
private:
          llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
public:
    Tokenizer(const std::string& path);
    ~Tokenizer();
public:
    bool is_eog(llama_token token_id);
    int32_t get_size();
    llama_token get_nl();
    llama_token get_bos();
    llama_token get_eos();
    llama_token get_eot();
    llama_token get_pad();
    llama_token piece_to_token(const std::string& piece, bool add_special = false, bool parse_special = true);
    std::string token_to_piece(llama_token token, std::string default_value = "", bool special = true);
    int tokenize(const char* text, int32_t length, llama_token* tokens, int32_t n_tokens, bool add_special = true, bool parse_special = true);
    std::string detokenize(llama_token* tokens, int32_t n_tokens, bool remove_special = false, bool unparse_special = false);
};

void euclidean(const float* embd, float* out, int size);

} // context
} // caiwei

#endif // CAIWEI_RUNTIME_TOKENIZER_HPP

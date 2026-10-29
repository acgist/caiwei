#include <cmath>

#include "llama.h"

#include "caiwei/log.hpp"
#include "caiwei/tokenizer.hpp"

caiwei::context::Tokenizer::Tokenizer() {
}

caiwei::context::Tokenizer::~Tokenizer() {
    if (this->model) {
        CW_LOG_I("释放分词模型");
        llama_model_free(this->model);
        this->model = nullptr;
    }
}

bool caiwei::context::Tokenizer::load(const std::string& path) {
    if (this->model != nullptr) {
        CW_LOG_I("释放旧的分词模型");
        llama_model_free(this->model);
        this->model = nullptr;
    }
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    this->model = llama_model_load_from_file(path.c_str(), params);
    if (this->model == nullptr) {
        CW_LOG_W("分词模型加载失败: %s", path.c_str());
        return false;
    }
    this->vocab = llama_model_get_vocab(this->model);
    if (this->vocab == nullptr) {
        CW_LOG_W("分词模型加载失败: %s", path.c_str());
        return false;
    }
    CW_LOG_I("分词模型加载成功: %s", path.c_str());
    return true;
}

bool caiwei::context::Tokenizer::is_eog(llama_token token_id) {
    return llama_vocab_is_eog(this->vocab, token_id);
}

int32_t caiwei::context::Tokenizer::get_size() {
    return llama_vocab_n_tokens(this->vocab);
}

llama_token caiwei::context::Tokenizer::get_nl() {
    return llama_vocab_nl(this->vocab);
}

llama_token caiwei::context::Tokenizer::get_bos() {
    return llama_vocab_bos(this->vocab);
}

llama_token caiwei::context::Tokenizer::get_eos() {
    return llama_vocab_eos(this->vocab);
}

llama_token caiwei::context::Tokenizer::get_eot() {
    return llama_vocab_eot(this->vocab);
}

llama_token caiwei::context::Tokenizer::get_pad() {
    return llama_vocab_pad(this->vocab);
}

llama_token caiwei::context::Tokenizer::piece_to_token(const std::string& piece, bool add_special, bool parse_special) {
    return caiwei::context::piece_to_token(this->vocab, piece, add_special, parse_special);
}

std::string caiwei::context::Tokenizer::token_to_piece(llama_token token, bool special) {
    return caiwei::context::token_to_piece(this->vocab, token, special);
}

int caiwei::context::Tokenizer::token_to_piece(llama_token token, std::string& piece, bool special) {
    return caiwei::context::token_to_piece(this->vocab, token, piece, special);
}

int caiwei::context::Tokenizer::tokenize(const char* text, int32_t length, llama_token* tokens, int32_t n_tokens, bool add_special, bool parse_special) {
    return caiwei::context::tokenize(this->vocab, text, length, tokens, n_tokens, add_special, parse_special);
}

std::vector<llama_token> caiwei::context::Tokenizer::tokenize(const std::string& text, bool add_special, bool parse_special) {
    return caiwei::context::tokenize(this->vocab, text, add_special, parse_special);
}

std::string caiwei::context::Tokenizer::detokenize(llama_token* tokens, int32_t n_tokens, bool remove_special, bool unparse_special) {
    return caiwei::context::detokenize(this->vocab, tokens, n_tokens, remove_special, unparse_special);
}

llama_token caiwei::context::piece_to_token(const llama_vocab* vocab, const std::string& piece, bool add_special, bool parse_special) {
    llama_token token{};
    if (llama_tokenize(vocab, piece.c_str(), piece.size(), &token, 1, add_special, parse_special) < 0) {
        CW_LOG_W("编码失败: %s", piece.c_str());
        return LLAMA_TOKEN_NULL;
    }
    return token;
}

std::string caiwei::context::token_to_piece(const llama_vocab* vocab, llama_token token, bool special) {
    if (token == LLAMA_TOKEN_NULL) {
        return "";
    }
    std::string piece;
    piece.resize(16);
    const int length = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, special);
    if (length < 0) {
        piece.resize(-length);
        if (llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, special) != -length) {
            CW_LOG_W("解码失败: %d", length);
            return "";
        }
    } else {
        piece.resize(length);
    }
    return piece;
}

int caiwei::context::token_to_piece(const llama_vocab* vocab, llama_token token, std::string& piece, bool special) {
    return llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, special);
}

int caiwei::context::tokenize(const llama_vocab* vocab, const char* text, int32_t length, llama_token* tokens, int32_t n_tokens, bool add_special, bool parse_special) {
    return llama_tokenize(vocab, text, length, tokens, n_tokens, add_special, parse_special);
}

std::vector<llama_token> caiwei::context::tokenize(const llama_vocab* vocab, const std::string& text, bool add_special, bool parse_special) {
    if (text.empty()) {
        return {};
    }
    const int n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, parse_special);
    std::vector<llama_token> tokens(n_tokens);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_special, parse_special) < 0) {
        CW_LOG_W("编码失败: %s", text.c_str());
        return {};
    }
    return tokens;
}

std::string caiwei::context::detokenize(const llama_vocab* vocab, llama_token* tokens, int32_t n_tokens, bool remove_special, bool unparse_special) {
    std::string text;
    text.resize(n_tokens);
    const int length = llama_detokenize(vocab, tokens, n_tokens, text.data(), text.size(), remove_special, unparse_special);
    if (length < 0) {
        text.resize(-length);
        if (llama_detokenize(vocab, tokens, n_tokens, text.data(), text.size(), remove_special, unparse_special) != -length) {
            CW_LOG_W("解码失败: %d", length);
            return "";
        }
    } else {
        text.resize(length);
    }
    return text;
}

void caiwei::context::euclidean(const float* embd, float* out, int size) {
    if (size < 0) {
        return;
    }
    if (size == 1) {
        if (embd == out) {
            return;
        }
        out[0] = embd[0];
        return;
    }
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += embd[i] * embd[i];
    }
    sum = std::sqrt(sum);
    const float norm = sum > 0.0 ? 1.0F / sum : 0.0F;
    for (int i = 0; i < size; i++) {
        out[i] = embd[i] * norm;
    }
}

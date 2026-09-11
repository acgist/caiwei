#include "caiwei/runtime/tokenizer.hpp"

#include "caiwei/log.hpp"

caiwei::context::Tokenizer::Tokenizer(const std::string& path) {
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    this->model = llama_model_load_from_file(path.c_str(), params);
    if (this->model == nullptr) {
        CW_LOG_W("加载Tokenizer模型失败: %s", path.c_str());
        return;
    }
    this->vocab = llama_model_get_vocab(this->model);
}

caiwei::context::Tokenizer::~Tokenizer() {
    if (this->model) {
        llama_model_free(this->model);
        this->model = nullptr;
    }
}

int32_t caiwei::context::Tokenizer::get_size() {
    return llama_vocab_n_tokens(this->vocab);
}

int32_t caiwei::context::Tokenizer::get_bos() {
    return llama_vocab_bos(this->vocab);
}

int32_t caiwei::context::Tokenizer::get_eos() {
    return llama_vocab_eos(this->vocab);
}

int32_t caiwei::context::Tokenizer::get_pad() {
    return llama_vocab_pad(this->vocab);
}

int32_t caiwei::context::Tokenizer::piece_to_token(const std::string& piece) {
    llama_token token;
    if (llama_tokenize(this->vocab, piece.c_str(), piece.size(), &token, 1, false, true) < 0) {
        CW_LOG_W("piece_to_token失败: %s", piece.c_str());
        return LLAMA_TOKEN_NULL;
    }
    return token;
}

std::string caiwei::context::Tokenizer::token_to_piece(int32_t token) {
    if (token == LLAMA_TOKEN_NULL) {
        return "";
    }
    std::string piece;
    piece.resize(64);
    const int length = llama_token_to_piece(this->vocab, token, piece.data(), piece.size(), 0, true);
    if (length < 0) {
        piece.resize(-length);
        if (llama_token_to_piece(this->vocab, token, piece.data(), piece.size(), 0, true) != -length) {
            return "";
        }
    } else {
        piece.resize(length);
    }
    return piece;
}

int caiwei::context::Tokenizer::tokenize(const char* text, int32_t length, int32_t* tokens, int32_t n_tokens) {
    return llama_tokenize(this->vocab, text, length, reinterpret_cast<llama_token*>(tokens), n_tokens, true, true);
}

std::string caiwei::context::Tokenizer::detokenize(int32_t* tokens, int32_t n_tokens) {
    std::string text;
    text.resize(n_tokens);
    const int length = llama_detokenize(this->vocab, reinterpret_cast<llama_token*>(tokens), n_tokens, text.data(), text.size(), false, false);
    if (length < 0) {
        text.resize(-length);
        if (llama_detokenize(this->vocab, reinterpret_cast<llama_token*>(tokens), n_tokens, text.data(), text.size(), false, false) != -length) {
            return "";
        }
    } else {
        text.resize(length);
    }
    return text;
}

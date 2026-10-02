#include "caiwei/log.hpp"
#include "caiwei/tokenizer.hpp"

caiwei::context::Tokenizer::Tokenizer(const std::string& path) {
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    this->model = llama_model_load_from_file(path.c_str(), params);
    if (this->model == nullptr) {
        CW_LOG_W("加载Tokenizer模型失败: %s", path.c_str());
        return;
    }
    // TODO
    // llama_model_chat_template
    this->vocab = llama_model_get_vocab(this->model);
}

caiwei::context::Tokenizer::~Tokenizer() {
    if (this->model) {
        llama_model_free(this->model);
        this->model = nullptr;
    }
}

bool caiwei::context::Tokenizer::is_eog(int32_t token_id) {
    return llama_vocab_is_eog(this->vocab, token_id);
}

int32_t caiwei::context::Tokenizer::get_nl() {
    return llama_vocab_nl(this->vocab);
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

int32_t caiwei::context::Tokenizer::get_eot() {
    return llama_vocab_eot(this->vocab);
}

int32_t caiwei::context::Tokenizer::get_pad() {
    return llama_vocab_pad(this->vocab);
}

int32_t caiwei::context::Tokenizer::piece_to_token(const std::string& piece, bool add_special, bool parse_special) {
    return caiwei::context::piece_to_token(this->vocab, piece, add_special, parse_special);
}

std::string caiwei::context::Tokenizer::token_to_piece(int32_t token, std::string default_value, bool special) {
    return caiwei::context::token_to_piece(this->vocab, static_cast<llama_token>(token), std::move(default_value), special);
}

int caiwei::context::Tokenizer::tokenize(const char* text, int32_t length, int32_t* tokens, int32_t n_tokens, bool add_special, bool parse_special) {
    return llama_tokenize(this->vocab, text, length, reinterpret_cast<llama_token*>(tokens), n_tokens, add_special, parse_special);
}

std::string caiwei::context::Tokenizer::detokenize(int32_t* tokens, int32_t n_tokens, bool remove_special, bool unparse_special) {
    return caiwei::context::detokenize(this->vocab, reinterpret_cast<llama_token*>(tokens), n_tokens, remove_special, unparse_special);
}

llama_token caiwei::context::piece_to_token(const llama_vocab* vocab, const std::string& piece, bool add_special, bool parse_special) {
    llama_token token;
    if (llama_tokenize(vocab, piece.c_str(), piece.size(), &token, 1, add_special, parse_special) < 0) {
        CW_LOG_W("编码失败: %s", piece.c_str());
        return LLAMA_TOKEN_NULL;
    }
    return token;
}

std::string caiwei::context::token_to_piece(const llama_vocab* vocab, llama_token token, std::string default_value, bool special) {
    if (token == LLAMA_TOKEN_NULL) {
        return std::move(default_value);
    }
    std::string piece;
    piece.resize(piece.capacity());
    const int length = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, special);
    if (length < 0) {
        piece.resize(-length);
        if (llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, special) != -length) {
            CW_LOG_W("解码失败: %d", length);
            return std::move(default_value);
        }
    } else {
        piece.resize(length);
    }
    return piece;
}

std::vector<llama_token> caiwei::context::tokenize(const llama_vocab* vocab, const std::string& text, bool add_special, bool parse_special) {
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
    const int length = llama_detokenize(vocab, reinterpret_cast<llama_token*>(tokens), n_tokens, text.data(), text.size(), remove_special, unparse_special);
    if (length < 0) {
        text.resize(-length);
        if (llama_detokenize(vocab, reinterpret_cast<llama_token*>(tokens), n_tokens, text.data(), text.size(), remove_special, unparse_special) != -length) {
            CW_LOG_W("解码失败: %d", length);
            return "";
        }
    } else {
        text.resize(length);
    }
    return text;
}

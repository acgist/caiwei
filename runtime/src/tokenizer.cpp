#include <cmath>

#include "llama.h"

#include "caiwei/env.hpp"
#include "caiwei/log.hpp"
#include "caiwei/tokenizer.hpp"

inline static int buffer_size(uint8_t v) {
    if (v < 0x80)           return 1;
    if ((v & 0xE0) == 0xC0) return 2;
    if ((v & 0xF0) == 0xE0) return 3;
    if ((v & 0xF8) == 0xF0) return 4;
    return 0;
}

caiwei::context::TokenBuffer::TokenBuffer() {
    this->buffer.resize(this->len);
}

int caiwei::context::TokenBuffer::get_length() {
    return this->len - this->pos;
}

char* caiwei::context::TokenBuffer::get_buffer() {
    return this->buffer.data() + this->pos;
}

std::string caiwei::context::TokenBuffer::get_token(int length) {
    if (length == 0) {
        return "";
    }
    if (length < 0 || this->pos + length > this->len) {
        // 该分支理论上绝对不会执行
        CW_LOG_W("长度错误: %d + %d > %d", this->pos, length, this->len);
        return "";
    }
    int i = 0;
    this->pos += length;
    const uint8_t* buf = reinterpret_cast<const uint8_t*>(this->buffer.data());
    while (i < this->pos) {
        const int size = buffer_size(buf[i]);
        if (size == 0) {
            CW_LOG_W("非法字节: %d", buf[i]);
            std::copy(buf + i + 1, buf + this->pos, this->buffer.begin() + i);
            --this->pos;
            continue;
        }
        if (i + size > this->pos) {
            break;
        }
        bool verify = true;
        for (int k = 1; k < size; ++k) {
            if ((buf[i + k] & 0xC0) != 0x80) {
                CW_LOG_W("非法字节: %d", buf[i + k]);
                verify = false;
                break;
            }
        }
        if (verify) {
            i += size;
        } else {
            std::copy(buf + i + 1, buf + this->pos, this->buffer.begin() + i);
            --this->pos;
        }
    }
    if (i == 0) {
        if (this->pos >= this->len) {
            CW_LOG_W("缓冲已满");
            std::copy(buf + 1, buf + this->pos, this->buffer.begin());
            --this->pos;
        }
        return "";
    }
    std::string ret = std::string(buffer.data(), i);
    if (i < this->pos) {
        std::copy(buf + i, buf + this->pos, this->buffer.begin());
    }
    this->pos -= i;
    return ret;
}

std::string caiwei::context::TokenBuffer::flush() {
    if (this->pos == 0) {
        return "";
    }
    std::string ret = std::string(buffer.data(), this->pos);
    this->pos = 0;
    return ret;
}

caiwei::context::Tokenizer::Tokenizer() {
}

caiwei::context::Tokenizer::~Tokenizer() {
    this->release();
}

void caiwei::context::Tokenizer::release() {
    if (this->vocab) {
        CW_LOG_I("释放分词模型vocab");
        this->vocab = nullptr;
    }
    if (this->model) {
        CW_LOG_I("释放分词模型");
        llama_model_free(this->model);
        this->model = nullptr;
    }
}

bool caiwei::context::Tokenizer::load(const std::string& path) {
    this->release();
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    this->model = llama_model_load_from_file(path.c_str(), params);
    if (this->model == nullptr) {
        CW_LOG_W("分词模型加载失败: %s", path.c_str());
        this->release();
        return false;
    }
    CW_LOG_I("分词模型加载成功: %s", path.c_str());
    this->vocab = llama_model_get_vocab(this->model);
    if (this->vocab == nullptr) {
        CW_LOG_W("分词模型vocab加载失败: %s", path.c_str());
        this->release();
        return false;
    }
    CW_LOG_I("分词模型vocab加载成功: %s", path.c_str());
    return true;
}

bool caiwei::context::Tokenizer::is_eog(llama_token token_id) {
    caiwei::env::check_not_nullptr(this->vocab);
    return llama_vocab_is_eog(this->vocab, token_id);
}

int32_t caiwei::context::Tokenizer::get_size() {
    caiwei::env::check_not_nullptr(this->vocab);
    return llama_vocab_n_tokens(this->vocab);
}

llama_token caiwei::context::Tokenizer::get_nl() {
    caiwei::env::check_not_nullptr(this->vocab);
    return llama_vocab_nl(this->vocab);
}

llama_token caiwei::context::Tokenizer::get_bos() {
    caiwei::env::check_not_nullptr(this->vocab);
    return llama_vocab_bos(this->vocab);
}

llama_token caiwei::context::Tokenizer::get_eos() {
    caiwei::env::check_not_nullptr(this->vocab);
    return llama_vocab_eos(this->vocab);
}

llama_token caiwei::context::Tokenizer::get_eot() {
    caiwei::env::check_not_nullptr(this->vocab);
    return llama_vocab_eot(this->vocab);
}

llama_token caiwei::context::Tokenizer::get_pad() {
    caiwei::env::check_not_nullptr(this->vocab);
    return llama_vocab_pad(this->vocab);
}

llama_token caiwei::context::Tokenizer::piece_to_token(const std::string& piece, bool add_special, bool parse_special) {
    return caiwei::context::piece_to_token(this->vocab, piece, add_special, parse_special);
}

std::string caiwei::context::Tokenizer::token_to_piece(llama_token token, bool special) {
    return caiwei::context::token_to_piece(this->vocab, token, special);
}

int caiwei::context::Tokenizer::token_to_piece(llama_token token, char* buf, int len, bool special) {
    return caiwei::context::token_to_piece(this->vocab, token, buf, len, special);
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
    if (piece.empty()) {
        return LLAMA_TOKEN_NULL;
    }
    caiwei::env::check_not_nullptr(vocab);
    llama_token token{};
    const int n_tokens = llama_tokenize(vocab, piece.c_str(), piece.size(), &token, 1, add_special, parse_special);
    if (n_tokens != 1) {
        CW_LOG_W("编码失败: %s = %d", piece.c_str(), n_tokens);
        return LLAMA_TOKEN_NULL;
    }
    return token;
}

std::string caiwei::context::token_to_piece(const llama_vocab* vocab, llama_token token, bool special) {
    if (token == LLAMA_TOKEN_NULL) {
        return "";
    }
    caiwei::env::check_not_nullptr(vocab);
    std::string piece;
    piece.resize(16);
    const int length = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, special);
    if (length < 0) {
        piece.resize(-length);
        if (llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, special) != -length) {
            CW_LOG_W("解码失败: %d = %d", token, length);
            return "";
        }
    } else {
        piece.resize(length);
    }
    return piece;
}

int caiwei::context::token_to_piece(const llama_vocab* vocab, llama_token token, char* buf, int len, bool special) {
    caiwei::env::check_not_nullptr(vocab);
    return llama_token_to_piece(vocab, token, buf, len, 0, special);
}

int caiwei::context::tokenize(const llama_vocab* vocab, const char* text, int32_t length, llama_token* tokens, int32_t n_tokens, bool add_special, bool parse_special) {
    caiwei::env::check_not_nullptr(vocab);
    return llama_tokenize(vocab, text, length, tokens, n_tokens, add_special, parse_special);
}

std::vector<llama_token> caiwei::context::tokenize(const llama_vocab* vocab, const std::string& text, bool add_special, bool parse_special) {
    if (text.empty()) {
        return {};
    }
    caiwei::env::check_not_nullptr(vocab);
    const int n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, parse_special);
    if (n_tokens <= 0) {
        CW_LOG_W("编码失败: %s", text.c_str());
        return {};
    }
    std::vector<llama_token> tokens(n_tokens);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_special, parse_special) < 0) {
        CW_LOG_W("编码失败: %s", text.c_str());
        return {};
    }
    return tokens;
}

std::string caiwei::context::detokenize(const llama_vocab* vocab, llama_token* tokens, int32_t n_tokens, bool remove_special, bool unparse_special) {
    if (tokens == nullptr || n_tokens <= 0) {
        return "";
    }
    caiwei::env::check_not_nullptr(vocab);
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

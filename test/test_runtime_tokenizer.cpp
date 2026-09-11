#include "test.hpp"

#include "caiwei/runtime/tokenizer.hpp"

int main() {
    caiwei::context::Tokenizer tokenizer("Qwen3-Embedding-0.6B.tokenizer.gguf");
    int32_t bos = tokenizer.get_bos();
    int32_t eos = tokenizer.get_eos();
    int32_t pad = tokenizer.get_pad();
    std::string bos_piece = tokenizer.token_to_piece(bos);
    std::string eos_piece = tokenizer.token_to_piece(eos);
    std::string pad_piece = tokenizer.token_to_piece(pad);
    int bthinking = tokenizer.piece_to_token("<think>");
    int ethinking = tokenizer.piece_to_token("</think>");
    int btoolcall = tokenizer.piece_to_token("<tool_call>");
    int etoolcall = tokenizer.piece_to_token("</tool_call>");
    assert(bos == tokenizer.piece_to_token(bos_piece));
    assert(eos == tokenizer.piece_to_token(eos_piece));
    assert(pad == tokenizer.piece_to_token(pad_piece));
    assert(bthinking == 151667);
    assert(ethinking == 151668);
    assert(btoolcall == 151657);
    assert(etoolcall == 151658);
    std::string piece = tokenizer.token_to_piece(151644);
    assert(piece == "<|im_start|>");
    std::string prompt = "<|im_start|>你好这是一个测试<think>think</think><tool_call>tool_call</tool_call><|im_end|>";
    std::vector<int32_t> tokens(64);
    int length = tokenizer.tokenize(prompt.data(), prompt.size(), tokens.data(), tokens.size());
    std::string text = tokenizer.detokenize(tokens.data(), length);
    assert(text == "你好这是一个测试<think>think</think><tool_call>tool_call</tool_call>");
    return 0;
}
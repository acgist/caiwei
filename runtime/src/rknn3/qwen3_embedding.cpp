#include "caiwei/runtime/rknn3.hpp"

caiwei::context::EmbeddingRKNN3Context::EmbeddingRKNN3Context() {
    
}

caiwei::context::EmbeddingRKNN3Context::~EmbeddingRKNN3Context() {

}

std::vector<rknn3_llm_input> caiwei::context::EmbeddingRKNN3Context::get_inputs(rknn3_session* session, const caiwei::text::CompletionsRequest& request) {
    rknn3_llm_tensor tensor;
    // LLM Input
    tensor.name = "input";
    tensor.prompt = "formatted_prompt";
    tensor.embed = NULL;
    tensor.tokens = NULL;
    tensor.n_tokens = 0;
    tensor.enable_thinking = false;
    std::vector<rknn3_llm_input> inputs(1);
    inputs[0].input_type = RKNN3_LLM_INPUT_PROMPT;
    inputs[0].llm_input  = tensor;
    return inputs;
}

std::generator<std::string> caiwei::context::EmbeddingRKNN3Context::run(const caiwei::text::CompletionsRequest& request) {
    // npy_shape[0] = output_tensors[0].attr->n_elems;
    // npy_save_float_buffer_to_file("output.npy", model_output, output_tensors[0].attr->n_elems, npy_shape, npy_ndim);
    return {};
}

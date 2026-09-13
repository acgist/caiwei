#include "caiwei/runtime/rknn3.hpp"

static const char *DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query";

static const char *RERANKER_PREFIX = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
static const char *RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n";

static std::string format_reranker_prompt(const char *instruction, const char *query, const char *doc)
{
    const char *inst = instruction ? instruction : DEFAULT_INSTRUCTION;
    return std::string(RERANKER_PREFIX) +
           "<Instruct>: " + inst +
           "\n<Query>: " + query +
           "\n<Document>: " + doc +
           "\n" + RERANKER_SUFFIX;
}

caiwei::context::RerankingRKNN3Context::RerankingRKNN3Context() {
    
}

caiwei::context::RerankingRKNN3Context::~RerankingRKNN3Context() {

}

std::vector<rknn3_llm_input> caiwei::context::RerankingRKNN3Context::get_inputs(rknn3_session* session, const caiwei::text::CompletionsRequest& request) {
    auto formatted_prompt = format_reranker_prompt(DEFAULT_INSTRUCTION, "query", "document");

    rknn3_llm_tensor tensor;
    // LLM Input
    tensor.name = "input";
    tensor.prompt = formatted_prompt.c_str();
    tensor.embed = NULL;
    tensor.tokens = NULL;
    tensor.n_tokens = 0;
    tensor.enable_thinking = false;
    std::vector<rknn3_llm_input> inputs(1);
    inputs[0].input_type = RKNN3_LLM_INPUT_PROMPT;
    inputs[0].llm_input  = tensor;
    return inputs;
}

std::generator<std::string> caiwei::context::RerankingRKNN3Context::run(const caiwei::text::CompletionsRequest& request) {
    auto reranker_score = model_output[0];
    return {};
}

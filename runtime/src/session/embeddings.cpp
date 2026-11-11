#include "caiwei/env.hpp"
#include "caiwei/manager.hpp"
#include "caiwei/session.hpp"

caiwei::session::EmbeddingsSession::EmbeddingsSession(caiwei::text::EmbeddingsRequest& request) : request(request) {
}

std::string caiwei::session::EmbeddingsSession::get() {
    auto ptr = caiwei::manager::get_context<caiwei::context::EmbeddingContext>(this->request.model);
    caiwei::env::check_not_nullptr(ptr.get(), "模型无效");
    auto result = ptr->run(this->request);
    caiwei::text::EmbeddingsResponse response;
    response.model = request.model;
    size_t input_size = 0;
    if (std::holds_alternative<std::string>(this->request.input)) {
        input_size = 1;
    } else {
        input_size = std::get<std::vector<std::string>>(request.input).size();
    }
    caiwei::env::check_eq(input_size, result.result.size(), "推理失败");
    for (size_t i = 0; i < result.result.size(); ++i) {
        caiwei::text::EmbeddingsResponseData data;
        data.index = i;
        data.embedding = std::move(result.result[i]);
        response.data.push_back(std::move(data));
    }
    response.usage = {
        .prompt_tokens = result.prompt_tokens,
        .total_tokens  = result.total_tokens
    };
    return caiwei::text::to_json(response);
}

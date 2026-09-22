#include "caiwei/manager.hpp"
#include "caiwei/session.hpp"

caiwei::session::EmbeddingsSession::EmbeddingsSession(caiwei::text::EmbeddingsRequest& request) : request(request) {
}

std::string caiwei::session::EmbeddingsSession::get() {
    auto ptr = caiwei::manager::get_context<caiwei::context::EmbeddingContext, caiwei::text::EmbeddingsRequest, caiwei::text::EmbeddingResult>(this->request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    auto result = ptr->run(this->request);
    caiwei::text::EmbeddingResponse response;
    response.model = request.model;
    for (int i = 0; i < result.result.size(); ++i) {
        caiwei::text::EmbeddingResponseData data;
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

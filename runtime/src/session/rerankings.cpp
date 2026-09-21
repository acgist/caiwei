#include "caiwei/manager.hpp"
#include "caiwei/session.hpp"

caiwei::session::RerankingsSession::RerankingsSession(const caiwei::text::RerankingsRequest& request) : request(request) {
}

std::string caiwei::session::RerankingsSession::get() {
    auto ptr = caiwei::manager::get_context<caiwei::context::RerankingContext, caiwei::text::RerankingsRequest, caiwei::text::RerankingResult>(this->request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    auto result = ptr->run(this->request);
    caiwei::text::RerankingResponse response;
    response.model = request.model;
    for (int i = 0; i < result.result.size(); ++i) {
        caiwei::text::RerankingResponseData data;
        data.index = i;
        data.score = result.result[i][0];
        response.data.push_back(std::move(data));
    }
    response.usage = {
        .prompt_tokens = result.prompt_tokens,
        .total_tokens  = result.total_tokens
    };
    return caiwei::text::to_json(response);
}

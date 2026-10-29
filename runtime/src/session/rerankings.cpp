#include "caiwei/env.hpp"
#include "caiwei/manager.hpp"
#include "caiwei/session.hpp"

caiwei::session::RerankingsSession::RerankingsSession(caiwei::text::RerankingsRequest& request) : request(request) {
}

std::string caiwei::session::RerankingsSession::get() {
    auto ptr = caiwei::manager::get_context<caiwei::context::RerankingContext>(this->request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    auto result = ptr->run(this->request);
    caiwei::text::RerankingsResponse response;
    response.model = request.model;
    for (size_t i = 0; i < result.result.size(); ++i) {
        caiwei::text::RerankingsResponseData data;
        data.index = i;
        if (result.result[i].empty()) {
            CW_LOG_W("结果为空: %zu", i);
            data.score = 0.0F;
        } else {
            data.score = result.result[i][0];
        }
        response.data.push_back(std::move(data));
    }
    response.usage = {
        .prompt_tokens = result.prompt_tokens,
        .total_tokens  = result.total_tokens
    };
    return caiwei::text::to_json(response);
}

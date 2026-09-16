#include "caiwei/manager.hpp"
#include "caiwei/session.hpp"

caiwei::session::RerankingsSession::RerankingsSession(const caiwei::text::RerankingsRequest& request) : request(request) {
}

std::string caiwei::session::RerankingsSession::get() {
    auto ptr = caiwei::manager::get_context<caiwei::context::RerankingContext, caiwei::text::RerankingsRequest, std::string>(this->request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    return ptr->run(this->request);
}

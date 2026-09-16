#include "caiwei/manager.hpp"
#include "caiwei/session.hpp"

caiwei::session::EmbeddingsSession::EmbeddingsSession(const caiwei::text::EmbeddingsRequest& request) : request(request) {
}

std::string caiwei::session::EmbeddingsSession::get() {
    auto ptr = caiwei::manager::get_context<caiwei::context::EmbeddingContext, caiwei::text::EmbeddingsRequest, std::string>(this->request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    return ptr->run(this->request);
}

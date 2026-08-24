#include "test.hpp"
#include "caiwei/env.hpp"
#include "caiwei/log.hpp"

int main() {
    init_test();
    std::string unknown = caiwei::env::get("CAIWEI_UNKNOWN");
    std::string version = caiwei::env::get("CAIWEI_VERSION");
    LOG_INFO("CAIWEI_VERSION: %s", version.c_str());
    caiwei::env::set("CAIWEI_VERSION", "2.0.0");
    version = caiwei::env::get("CAIWEI_VERSION");
    LOG_INFO("CAIWEI_VERSION: %s", version.c_str());
    float version_float = caiwei::env::get_float("CAIWEI_VERSION");
    LOG_INFO("CAIWEI_VERSION: %f", version_float);
    bool security = caiwei::env::get_bool("CAIWEI_SECURITY");
    LOG_INFO("CAIWEI_SECURITY: %d", security);
    caiwei::env::set("CAIWEI_SECURITY", "ON");
    security = caiwei::env::get_bool("CAIWEI_SECURITY");
    LOG_INFO("CAIWEI_SECURITY: %d", security);
    LOG_INFO("ID: %" PRId64, caiwei::env::timestamp());
    LOG_INFO("ID: %s", caiwei::env::id().c_str());
    LOG_INFO("ID: %s", caiwei::env::id().c_str());
    return 0;
}

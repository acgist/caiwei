#include "test_media.hpp"

int main() {
    init_test();
    std::string unknown = caiwei::env::get("CAIWEI_UNKNOWN");
    std::string version = caiwei::env::get("CAIWEI_VERSION");
    CW_LOG_I("CAIWEI_VERSION: %s", version.c_str());
    caiwei::env::set("CAIWEI_VERSION", "2.0.0");
    version = caiwei::env::get("CAIWEI_VERSION");
    CW_LOG_I("CAIWEI_VERSION: %s", version.c_str());
    float version_float = caiwei::env::get_float("CAIWEI_VERSION");
    CW_LOG_I("CAIWEI_VERSION: %f", version_float);
    bool security = caiwei::env::get_bool("CAIWEI_SECURITY");
    CW_LOG_I("CAIWEI_SECURITY: %d", security);
    caiwei::env::set("CAIWEI_SECURITY", "ON");
    security = caiwei::env::get_bool("CAIWEI_SECURITY");
    CW_LOG_I("CAIWEI_SECURITY: %d", security);
    CW_LOG_I("ID: %" PRId64, caiwei::env::timestamp());
    CW_LOG_I("ID: %s", caiwei::env::id().c_str());
    for (int i = 0; i < 9998; i++) {
        caiwei::env::id();
    }
    CW_LOG_I("ID: %s", caiwei::env::id().c_str());
    CW_LOG_I("ID: %s", caiwei::env::id().c_str());
    CW_LOG_I("ID: %s", caiwei::env::id().c_str());
    stop_test();
    return 0;
}

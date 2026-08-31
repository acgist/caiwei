/**
 * 环境配置
 * 
 * Win  : $env:name="value"
 * Linux: export name="value"
 */
#ifndef CAIWEI_ENV_HPP
#define CAIWEI_ENV_HPP

#include <ctime>
#include <atomic>
#include <chrono>
#include <string>

namespace caiwei {
namespace env    {

const int max_id_index = 10000;

extern std::atomic_uint32_t id_index;

extern std::string get       (const std::string& name);
extern int         get_int   (const std::string& name);
extern bool        get_bool  (const std::string& name);
extern float       get_float (const std::string& name);
extern std::string get_string(const std::string& name);

extern void set(const std::string& name, const std::string& value);

extern void print_all_env();

inline size_t timestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

inline std::string id() {
    const uint32_t cur = id_index.fetch_add(1, std::memory_order_acq_rel);
    const uint32_t val = cur % max_id_index;
    return std::to_string(caiwei::env::timestamp() * max_id_index + val);
}

inline std::string yyyyMMdd_HHmmss() {
    std::string buffer;
    buffer.resize(20); // yyyy-MM-dd HH:mm:ss
    auto  tt { std::time(0)        };
    auto* tm { std::localtime(&tt) };
    std::strftime(buffer.data(), buffer.size(), "%Y-%m-%d %H:%M:%S", tm);
    return buffer;
}

} // namespace env
} // namespace caiwei
#endif // CAIWEI_ENV_HPP

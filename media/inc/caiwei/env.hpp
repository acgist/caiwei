/**
 * 环境
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

const int min_id_index =  0; // 最小ID序号
const int max_id_index = 99; // 最大ID序号

extern std::atomic_int id_index; // 当前ID序号

extern std::string get       (const std::string& name);
extern int         get_int   (const std::string& name);
extern bool        get_bool  (const std::string& name);
extern float       get_float (const std::string& name);
extern std::string get_string(const std::string& name);

extern void set(const std::string& name, const std::string& value);

extern void print_all_env();

inline std::string yyyyMMdd_HHmmss() {
    std::string buffer;
    buffer.resize(20); // yyyy-MM-dd HH:mm:ss
    auto  tt { std::time(0)        };
    auto* tm { std::localtime(&tt) };
    std::strftime(buffer.data(), buffer.size(), "%Y-%m-%d %H:%M:%S", tm);
    return buffer;
}

inline size_t timestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

inline std::string id() {
    int value = ++id_index;
    while(value > max_id_index) {
        if(id_index.compare_exchange_strong(value, min_id_index)) {
            value = min_id_index;
        } else {
            value = ++id_index;
        }
    }
    return std::to_string(caiwei::env::timestamp() * 100 + value);
}

} // namespace env
} // namespace caiwei
#endif // CAIWEI_ENV_HPP

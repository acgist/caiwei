/**
 * 环境模块
 */
#ifndef CAIWEI_ENV_HPP
#define CAIWEI_ENV_HPP

#include <ctime>
#include <atomic>
#include <chrono>
#include <string>

namespace caiwei {
namespace env    {

extern std::atomic_int id_index; // 当前ID序号
const int min_id_index =  0;     // 最小ID序号
const int max_id_index = 99;     // 最大ID序号


/**
 * 读取配置
 * 
 * @param name 名称
 * 
 * @return 值
 */
extern std::string get(const std::string& name);

/**
 * 设置配置
 * 
 * @param name  名称
 * @param value 值
 */
extern void set(const std::string& name, const std::string& value);

/**
 * @return yyyy-MM-dd HH:mm:ss
 */
inline std::string yyyyMMdd_HHmmss() {
    std::string buffer;
    buffer.resize(20);
    auto  now  { std::time(0)         };
    auto* time { std::localtime(&now) };
    std::strftime(buffer.data(), buffer.size(), "%Y-%m-%d %H:%M:%S", time);
    return buffer;
}

/**
 * @return 时间戳
 */
inline size_t timestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

/**
 * @return ID
 */
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

/**
 * 环境配置
 * 
 * Win  : $env:name="value"
 * Linux: export name="value"
 */
#ifndef CAIWEI_MEDIA_ENV_HPP
#define CAIWEI_MEDIA_ENV_HPP

#include <ctime>
#include <atomic>
#include <chrono>
#include <string>

namespace caiwei {
namespace env    {

class MessageCodeException : public std::runtime_error {
public:
    const std::string code;
    const std::string message;
public:
    MessageCodeException(const std::string message) : code("9999"), message(), std::runtime_error("") {}
    MessageCodeException(const std::string code, const std::string message) : code(std::move(code)), message(std::move(message)), std::runtime_error("") {}
    virtual ~MessageCodeException() = default;
public:
    const char* what() const noexcept override {
        return this->message.c_str();
    }
};

inline void check_bool(bool value, const std::string& message, const std::string& code = "9999") {
    if (value) {
        throw MessageCodeException(code, message);
    }
}

inline void check_empty(const std::string& value, const std::string& message = "字符串不能为空", const std::string& code = "9999") {
    if (value.empty()) {
        throw MessageCodeException(code, message);
    }
}

template<typename T>
inline void check_range(T val, T min, T max, const std::string& message = "数据范围错误", const std::string& code = "9999") {
    if (val < min || val > max) {
        throw MessageCodeException(code, message);
    }
}

inline void check_nullptr(const void* ptr, const std::string& message = "指针不能为空", const std::string& code = "9999") {
    if (ptr == nullptr) {
        throw MessageCodeException(code, message);
    }
}

const int max_id_index = 10000;

extern std::atomic_uint32_t id_index;

std::string get       (const std::string& name);
int         get_int   (const std::string& name);
int64_t     get_long  (const std::string& name);
bool        get_bool  (const std::string& name);
float       get_float (const std::string& name);
std::string get_string(const std::string& name);

void set(const std::string& name, const std::string& value);

void print_all_env();

inline size_t timestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

inline size_t unix_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
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
#endif // CAIWEI_MEDIA_ENV_HPP

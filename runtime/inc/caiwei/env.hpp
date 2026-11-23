/**
 * 环境配置
 *
 * Win  : $env:name="value"
 * Linux: export name="value"
 */
#ifndef CAIWEI_RUNTIME_ENV_HPP
#define CAIWEI_RUNTIME_ENV_HPP

#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace caiwei {
namespace env {

class MessageCodeException : public std::runtime_error {
public:
    int status;
    std::string code;
public:
    MessageCodeException(std::string message) : std::runtime_error(std::move(message)), status(500), code("9999") {
    }
    MessageCodeException(std::string code, std::string message) :
        std::runtime_error(std::move(message)),
        code(std::move(code)) {
        if (this->code.size() != 4) {
            // clang-format off
            this->code   = "9999";
            this->status = 500;
            // clang-format on
        } else {
            this->status = std::atoi(this->code.c_str()) % 1000;
            if (this->status < 100 || this->status > 599) {
                this->status = 500;
            }
        }
    }
};

// clang-format off
template<typename T>
concept Comparable = requires(T a, T z) {
    { a <  z } -> std::convertible_to<bool>;
    { a >  z } -> std::convertible_to<bool>;
    { a == z } -> std::convertible_to<bool>;
    { a != z } -> std::convertible_to<bool>;
};
// clang-format on

template<Comparable T>
inline void check_eq(const T& src, const T& dst, const std::string& message = "数据错误", const std::string& code = "1400") {
    if (src != dst) {
        throw MessageCodeException(code, message);
    }
}

template<Comparable T>
inline void
check_in(const T& val, const T& min, const T& max, const std::string& message = "数据错误", const std::string& code = "1400") {
    if (val < min || val > max) {
        throw MessageCodeException(code, message);
    }
}

template<typename T>
inline void check_not_empty(const T& value, const std::string& message = "数据错误", const std::string& code = "1400") {
    if (value.empty()) {
        throw MessageCodeException(code, message);
    }
}

inline void check_not_nullptr(const void* ptr, const std::string& message = "数据错误", const std::string& code = "1400") {
    if (ptr == nullptr) {
        throw MessageCodeException(code, message);
    }
}

inline constexpr uint32_t max_id_index = 10000;

extern std::atomic<uint32_t> id_index;

// clang-format off
std::string get      (const std::string& name);
bool        get_bool (const std::string& name);
int         get_int  (const std::string& name);
float       get_float(const std::string& name);
std::vector<std::string> get_vector(const std::string& name, const char delim = ',', const bool skip_empty = true);
// clang-format on

void set(const std::string& name, const std::string& value, bool force = false);

void init();
void stop();
void print_all_env();

void set_thread_name(const std::string& name);

const std::vector<std::string>& get_allow_path();
const std::vector<std::string>& get_allow_host();
const std::vector<std::string>& get_allow_media_type();

inline int64_t timestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

inline int64_t unix_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

inline int64_t steady_timestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline std::string id() {
    const uint32_t cur = id_index.fetch_add(1, std::memory_order_relaxed);
    const uint32_t val = cur % max_id_index;
    return std::to_string(caiwei::env::timestamp() * max_id_index + val);
}

} // namespace env
} // namespace caiwei

#endif // CAIWEI_RUNTIME_ENV_HPP

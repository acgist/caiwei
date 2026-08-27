/**
 * 模型上下文
 * 
 * session -> context -> runtime
 */
#ifndef CAIWEI_CONTEXT_HPP
#define CAIWEI_CONTEXT_HPP

#include "caiwei/runtime.hpp"

namespace caiwei  {
namespace context {

class Context {

private:
    std::shared_ptr<caiwei::runtime::Runtime> runtime{ nullptr };
    std::atomic_int32_t ref_count = 0;
public:
    Context(std::shared_ptr<caiwei::runtime::Runtime> runtime);
    ~Context();
public:
    uint32_t ref();
    uint32_t unref();
    template <typename I, typename O>
    O run(const I& input);

};



} // namespace context
} // namespace caiwei

#endif // CAIWEI_CONTEXT_HPP

#include "caiwei/manager.hpp"
#include "caiwei/session.hpp"

#include <thread>

using ClsWrapper  = caiwei::manager::ContextWrapper<caiwei::context::ClsContext,  caiwei::media::ImageFrame, std::vector<caiwei::image::Cls>>;
using DetWrapper  = caiwei::manager::ContextWrapper<caiwei::context::DetContext,  caiwei::media::ImageFrame, std::vector<caiwei::image::Box>>;
using SegWrapper  = caiwei::manager::ContextWrapper<caiwei::context::SegContext,  caiwei::media::ImageFrame, std::vector<caiwei::image::Seg>>;
using PoseWrapper = caiwei::manager::ContextWrapper<caiwei::context::PoseContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Pose>>;
using ASRWrapper  = caiwei::manager::ContextWrapper<caiwei::context::ASRContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>;
using LLMWrapper  = caiwei::manager::ContextWrapper<caiwei::context::LLMContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>;
using VLMWrapper  = caiwei::manager::ContextWrapper<caiwei::context::VLMContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>;

static bool session_stream(caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback);

static void asr_thread_session (caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback);
static void llm_thread_session (caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback);
static void vlm_thread_session (caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback);
static void yolo_thread_session(caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback);

static void sync_cls (caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_det (caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_seg (caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_pose(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_asr (caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_llm (caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_vlm (caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_generator(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response, std::generator<caiwei::text::Result> generator);

static void async_cls (caiwei::text::CompletionsRequest& request, ClsWrapper * wrapper);
static void async_det (caiwei::text::CompletionsRequest& request, DetWrapper * wrapper);
static void async_seg (caiwei::text::CompletionsRequest& request, SegWrapper * wrapper);
static void async_pose(caiwei::text::CompletionsRequest& request, PoseWrapper* wrapper);
static void async_asr (caiwei::text::CompletionsRequest& request, ASRWrapper * wrapper);
static void async_llm (caiwei::text::CompletionsRequest& request, LLMWrapper * wrapper);
static void async_vlm (caiwei::text::CompletionsRequest& request, VLMWrapper * wrapper);
static void async_generator(caiwei::text::CompletionsRequest& request, std::generator<caiwei::text::Result> generator);

caiwei::session::CompletionsSession::CompletionsSession(caiwei::text::CompletionsRequest& request, Callback callback)
    : StatefulSession(callback)
    , request(request) {
}

std::string caiwei::session::CompletionsSession::get_sync() {
    const auto* context_info = caiwei::context::get_context_info(request.model);
    caiwei::env::check_nullptr(context_info, "模型无效");
    caiwei::text::CompletionsResponse response;
    response.created = request.created;
    response.id      = request.id;
    response.model   = request.model;
    if (context_info->type == caiwei::context::Type::CLS) {
        sync_cls(request, response);
    } else if (context_info->type == caiwei::context::Type::DET) {
        sync_det(request, response);
    } else if (context_info->type == caiwei::context::Type::SEG) {
        sync_seg(request, response);
    } else if (context_info->type == caiwei::context::Type::POSE) {
        sync_pose(request, response);
    } else if (context_info->type == caiwei::context::Type::ASR) {
        sync_asr(request, response);
    } else if (context_info->type == caiwei::context::Type::LLM) {
        sync_llm(request, response);
    } else if (context_info->type == caiwei::context::Type::VLM) {
        sync_vlm(request, response);
    } else {
        throw caiwei::env::MessageCodeException("9999", "模型类型错误");
    }
    return caiwei::text::to_json(response);
}

std::future<bool> caiwei::session::CompletionsSession::get() {
    std::promise<bool> promise;
    if (this->callback == nullptr) {
        promise.set_value(false);
    } else {
        promise.set_value(session_stream(this->request, this->callback));
    }
    return promise.get_future();
}

static bool session_stream(caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback) {
    std::unique_ptr<ClsWrapper>  cls_ptr { nullptr };
    std::unique_ptr<DetWrapper>  det_ptr { nullptr };
    std::unique_ptr<SegWrapper>  seg_ptr { nullptr };
    std::unique_ptr<PoseWrapper> pose_ptr{ nullptr };
    std::unique_ptr<ASRWrapper>  asr_ptr { nullptr };
    std::unique_ptr<LLMWrapper>  llm_ptr { nullptr };
    std::unique_ptr<VLMWrapper>  vlm_ptr { nullptr };
    std::vector<std::string> all_model_list = { request.model };
    if (request.extra_body.has_value() && request.extra_body.value().model_list.has_value()) {
        auto& model_list = request.extra_body.value().model_list.value();
        all_model_list.insert(all_model_list.end(), model_list.begin(), model_list.end());
    }
    bool call_once = !request.extra_body.has_value() || !request.extra_body.value().media_url.has_value() || !request.extra_body.value().media_type.has_value();
    for (const auto& model : all_model_list) {
        const auto* context_info = caiwei::context::get_context_info(model);
        caiwei::env::check_nullptr(context_info, "模型无效");
        if (context_info->type == caiwei::context::Type::CLS) {
            cls_ptr = caiwei::manager::get_context<caiwei::context::ClsContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Cls>>(model);
            if (call_once) {
                async_cls(request, cls_ptr.get());
            }
        } else if (context_info->type == caiwei::context::Type::DET) {
            det_ptr = caiwei::manager::get_context<caiwei::context::DetContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Box>>(model);
            if (call_once) {
                async_det(request, det_ptr.get());
            }
        } else if (context_info->type == caiwei::context::Type::SEG) {
            seg_ptr = caiwei::manager::get_context<caiwei::context::SegContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Seg>>(model);
            if (call_once) {
                async_seg(request, seg_ptr.get());
            }
        } else if (context_info->type == caiwei::context::Type::POSE) {
            pose_ptr = caiwei::manager::get_context<caiwei::context::PoseContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Pose>>(model);
            if (call_once) {
                async_pose(request, pose_ptr.get());
            }
        } else if (context_info->type == caiwei::context::Type::ASR) {
            asr_ptr = caiwei::manager::get_context<caiwei::context::ASRContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>(model);
            if (call_once) {
                async_asr(request, asr_ptr.get());
            }
        } else if (context_info->type == caiwei::context::Type::LLM) {
            llm_ptr = caiwei::manager::get_context<caiwei::context::LLMContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>(model);
            if (call_once) {
                async_llm(request, llm_ptr.get());
            }
        } else if (context_info->type == caiwei::context::Type::VLM) {
            vlm_ptr = caiwei::manager::get_context<caiwei::context::VLMContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>(model);
            if (call_once) {
                async_vlm(request, vlm_ptr.get());
            }
        } else {
            throw caiwei::env::MessageCodeException("9999", "模型类型错误");
        }
    }
    if (!call_once) {
        // TODO
    }
    return true;
}

static void sync_cls(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response) {
    auto ptr = caiwei::manager::get_context<caiwei::context::ClsContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Cls>>(request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    // TODO 实现
}

static void sync_det(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response) {
    auto ptr = caiwei::manager::get_context<caiwei::context::DetContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Box>>(request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    // TODO 实现
}

static void sync_seg(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response) {
    auto ptr = caiwei::manager::get_context<caiwei::context::SegContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Seg>>(request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    // TODO 实现
}

static void sync_pose(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response) {
    auto ptr = caiwei::manager::get_context<caiwei::context::PoseContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Pose>>(request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    // TODO 实现
}

static void sync_asr(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response) {
    auto ptr = caiwei::manager::get_context<caiwei::context::ASRContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>(request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    sync_generator(request, response, ptr->run(request));
}

static void sync_llm(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response) {
    auto ptr = caiwei::manager::get_context<caiwei::context::LLMContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>(request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    sync_generator(request, response, ptr->run(request));
}

static void sync_vlm(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response) {
    auto ptr = caiwei::manager::get_context<caiwei::context::VLMContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>(request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    sync_generator(request, response, ptr->run(request));
}

static void sync_generator(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response, std::generator<caiwei::text::Result> generator) {
    std::string content;
    std::string thinking;
    std::string toolcall;
    caiwei::text::CompletionsResponseChoice choice;
    for (const auto& result : generator) {
        if (result.thinking) {
            thinking += result.token;
        } else if (result.toolcall) {
            toolcall += result.token;
        } else {
            content += result.token;
        }
        if (!result.finish_reason.empty()) {
            choice.finish_reason = result.finish_reason;
            response.usage = {
                .prompt_tokens     = result.prompt_tokens,
                .completion_tokens = result.completion_tokens,
                .total_tokens      = result.total_tokens,
            };
        }
    }
    choice.index = 0;
    choice.message.role = caiwei::text::ROLE_ASSISTANT;
    if (!thinking.empty()) {
        choice.message.reasoning_content = thinking;
    }
    if (!toolcall.empty()) {
        caiwei::text::CompletionsResponseChoiceMessageToolCallFunction function;
        nlohmann::json json = nlohmann::json::parse(toolcall);
        function.name      = json["name"].get<std::string>();
        function.arguments = json["arguments"].dump();
        caiwei::text::CompletionsResponseChoiceMessageToolCall tool_call;
        tool_call.id    = caiwei::env::id();
        tool_call.type  = "function";
        tool_call.index = 0;
        tool_call.function = function;
        if (choice.message.tool_calls.has_value()) {
            choice.message.tool_calls.value().push_back(std::move(tool_call));
        } else {
            choice.message.tool_calls = { tool_call };
        }
    }
    if (!content.empty()) {
        choice.message.content = content;
    }
    response.choices.push_back(std::move(choice));
}

static void async_cls(caiwei::text::CompletionsRequest& request, ClsWrapper* wrapper) {

}

static void async_det(caiwei::text::CompletionsRequest& request, DetWrapper* wrapper) {

}

static void async_seg(caiwei::text::CompletionsRequest& request, SegWrapper* wrapper) {

}

static void async_pose(caiwei::text::CompletionsRequest& request, PoseWrapper* wrapper) {

}

static void async_asr(caiwei::text::CompletionsRequest& request, ASRWrapper* wrapper) {

}

static void async_llm(caiwei::text::CompletionsRequest& request, LLMWrapper* wrapper) {

}

static void async_vlm(caiwei::text::CompletionsRequest& request, VLMWrapper* wrapper) {

}

static void async_generator(caiwei::text::CompletionsRequest& request, std::generator<caiwei::text::Result> generator) {
    for (const auto& result : generator) {
        caiwei::text::CompletionsChunk chunk;
        chunk.id      = request.id;
        chunk.model   = request.model;
        chunk.created = request.created;
        caiwei::text::CompletionsChunkChoice choice;
        choice.index = 0;
        choice.delta.role = caiwei::text::ROLE_ASSISTANT;
        if (result.thinking) {
            choice.delta.reasoning_content = result.token;
        } else if (result.toolcall) {
            caiwei::text::CompletionsChunkChoiceMessageToolCallFunction function;
            function.name      = result.result_toolcall->get_name();
            function.arguments = result.result_toolcall->get_arguments();
            caiwei::text::CompletionsChunkChoiceMessageToolCall tool_call;
            tool_call.id    = result.result_toolcall->toolcall_id;
            tool_call.type  = "function";
            tool_call.index = result.result_toolcall->toolcall_index;
            tool_call.function = function;
            if (choice.delta.tool_calls.has_value()) {
                choice.delta.tool_calls.value().push_back(std::move(tool_call));
            } else {
                choice.delta.tool_calls = { tool_call };
            }
        } else {
            choice.delta.content = result.token;
        }
        if (!result.finish_reason.empty()) {
            choice.finish_reason = result.finish_reason;
            chunk.usage = caiwei::text::CompletionsChunkUsage {
                .prompt_tokens     = result.prompt_tokens,
                .completion_tokens = result.completion_tokens,
                .total_tokens      = result.total_tokens,
            };
        }
        chunk.choices.push_back(std::move(choice));
    }
}

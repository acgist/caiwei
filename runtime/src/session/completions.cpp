#include "caiwei/log.hpp"
#include "caiwei/media.hpp"
#include "caiwei/manager.hpp"
#include "caiwei/session.hpp"

#include <mutex>
#include <thread>
#include <algorithm>
#include <condition_variable>

extern "C" {
#include "libavcodec/avcodec.h"
}

using ClsWrapper  = caiwei::manager::ContextWrapper<caiwei::context::ClsContext,  caiwei::media::ImageFrame, std::vector<caiwei::image::Cls>>;
using DetWrapper  = caiwei::manager::ContextWrapper<caiwei::context::DetContext,  caiwei::media::ImageFrame, std::vector<caiwei::image::Box>>;
using SegWrapper  = caiwei::manager::ContextWrapper<caiwei::context::SegContext,  caiwei::media::ImageFrame, std::vector<caiwei::image::Seg>>;
using PoseWrapper = caiwei::manager::ContextWrapper<caiwei::context::PoseContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Pose>>;
using ASRWrapper  = caiwei::manager::ContextWrapper<caiwei::context::ASRContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>;
using LLMWrapper  = caiwei::manager::ContextWrapper<caiwei::context::LLMContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>;
using VLMWrapper  = caiwei::manager::ContextWrapper<caiwei::context::VLMContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>;

static bool session_stream(caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback);

static void asr_thread_session (caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback, std::mutex& mutex, std::condition_variable& cv, ASRWrapper* asr_wrapper, std::vector<uint8_t>& audio_frame_buffer);
static void llm_thread_session (caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback, std::mutex& mutex, std::condition_variable& cv, LLMWrapper* llm_wrapper);
static void vlm_thread_session (caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback, std::mutex& mutex, std::condition_variable& cv, VLMWrapper* vlm_wrapper, std::vector<caiwei::media::VideoFrame>& video_frame_buffer, size_t& video_frame_size);
static void yolo_thread_session(caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback, std::mutex& mutex, std::condition_variable& cv, ClsWrapper* cls_wrapper, DetWrapper* det_wrapper, SegWrapper* seg_wrapper, PoseWrapper* pose_wrapper, std::vector<caiwei::media::ImageFrame>& image_frame_buffer, size_t& image_frame_size);

static void sync_cls (caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_det (caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_seg (caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_pose(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_asr (caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_llm (caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_vlm (caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response);
static void sync_generator(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response, std::generator<caiwei::text::Result> generator);

static bool async_cls (caiwei::text::CompletionsRequest& request, ClsWrapper * wrapper, caiwei::session::Callback& callback);
static bool async_det (caiwei::text::CompletionsRequest& request, DetWrapper * wrapper, caiwei::session::Callback& callback);
static bool async_seg (caiwei::text::CompletionsRequest& request, SegWrapper * wrapper, caiwei::session::Callback& callback);
static bool async_pose(caiwei::text::CompletionsRequest& request, PoseWrapper* wrapper, caiwei::session::Callback& callback);
static bool async_asr (caiwei::text::CompletionsRequest& request, ASRWrapper * wrapper, caiwei::session::Callback& callback);
static bool async_llm (caiwei::text::CompletionsRequest& request, LLMWrapper * wrapper, caiwei::session::Callback& callback);
static bool async_vlm (caiwei::text::CompletionsRequest& request, VLMWrapper * wrapper, caiwei::session::Callback& callback);
static bool async_generator(caiwei::text::CompletionsRequest& request, std::generator<caiwei::text::Result> generator, caiwei::session::Callback& callback);

static void fill_media_request(caiwei::text::CompletionsRequest& request);

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
    fill_media_request(request);
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
    for (const auto& model : all_model_list) {
        const auto* context_info = caiwei::context::get_context_info(model);
        caiwei::env::check_nullptr(context_info, "模型无效");
        if (context_info->type == caiwei::context::Type::CLS) {
            cls_ptr = caiwei::manager::get_context<caiwei::context::ClsContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Cls>>(model);
        } else if (context_info->type == caiwei::context::Type::DET) {
            det_ptr = caiwei::manager::get_context<caiwei::context::DetContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Box>>(model);
        } else if (context_info->type == caiwei::context::Type::SEG) {
            seg_ptr = caiwei::manager::get_context<caiwei::context::SegContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Seg>>(model);
        } else if (context_info->type == caiwei::context::Type::POSE) {
            pose_ptr = caiwei::manager::get_context<caiwei::context::PoseContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Pose>>(model);
        } else if (context_info->type == caiwei::context::Type::ASR) {
            asr_ptr = caiwei::manager::get_context<caiwei::context::ASRContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>(model);
        } else if (context_info->type == caiwei::context::Type::LLM) {
            llm_ptr = caiwei::manager::get_context<caiwei::context::LLMContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>(model);
        } else if (context_info->type == caiwei::context::Type::VLM) {
            vlm_ptr = caiwei::manager::get_context<caiwei::context::VLMContext, caiwei::text::CompletionsRequest, std::generator<caiwei::text::Result>>(model);
        } else {
            throw caiwei::env::MessageCodeException("9999", "模型类型错误");
        }
    }
    bool media_stream = request.extra_body.has_value() && request.extra_body.value().media_url.has_value() && request.extra_body.value().media_type.has_value();
    if (media_stream) {
        std::thread asr_thread;
        std::thread llm_thread;
        std::thread vlm_thread;
        std::thread yolo_thread;
        std::mutex asr_mutex;
        std::mutex llm_mutex;
        std::mutex vlm_mutex;
        std::mutex yolo_mutex;
        std::condition_variable asr_cv;
        std::condition_variable llm_cv;
        std::condition_variable vlm_cv;
        std::condition_variable yolo_cv;
        std::string url  = request.extra_body.value().media_url.value();
        std::string type = request.extra_body.value().media_type.value();
        request.messages.push_back(caiwei::text::CompletionsRequestMessage{
            .role = caiwei::text::ROLE_USER,
            .audio_data = {},
            .image_data = {},
            .video_data = {},
        });
        const int video_fps       = request.extra_body.value().video_fps.value_or(8);
        const int asr_frames      = request.extra_body.value().asr_frames.value_or(16000);
        const int asr_queue_size  = request.extra_body.value().asr_queue_size.value_or(128000);
        const int vlm_frames      = request.extra_body.value().vlm_frames.value_or(8);
        const int yolo_queue_size = request.extra_body.value().yolo_queue_size.value_or(8);
        auto& message = request.messages.back();
        message.audio_data.resize(1);
        message.image_data.resize(1);
        message.video_data.resize(1);
        message.audio_data[0].data.reserve(asr_frames);
        message.video_data[0].resize(vlm_frames);
        size_t frame_count = 0;
        size_t image_frame_size = 0;
        size_t video_frame_size = 0;
        std::vector<uint8_t>                   audio_frame_buffer;
        std::vector<caiwei::media::ImageFrame> image_frame_buffer;
        std::vector<caiwei::media::VideoFrame> video_frame_buffer;
        audio_frame_buffer.reserve(asr_queue_size);
        image_frame_buffer.resize(yolo_queue_size);
        video_frame_buffer.resize(vlm_frames);
        const bool enable_asr  = asr_ptr != nullptr;
        const bool enable_llm  = llm_ptr != nullptr;
        const bool enable_vlm  = vlm_ptr != nullptr;
        const bool enable_yolo = cls_ptr != nullptr || det_ptr != nullptr || seg_ptr != nullptr || pose_ptr != nullptr;
        if (enable_asr) {
            asr_thread = std::thread(asr_thread_session, std::ref(request), std::ref(callback), std::ref(asr_mutex), std::ref(asr_cv), asr_ptr.get(), std::ref(audio_frame_buffer));
        }
        if (enable_llm) {
            llm_thread = std::thread(llm_thread_session, std::ref(request), std::ref(callback), std::ref(llm_mutex), std::ref(llm_cv), llm_ptr.get());
        }
        if (enable_vlm) {
            vlm_thread = std::thread(vlm_thread_session, std::ref(request), std::ref(callback), std::ref(vlm_mutex), std::ref(vlm_cv), vlm_ptr.get(), std::ref(video_frame_buffer), std::ref(video_frame_size));
        }
        if (enable_yolo) {
            yolo_thread = std::thread(yolo_thread_session, std::ref(request), std::ref(callback), std::ref(yolo_mutex), std::ref(yolo_cv), cls_ptr.get(), det_ptr.get(), seg_ptr.get(), pose_ptr.get(), std::ref(image_frame_buffer), std::ref(image_frame_size));
        }
        caiwei::media::MediaDemuxer media_demuxer(type, url, [&](caiwei::media::AudioFrame& frame) {
            if (enable_asr) {
                std::lock_guard<std::mutex> lock(asr_mutex);
                if (audio_frame_buffer.size() + frame.data.size() <= asr_queue_size) {
                    audio_frame_buffer.insert(
                        audio_frame_buffer.end(),
                        frame.data.begin(),
                        frame.data.end()
                    );
                    if (audio_frame_buffer.size() >= asr_frames) {
                        asr_cv.notify_one();
                    }
                } else {
                    asr_cv.notify_one();
                    CW_LOG_W("音频数据队列已满无法添加数据");
                }
            }
            return callback(nullptr, nullptr, 0);
        }, [&](caiwei::media::VideoFrame& frame) {
            if (++frame_count % video_fps != 0) {
                return callback(nullptr, nullptr, 0);
            }
            if (enable_yolo) {
                std::lock_guard<std::mutex> lock(yolo_mutex);
                if (image_frame_size + 1 <= yolo_queue_size) {
                    auto& data = image_frame_buffer[image_frame_size];
                    if (
                        data.width       != frame.width    ||
                        data.height      != frame.height   ||
                        data.channels    != frame.channels ||
                        data.data.size() != frame.data.size()
                    ) {
                        data.width    = frame.width;
                        data.height   = frame.height;
                        data.channels = frame.channels;
                        data.data.resize(frame.data.size());
                    }
                    if (enable_vlm) {
                        std::copy(frame.data.begin(), frame.data.end(), data.data.begin());
                    } else {
                        frame.data.swap(data.data);
                    }
                    ++image_frame_size;
                    yolo_cv.notify_one();
                } else {
                    yolo_cv.notify_one();
                    CW_LOG_W("图片数据队列已满无法添加数据");
                }
            }
            if (enable_vlm) {
                std::lock_guard<std::mutex> lock(vlm_mutex);
                if (video_frame_size + 1 <= vlm_frames) {
                    auto& data = video_frame_buffer[video_frame_size];
                    if (
                        data.width       != frame.width    ||
                        data.height      != frame.height   ||
                        data.channels    != frame.channels ||
                        data.data.size() != frame.data.size()
                    ) {
                        data.width    = frame.width;
                        data.height   = frame.height;
                        data.channels = frame.channels;
                        data.data.resize(frame.data.size());
                    }
                    if (enable_yolo) {
                        std::copy(frame.data.begin(), frame.data.end(), data.data.begin());
                    } else {
                        frame.data.swap(data.data);
                    }
                    ++video_frame_size;
                    if (video_frame_size == vlm_frames) {
                        vlm_cv.notify_one();
                    }
                } else {
                    vlm_cv.notify_one();
                    CW_LOG_W("视频数据队列已满无法添加数据");
                }
            }
            return callback(nullptr, nullptr, 0);
        });
        CW_LOG_I("开始解析视频: %s = %s", type.c_str(), url.c_str());
        bool ret = media_demuxer.open(
            caiwei::media::AudioInfo(1, 16000, AV_SAMPLE_FMT_S16),
            caiwei::media::VideoInfo(640, 0, AV_PIX_FMT_RGB24)
        );
        media_demuxer.stop();
        if (asr_thread.joinable()) {
            asr_thread.join();
        }
        if (llm_thread.joinable()) {
            llm_thread.join();
        }
        if (vlm_thread.joinable()) {
            vlm_thread.join();
        }
        if (yolo_thread.joinable()) {
            yolo_thread.join();
        }
    } else {
        fill_media_request(request);
        async_cls (request, cls_ptr .get(), callback);
        async_det (request, det_ptr .get(), callback);
        async_seg (request, seg_ptr .get(), callback);
        async_pose(request, pose_ptr.get(), callback);
        async_asr (request, asr_ptr .get(), callback);
        async_llm (request, llm_ptr .get(), callback);
        async_vlm (request, vlm_ptr .get(), callback);
    }
    return true;
}

template <typename T>
inline void sync_yolo(std::unique_ptr<T> ptr, caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response) {
    int index = 0;
    auto& messages = request.messages;
    for (auto& message : messages) {
        if (message.role != caiwei::text::ROLE_USER) {
            continue;
        }
        auto& image_data = message.image_data;
        if (image_data.empty()) {
            continue;
        }
        for (auto& value : image_data) {
            auto ret = ptr->run(value);
            caiwei::text::CompletionsResponseChoice choice;
            choice.index = index++;
            choice.message.role = caiwei::text::ROLE_ASSISTANT;
            choice.message.content = caiwei::image::to_json(ret);
            response.choices.push_back(std::move(choice));
        }
    }
}

static void sync_cls(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response) {
    auto ptr = caiwei::manager::get_context<caiwei::context::ClsContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Cls>>(request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    sync_yolo(std::move(ptr), request, response);
}

static void sync_det(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response) {
    auto ptr = caiwei::manager::get_context<caiwei::context::DetContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Box>>(request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    sync_yolo(std::move(ptr), request, response);
}

static void sync_seg(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response) {
    auto ptr = caiwei::manager::get_context<caiwei::context::SegContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Seg>>(request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    sync_yolo(std::move(ptr), request, response);
}

static void sync_pose(caiwei::text::CompletionsRequest& request, caiwei::text::CompletionsResponse& response) {
    auto ptr = caiwei::manager::get_context<caiwei::context::PoseContext, caiwei::media::ImageFrame, std::vector<caiwei::image::Pose>>(request.model);
    caiwei::env::check_nullptr(ptr.get(), "模型无效");
    sync_yolo(std::move(ptr), request, response);
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

template <typename T>
inline bool async_yolo(caiwei::text::CompletionsRequest& request, T* wrapper, caiwei::session::Callback& callback) {
    if (wrapper == nullptr) {
        return true;
    }
    caiwei::text::CompletionsChunk chunk;
    chunk.created = request.created;
    chunk.id      = request.id;
    chunk.model   = request.model;
    int index = 0;
    auto& messages = request.messages;
    for (auto& message : messages) {
        if (message.role != caiwei::text::ROLE_USER) {
            continue;
        }
        auto& image_data = message.image_data;
        if (image_data.empty()) {
            continue;
        }
        for (auto& value : image_data) {
            auto ret = wrapper->run(value);
            caiwei::text::CompletionsChunkChoice choice;
            choice.index = index++;
            choice.delta.role = caiwei::text::ROLE_ASSISTANT;
            choice.delta.content = caiwei::image::to_json(ret);
            chunk.choices.push_back(std::move(choice));
        }
    }
    std::string message = caiwei::text::to_json(chunk);
    return callback("data", message.data(), message.size());
}

static bool async_cls(caiwei::text::CompletionsRequest& request, ClsWrapper* wrapper, caiwei::session::Callback& callback) {
    return async_yolo(request, wrapper, callback);
}

static bool async_det(caiwei::text::CompletionsRequest& request, DetWrapper* wrapper, caiwei::session::Callback& callback) {
    return async_yolo(request, wrapper, callback);
}

static bool async_seg(caiwei::text::CompletionsRequest& request, SegWrapper* wrapper, caiwei::session::Callback& callback) {
    return async_yolo(request, wrapper, callback);
}

static bool async_pose(caiwei::text::CompletionsRequest& request, PoseWrapper* wrapper, caiwei::session::Callback& callback) {
    return async_yolo(request, wrapper, callback);
}

static bool async_asr(caiwei::text::CompletionsRequest& request, ASRWrapper* wrapper, caiwei::session::Callback& callback) {
    if (wrapper == nullptr) {
        return true;
    }
    return async_generator(request, wrapper->run(request), callback);
}

static bool async_llm(caiwei::text::CompletionsRequest& request, LLMWrapper* wrapper, caiwei::session::Callback& callback) {
    if (wrapper == nullptr) {
        return true;
    }
    return async_generator(request, wrapper->run(request), callback);
}

static bool async_vlm(caiwei::text::CompletionsRequest& request, VLMWrapper* wrapper, caiwei::session::Callback& callback) {
    if (wrapper == nullptr) {
        return true;
    }
    return async_generator(request, wrapper->run(request), callback);
}

static bool async_generator(caiwei::text::CompletionsRequest& request, std::generator<caiwei::text::Result> generator, caiwei::session::Callback& callback) {
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
    // TODO
    return callback("data", nullptr, 0);
}

static void fill_media_request(caiwei::text::CompletionsRequest& request) {
    // TODO
}

static void asr_thread_session(caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback, std::mutex& mutex, std::condition_variable& cv, ASRWrapper* asr_wrapper, std::vector<uint8_t>& audio_frame_buffer) {
    if (asr_wrapper == nullptr) {
        return;
    }
    auto& message = request.messages.back();
    auto& audio_data = message.audio_data[0];
    const int asr_frames     = request.extra_body.value().asr_frames.value_or(16000);
    const int asr_queue_size = request.extra_body.value().asr_queue_size.value_or(128000);
    std::vector<uint8_t> audio_copy;
    std::vector<uint8_t> audio_slice;
    audio_copy.reserve(asr_queue_size);
    audio_slice.reserve(asr_frames);
    bool callable = true;
    while (callable) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait_for(lock, std::chrono::seconds(1));
            if (audio_frame_buffer.empty()) {
                callable = callback(nullptr, nullptr, 0);
                continue;
            }
            audio_copy.swap(audio_frame_buffer);
        }
        int index = 0;
        while (callable && index < audio_copy.size()) {
            if (index + asr_frames > audio_copy.size()) {
                audio_slice.assign(audio_copy.begin() + index, audio_copy.end());
                break;
            }
            if (audio_slice.empty()) {
                audio_slice.assign(audio_copy.begin() + index, audio_copy.begin() + index + asr_frames);
                index += asr_frames;
            } else {
                audio_slice.insert(audio_slice.end(), audio_copy.begin() + index, audio_copy.begin() + index + asr_frames - audio_slice.size());
                index += asr_frames - audio_slice.size();
            }
            audio_data.data.swap(audio_slice);
            audio_slice.resize(0);
            callable = async_asr(request, asr_wrapper, callback);
        }
        audio_copy.resize(0);
    }
}

static void llm_thread_session(caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback, std::mutex& mutex, std::condition_variable& cv, LLMWrapper* llm_wrapper) {
    if (llm_wrapper == nullptr) {
        return;
    }
    async_llm(request, llm_wrapper, callback);
}

static void vlm_thread_session(caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback, std::mutex& mutex, std::condition_variable& cv, VLMWrapper* vlm_wrapper, std::vector<caiwei::media::VideoFrame>& video_frame_buffer, size_t& video_frame_size) {
    if (vlm_wrapper == nullptr) {
        return;
    }
    auto& message = request.messages.back();
    auto& video_data = message.video_data[0];
    const int vlm_frames = request.extra_body.value().vlm_frames.value_or(8);
    bool callable = true;
    while (callable) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait_for(lock, std::chrono::seconds(1));
            if (video_frame_size < vlm_frames) {
                callable = callback(nullptr, nullptr, 0);
                continue;
            }
            video_data.swap(video_frame_buffer);
            video_frame_size = 0;
        }
        callable = async_vlm(request, vlm_wrapper, callback);
    }
}

static void yolo_thread_session(caiwei::text::CompletionsRequest& request, caiwei::session::Callback callback, std::mutex& mutex, std::condition_variable& cv, ClsWrapper* cls_wrapper, DetWrapper* det_wrapper, SegWrapper* seg_wrapper, PoseWrapper* pose_wrapper, std::vector<caiwei::media::ImageFrame>& image_frame_buffer, size_t& image_frame_size) {
    if (cls_wrapper == nullptr || det_wrapper == nullptr || seg_wrapper == nullptr || pose_wrapper == nullptr) {
        return;
    }
    auto& message = request.messages.back();
    auto& image_data = message.image_data[0];
    const int yolo_queue_size = request.extra_body.value().yolo_queue_size.value_or(8);
    std::vector<caiwei::media::ImageFrame> image_copy;
    image_copy.resize(yolo_queue_size);
    size_t image_copy_size = 0;
    bool callable = true;
    while (callable) {
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait_for(lock, std::chrono::seconds(1));
            if (image_frame_size <= 0) {
                callable = callback(nullptr, nullptr, 0);
                continue;
            }
            image_copy_size = image_frame_size;
            image_copy.swap(image_frame_buffer);
            image_frame_size = 0;
        }
        for (size_t i = 0; callable && i < image_copy_size; ++i) {
            auto& frame = image_copy[i];
            image_data.width    = frame.width;
            image_data.height   = frame.height;
            image_data.channels = frame.channels;
            image_data.data.swap(frame.data);
            if (callable) {
                callable = async_cls(request, cls_wrapper, callback);
            }
            if (callable) {
                callable = async_det(request, det_wrapper, callback);
            }
            if (callable) {
                callable = async_seg(request, seg_wrapper, callback);
            }
            if (callable) {
                callable = async_pose(request, pose_wrapper, callback);
            }
        }
    }
}
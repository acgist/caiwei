/**
 * 文本结构
 */
#ifndef CAIWEI_MEDIA_TEXT_DATA_HPP
#define CAIWEI_MEDIA_TEXT_DATA_HPP

#include <string>
#include <cstdint>
#include <variant>
#include <optional>

#include "nlohmann/json.hpp"

#ifndef CW_TRANSIENT
#define CW_TRANSIENT
#endif

namespace caiwei {
namespace text   {
    
const std::string ROLE_USER      = "user";
const std::string ROLE_TOOL      = "tool";
const std::string ROLE_SYSTEM    = "system";
const std::string ROLE_FUNCTION  = "function";
const std::string ROLE_ASSISTANT = "assistant";
const std::string ROLE_DEVELOPER = "developer";

const std::string FINISH_REASON_STOP           = "stop";
const std::string FINISH_REASON_LENGTH         = "length";
const std::string FINISH_REASON_TOOL_CALLS     = "tool_calls";
const std::string FINISH_REASON_CONTENT_FILTER = "content_filter";

struct CompletionsRequestMessageContentItemUrl {
    std::optional<std::string> url;
};

using AudioData = std::vector<std::vector<float>>;
using ImageData = std::vector<std::vector<uint8_t>>;
using VideoData = std::vector<std::vector<std::vector<uint8_t>>>;

// 支持格式: file/http/base64
// file://
// http://
// data:image/png;base64,
// data:image/jpeg;base64,
struct CompletionsRequestMessageContentItem {
    std::optional<std::string> type;
    std::optional<std::string> text;
    std::optional<std::string> audio;
    std::optional<std::string> image;
    std::optional<std::string> video;
    std::optional<CompletionsRequestMessageContentItemUrl> audio_url;
    std::optional<CompletionsRequestMessageContentItemUrl> image_url;
    std::optional<CompletionsRequestMessageContentItemUrl> video_url;
    CW_TRANSIENT AudioData audio_data;
    CW_TRANSIENT ImageData image_data;
    CW_TRANSIENT VideoData video_data;
};

using CompletionsRequestMessageContent = std::variant<std::string, std::vector<CompletionsRequestMessageContentItem>>;

struct CompletionsRequestMessageToolCallFunction {
    std::optional<std::string> name;
    std::optional<std::string> arguments;
};

struct CompletionsRequestMessageToolCall {
    std::optional<std::string> id;
    std::optional<std::string> type;
    std::optional<CompletionsRequestMessageToolCallFunction> function;
};

struct CompletionsRequestMessage {
    std::optional<std::string> role;
    std::optional<std::string> name;
    std::optional<std::string> tool_call_id;
    std::optional<CompletionsRequestMessageContent> content;
    std::optional<std::vector<CompletionsRequestMessageToolCall>> tool_calls;
};

struct CompletionsRequestToolFunction {
    std::optional<std::string> name;
    std::optional<std::string> description;
    std::optional<nlohmann::json> parameters;
};

struct CompletionsRequestTool {
    std::optional<std::string> type;
    std::optional<CompletionsRequestToolFunction> function;
};

struct CompletionsRequestExtraBody {
    std::optional<int> video_fps  = 8;     // 视频识别间隔帧数: VLM/YOLO
    std::optional<int> asr_frames = 16000; // ASR识别帧数: 16000 * 1 * 16 * 0.5 / 8 = 16000
    std::optional<int> vlm_frames = 8;     // VLM识别帧数
    std::optional<int> audio_queue_size = 128000; // 音频识别队列大小: 16000 * 1 * 16 * 4 / 8 = 128000
    std::optional<int> video_queue_size = 8;      // 视频识别队列大小
    std::optional<std::string> video_url;  // 持续识别视频文件地址
    std::optional<std::string> video_type; // 持续识别视频文件类型
    std::optional<bool> enable_thinking = false;
    std::optional<std::vector<std::string>> model_list;
};

struct CompletionsRequest {
    bool stream = false;
    std::string model;
    std::vector<CompletionsRequestMessage> messages;
    std::optional<int>   seed;
    std::optional<int>   top_k;
    std::optional<float> top_p;
    std::optional<float> temperature;
    std::optional<float> repeat_penalty;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<uint32_t> max_completion_tokens;
    std::optional<std::vector<CompletionsRequestTool>> tools;
    std::optional<CompletionsRequestExtraBody> extra_body;
    CW_TRANSIENT std::string id;
    CW_TRANSIENT uint32_t    index;
    CW_TRANSIENT uint32_t    created;
};

struct CompletionsResponseChoiceMessageToolCallFunction {
    std::optional<std::string> name;
    std::optional<std::string> arguments;
};

struct CompletionsResponseChoiceMessageToolCall {
    std::optional<std::string> id;
    std::optional<std::string> type;
    std::optional<uint32_t>    index;
    std::optional<CompletionsResponseChoiceMessageToolCallFunction> function;
};

struct CompletionsResponseChoiceMessage {
    std::optional<std::string> role;
    std::optional<std::string> content;
    std::optional<std::string> refusal;
    std::optional<std::string> reasoning_content;
    std::optional<std::vector<CompletionsResponseChoiceMessageToolCall>> tool_calls;
};

struct CompletionsResponseChoice {
    uint32_t index;
    std::string finish_reason;
    CompletionsResponseChoiceMessage message;
};

struct CompletionsResponseUsage {
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    uint32_t total_tokens;
};

struct CompletionsResponse {
    uint32_t    created;
    std::string id;
    std::string model;
    std::string object = "chat.completion";
    std::vector<CompletionsResponseChoice>  choices;
    std::optional<CompletionsResponseUsage> usage;
};

struct CompletionsChunkChoiceMessageToolCallFunction {
    std::optional<std::string> name;
    std::optional<std::string> arguments;
};

struct CompletionsChunkChoiceMessageToolCall {
    std::optional<std::string> id;
    std::optional<std::string> type;
    std::optional<uint32_t>    index;
    std::optional<CompletionsChunkChoiceMessageToolCallFunction> function;
};

struct CompletionsChunkDelta {
    std::optional<std::string> role;
    std::optional<std::string> content;
    std::optional<std::string> refusal;
    std::optional<std::string> reasoning_content;
    std::optional<std::vector<CompletionsChunkChoiceMessageToolCall>> tool_calls;
};

struct CompletionsChunkChoice {
    uint32_t index;
    std::string finish_reason;
    CompletionsChunkDelta delta;
};

struct CompletionsChunkUsage {
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    uint32_t total_tokens;
};

struct CompletionsChunk {
    uint32_t    created;
    std::string id;
    std::string model;
    std::string object = "chat.completion.chunk";
    std::vector<CompletionsChunkChoice>  choices;
    std::optional<CompletionsChunkUsage> usage;
};

struct EmbeddingRequest {
    std::string model;
    std::variant<std::string, std::vector<std::string>> input;
};

struct EmbeddingResponseData {
    uint32_t    index;
    std::string object = "embedding";
    std::vector<float> embedding;
};

struct EmbeddingResponseUsage {
    std::optional<uint32_t> prompt_tokens;
    std::optional<uint32_t> total_tokens;
};

struct EmbeddingResponse {
    std::string object = "list";
    std::string model;
    std::vector<EmbeddingResponseData>    data;
    std::optional<EmbeddingResponseUsage> usage;
};

struct RerankingRequest {
    std::string model;
    std::string query;
    std::vector<std::string>   documents;
    std::optional<std::string> instruct;
};

struct RerankingResponseData {
    uint32_t    index;
    std::string object = "reranking";
    float       score;
};

struct RerankingResponseUsage {
    std::optional<uint32_t> prompt_tokens;
    std::optional<uint32_t> total_tokens;
};

struct RerankingResponse {
    std::string object = "list";
    std::string model;
    std::vector<RerankingResponseData>    data;
    std::optional<RerankingResponseUsage> usage;
};

struct SpecialToken {
    std::string bos;
    std::string eos;
    std::string pad;
    std::string b_audio;
    std::string c_audio;
    std::string e_audio;
    std::string b_image;
    std::string c_image;
    std::string e_image;
    std::string b_video;
    std::string c_video;
    std::string e_video;
    std::string b_thinking;
    std::string e_thinking;
    std::string b_toolcall;
    std::string e_toolcall;
    std::string enable_thinking;
};

struct ResultToolcall {
    bool name_return      = false;
    bool arguments_return = false;
    std::string toolcall_id;
    uint32_t    toolcall_index;
    std::string token;
    std::string content;
    std::string arguments;
    
    ResultToolcall();

    void reset();
    void finish();
    void put_token(std::string token);
    std::string get_name();
    std::string get_arguments();
};

struct Result {
    bool thinking = false;
    bool toolcall = false;
    std::string token;
    std::string finish_reason;
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    uint32_t total_tokens;
    ResultToolcall* result_toolcall = nullptr;

    Result(bool thinking, bool toolcall, std::string token);
    Result(bool thinking, bool toolcall, ResultToolcall* result_toolcall);
    Result(bool thinking, bool toolcall, std::string finish_reason, uint32_t prompt_tokens, uint32_t completion_tokens);
};

std::string completions_response(const CompletionsRequest& request, const std::string& finish_reason, std::string content, std::string thinking, std::string toolcall);
std::string completions_chunk   (const CompletionsRequest& request, const Result& result);

CompletionsRequest json_to_completions(const std::string& json);
EmbeddingRequest   json_to_embedding  (const std::string& json);
RerankingRequest   json_to_reranking  (const std::string& json);
std::string to_json(const CompletionsResponse& response);
std::string to_json(const CompletionsChunk   & chunk);
std::string to_json(const EmbeddingResponse  & response);
std::string to_json(const RerankingResponse  & response);

} // namespace text
} // namespace caiwei

#endif // CAIWEI_MEDIA_TEXT_DATA_HPP

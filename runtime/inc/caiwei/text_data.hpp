#ifndef CAIWEI_RUNTIME_TEXT_DATA_HPP
#define CAIWEI_RUNTIME_TEXT_DATA_HPP

#include <string>
#include <cstdint>
#include <variant>
#include <optional>

#include "nlohmann/json.hpp"

#include "caiwei/media_data.hpp"

#ifndef CW_TRANSIENT
#define CW_TRANSIENT
#endif

namespace caiwei {
namespace text   {
    
inline constexpr const char* ROLE_TOOL      = "tool";
inline constexpr const char* ROLE_USER      = "user";
inline constexpr const char* ROLE_SYSTEM    = "system";
inline constexpr const char* ROLE_FUNCTION  = "function";
inline constexpr const char* ROLE_ASSISTANT = "assistant";
inline constexpr const char* ROLE_DEVELOPER = "developer";

inline constexpr const char* TYPE_FUNCTION  = "function";
inline constexpr const char* TYPE_EMBEDDING = "embedding";
inline constexpr const char* TYPE_RERANKING = "reranking";

inline constexpr const char* FINISH_REASON_STOP       = "stop";
inline constexpr const char* FINISH_REASON_LENGTH     = "length";
inline constexpr const char* FINISH_REASON_TOOL_CALLS = "tool_calls";

inline constexpr const char* CONTENT_TYPE_DATA  = "data:";
inline constexpr const char* CONTENT_TYPE_FILE  = "file:";
inline constexpr const char* CONTENT_TYPE_HTTP  = "http:";
inline constexpr const char* CONTENT_TYPE_HTTPS = "https:";

struct CompletionsRequestMessageContentItemUrl {
    std::optional<std::string> url;
};

struct CompletionsRequestMessageContentItem {
    std::optional<std::string> type;
    std::optional<std::string> text;
    std::optional<std::string> audio;
    std::optional<std::string> image;
    std::optional<std::string> video;
    std::optional<CompletionsRequestMessageContentItemUrl> audio_url;
    std::optional<CompletionsRequestMessageContentItemUrl> image_url;
    std::optional<CompletionsRequestMessageContentItemUrl> video_url;
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
    CW_TRANSIENT std::vector<caiwei::media::AudioFrame>              audio_data;
    CW_TRANSIENT std::vector<caiwei::media::ImageFrame>              image_data;
    CW_TRANSIENT std::vector<std::vector<caiwei::media::VideoFrame>> video_data;
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

struct CompletionsRequest {
    bool stream = false;
    std::string model;
    std::vector<CompletionsRequestMessage> messages;
    std::optional<int>   seed;
    std::optional<int>   top_k          = 10;
    std::optional<float> top_p          = 0.95F;
    std::optional<float> temperature    = 0.8F;
    std::optional<float> repeat_penalty = 1.04F;
    std::optional<float> presence_penalty  = 0.0F;
    std::optional<float> frequency_penalty = 0.0F;
    std::optional<uint32_t> max_completion_tokens;
    std::optional<std::vector<CompletionsRequestTool>> tools;
    // 扩展字段
    std::optional<int> video_fps   = 8;      // 视频识别间隔帧数: VLM/YOLO
    std::optional<int> asr_samples = 32000;  // ASR采样队列数量: 16000 * 1 * 16 *  1 / 8 = 32000
    std::optional<int> vlm_frames  = 8;      // VLM帧数队列数量
    std::optional<bool> drop_on_full    = false; // 队列满时是否丢弃数据
    std::optional<bool> enable_thinking = false; // 是否开启思考模式
    std::optional<std::string> media_url;  // 持续识别视频文件地址
    std::optional<std::string> media_type; // 持续识别视频文件类型
    std::optional<std::vector<std::string>> model_list;
    CW_TRANSIENT std::string id;
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
    std::vector<CompletionsResponseChoice> choices;
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
    std::vector<CompletionsChunkChoice> choices;
    std::optional<CompletionsChunkUsage> usage;
};

struct EmbeddingsRequest {
    std::string model;
    std::variant<std::string, std::vector<std::string>> input;
};

struct EmbeddingsResponseData {
    uint32_t    index;
    std::string object = "embedding";
    std::vector<float> embedding;
};

struct EmbeddingsResponseUsage {
    std::optional<uint32_t> prompt_tokens;
    std::optional<uint32_t> total_tokens;
};

struct EmbeddingsResponse {
    std::string object = "list";
    std::string model;
    std::vector<EmbeddingsResponseData> data;
    std::optional<EmbeddingsResponseUsage> usage;
};

struct RerankingsRequest {
    std::string model;
    std::string query;
    std::vector<std::string> documents;
    std::optional<std::string> instruct;
};

struct RerankingsResponseData {
    uint32_t    index;
    std::string object = "reranking";
    float       score;
};

struct RerankingsResponseUsage {
    std::optional<uint32_t> prompt_tokens;
    std::optional<uint32_t> total_tokens;
};

struct RerankingsResponse {
    std::string object = "list";
    std::string model;
    std::vector<RerankingsResponseData> data;
    std::optional<RerankingsResponseUsage> usage;
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
    std::string media_marker;
    std::string audio_marker;
    std::string image_marker;
    std::string video_marker;
    std::string enable_thinking;
    std::string instruction_key;
    std::string query_key;
    std::string document_key;
    std::string system_prompt;
    std::string instruction_prompt;

    void load_asr();
    void load_llm();
    void load_vlm();
    void load_embedding();
    void load_reranking();
};

struct ToolcallSession {
    bool name_return      = false;
    bool arguments_return = false;
    std::string toolcall_id;
    uint32_t    toolcall_index;
    std::string token;
    std::string content;
    std::string arguments;
    
    ToolcallSession();

    void reset();
    void finish();
    void put_token(std::string token);
    std::string get_name();
    std::string get_arguments();
};

struct CompletionResult {
    bool thinking;
    bool toolcall;
    std::string token;
    std::string finish_reason;
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    uint32_t total_tokens;
    std::string toolcall_name;
    std::string toolcall_arguments;
    std::string toolcall_id;
    uint32_t    toolcall_index;

    CompletionResult(bool thinking, bool toolcall, std::string token, ToolcallSession& toolcall_session);
    CompletionResult(bool thinking, bool toolcall, uint32_t prompt_tokens, uint32_t completion_tokens);
    CompletionResult(std::string finish_reason, uint32_t prompt_tokens, uint32_t completion_tokens);
};

struct EmbeddingResult {
    std::vector<std::vector<float>> result;
    uint32_t prompt_tokens;
    uint32_t total_tokens;
};

using RerankingResult = EmbeddingResult;

CompletionsRequest json_to_completions(const std::string& json);
EmbeddingsRequest  json_to_embeddings (const std::string& json);
RerankingsRequest  json_to_rerankings (const std::string& json);
std::string to_json(const CompletionsResponse& response);
std::string to_json(const CompletionsChunk   & chunk);
std::string to_json(const EmbeddingsResponse & response);
std::string to_json(const RerankingsResponse & response);

} // text
} // caiwei

#endif // CAIWEI_RUNTIME_TEXT_DATA_HPP

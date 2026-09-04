/**
 * 文本结构
 */
#ifndef CAIWEI_MEDIA_TEXT_DATA_HPP
#define CAIWEI_MEDIA_TEXT_DATA_HPP

#include <map>
#include <string>
#include <variant>
#include <optional>

#include "nlohmann/json.hpp"

namespace caiwei {
namespace text   {

enum class Role {
    USER,
    TOOL,
    SYSTEM,
    ASSISTANT,
};

enum class Type {
    TEXT,
    AUDIO,
    IMAGE,
    VIDEO,
    AUDIO_URL,
    IMAGE_URL,
    VIDEO_URL,
};

struct ASRRequest {
    std::string model;
    std::string type; // 支持媒体类型
    std::string url; // TODO 支持文件上传 url
    std::optional<std::string> language;
    std::optional<std::string> response_format;
};

struct ASRResponse {
    std::string text; // TODO 流式返回
};

struct TTSRequest {
    std::string model;
    std::string input; // == text
    std::optional<std::string> voice;   // == speaker
    std::optional<std::string> instruct;
    std::optional<std::string> clone_text;
    std::optional<std::string> clone_audio;
    std::optional<std::string> language;
    std::optional<std::string> response_format; // AAC/MP3/PCM/WAV
};

struct TTSResponse {

};

struct CompletionsRequestMessageContentUrl {
    std::optional<std::string> url;
};

struct CompletionsRequestMessageContentItem {
    std::optional<std::string> type;
    std::optional<std::string> text;
    std::optional<std::string> audio;
    std::optional<std::string> image;
    std::optional<std::string> video;
    std::optional<CompletionsRequestMessageContentUrl> audio_url;
    std::optional<CompletionsRequestMessageContentUrl> image_url;
    std::optional<CompletionsRequestMessageContentUrl> video_url;
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

struct CompletionsRequest {
    bool stream = false;
    std::string model;
    std::vector<CompletionsRequestMessage> messages;
    std::optional<bool>  enable_thinking;
    std::optional<int>   seed;
    std::optional<float> top_p;
    std::optional<float> temperature;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<uint32_t> max_tokens;
    std::optional<std::vector<CompletionsRequestTool>> tools;
};

struct CompletionsResponseChoiceMessageToolCallFunction {
    std::optional<std::string> name;
    std::optional<std::string> arguments;
};

struct CompletionsResponseChoiceMessageToolCall {
    std::optional<std::string> id;
    std::optional<std::string> type;
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
    std::optional<std::string> finish_reason;
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
    std::optional<CompletionsResponseUsage> usage;
    std::vector<CompletionsResponseChoice>  choices;
};

struct CompletionsChunkChoiceMessageToolCallFunction {
    std::optional<std::string> name;
    std::optional<std::string> arguments;
};

struct CompletionsChunkChoiceMessageToolCall {
    std::optional<std::string> id;
    std::optional<std::string> type;
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
    std::optional<std::string> finish_reason;
    CompletionsChunkDelta delta;
};

struct CompletionsChunkUsage {
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    uint32_t total_tokens;
};

// TODO 最后 data: [DONE]
struct CompletionsChunk {
    uint32_t    created;
    std::string id;
    std::string model;
    std::string object = "chat.completion.chunk";
    std::optional<CompletionsChunkUsage> usage;
    std::vector<CompletionsChunkChoice>  choices;
};

struct EmbeddingRequestInputContent {
    std::optional<std::string> text;
    std::optional<std::string> image;
    std::optional<std::string> video;
};

using EmbeddingRequestInputItem = std::variant<std::string, EmbeddingRequestInputContent>;
using EmbeddingRequestInput = std::variant<EmbeddingRequestInputItem, std::vector<EmbeddingRequestInputItem>>;

struct EmbeddingRequest {
    std::string model;
    std::optional<std::string> instruct;
    EmbeddingRequestInput input;
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

struct RerankingRequestInputContent {
    std::optional<std::string> text;
    std::optional<std::string> image;
    std::optional<std::string> video;
};

using RerankingRequestInput = std::variant<std::string, RerankingRequestInputContent>;

struct RerankingRequest {
    std::string model;
    std::optional<std::string> instruct;
    RerankingRequestInput query;
    std::vector<RerankingRequestInput> documents;
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
    std::string b_think;
    std::string e_think;
    std::string b_tool_call;
    std::string e_tool_call;
    std::string enable_thinking;
};

} // namespace text
} // namespace caiwei

#endif // CAIWEI_MEDIA_TEXT_DATA_HPP

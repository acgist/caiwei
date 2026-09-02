#include "caiwei/runtime/llamacpp.hpp"

caiwei::context::LlamaCPPContext::LlamaCPPContext(std::string path) : path(std::move(path)) {
    CW_LOG_I("创建LlamaCPPRuntime: %s", this->path.c_str());
    llama_model_params params = llama_model_default_params();
    this->model = llama_model_load_from_file(this->path.c_str(), params);
    if (this->model != nullptr) {
        this->vocab = llama_model_get_vocab(this->model);
        this->model_chat_template = llama_model_chat_template(this->model, nullptr);
    }
}

caiwei::context::LlamaCPPContext::~LlamaCPPContext() {
    if (this->model) {
        CW_LOG_D("释放LlamaCPPRuntime: %s", this->path.c_str());
        llama_free_model(this->model);
        this->model = nullptr;
    }
}

std::shared_ptr<caiwei::context::ClsContext> caiwei::context::get_cls_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::DetContext> caiwei::context::get_det_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::OBBContext> caiwei::context::get_obb_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::SegContext> caiwei::context::get_seg_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::SemContext> caiwei::context::get_sem_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::PoseContext> caiwei::context::get_pose_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::DepthContext> caiwei::context::get_depth_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::ASRContext> caiwei::context::get_asr_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::TTSContext> caiwei::context::get_tts_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::LLMContext> caiwei::context::get_llm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    if (!std::filesystem::exists(info->path)) {
        CW_LOG_W("LlamaCPP模型无效: %s", info->path.c_str());
        return nullptr;
    }
    return std::make_shared<LLMLlamaCPPContext>(info->path, runtime);
}

std::shared_ptr<caiwei::context::VLMContext> caiwei::context::get_vlm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::MLLMContext> caiwei::context::get_mllm_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::EmbeddingContext> caiwei::context::get_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::RerankingContext> caiwei::context::get_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::VLEmbeddingContext> caiwei::context::get_vl_embedding_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

std::shared_ptr<caiwei::context::VLRerankingContext> caiwei::context::get_vl_reranking_context(const caiwei::context::ContextInfo* info, std::shared_ptr<caiwei::runtime::LlamaCPPRuntime> runtime) {
    return nullptr;
}

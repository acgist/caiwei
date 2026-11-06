import base64

from openai import OpenAI

stream = True
api_key = "api-key"
base_url = "http://127.0.0.1:8888/v1"


def base64_file(file_path: str) -> str:
    with open(file_path, "rb") as stream:
        data = stream.read()
        return base64.b64encode(data).decode("utf-8")


def print_resp(stream, resp):
    usage = None
    if stream:
        choices = {}
        for chunk in resp:
            if not chunk.choices:
                continue
            for choice in chunk.choices:
                delta = choice.delta
                if delta.content:
                    print(delta.content, end="", flush=True)
                    if choice.index not in choices:
                        choices[choice.index] = ""
                    choices[choice.index] += delta.content
            if chunk.usage:
                usage = chunk.usage
        print("")
        for index, text in choices.items():
            print(f"choice[{index}] = {text}")
    else:
        for choice in resp.choices:
            print(f"choice[{choice.index}] = {choice.message.content}")
        usage = resp.usage
    if usage:
        print(f"usage = {usage}")


def test_cls():
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    resp = client.chat.completions.create(
        model="yolo26n-cls",
        stream=stream,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "./samoyed.jpg"},
                        # "image_url": {"url": f"data:image/jpeg;base64,{base64_file('./samoyed.jpg')}"},
                    },
                ],
            },
        ],
    )
    print_resp(stream, resp)


def test_det():
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    resp = client.chat.completions.create(
        model="yolo26n-det",
        stream=stream,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "./acgist.jpg"},
                    },
                ],
            },
        ],
    )
    print_resp(stream, resp)


def test_asr():
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    resp = client.chat.completions.create(
        model="qwen3-asr",
        stream=stream,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": "./caiwei.mp3",
                    },
                ],
            },
        ],
    )
    print_resp(stream, resp)


def test_llm():
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    resp = client.chat.completions.create(
        model="qwen3-llm",
        stream=stream,
        messages=[{"role": "user", "content": "简单解释一下“碧螺萧萧”"}],
    )
    print_resp(stream, resp)


def test_vlm():
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    resp = client.chat.completions.create(
        model="qwen3-vlm",
        stream=stream,
        messages=[
            {"role": "system", "content": "按照用户要求识别图片"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "简单描述图片"},
                    {
                        "type": "image",
                        "image": "./acgist.jpg",
                    },
                ],
            },
        ],
    )
    print_resp(stream, resp)


def test_media():
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    resp = client.chat.completions.create(
        model="qwen3-asr",
        # model="qwen3-vlm",
        # model="yolo26n-cls",
        # model="yolo26n-det",
        stream=True,  # 必须使用stream模式
        extra_body={
            # "media_url": "./caiwei.mp3",
            "media_url": "./caiwei.mp4",
            "media_type": "file",
            "asr_samples": 32000,
            "vlm_frames": 8,
            "drop_on_full": False,
            # "model_list": ["qwen3-asr"],
            # "model_list": ["qwen3-vlm"],
            # "model_list": ["qwen3-asr", "qwen3-vlm"],
        },
        messages=[
            {"role": "system", "content": "这个一个AI识别系统"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "简单描述视频内容"},
                ],
            },
        ],
    )
    print_resp(stream, resp)


"""
$env:CAIWEI_CONTEXT_INFO=@"
CLS,YOLO,yolo26n-cls,yolo26n-cls.onnx
DET,YOLO,yolo26n-det,yolo26n-det.onnx
ASR,QWEN,qwen3-asr,D:/tmp/model/llama.cpp/Qwen3-ASR-0.6B-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf
LLM,QWEN,qwen3-llm,D:/tmp/model/llama.cpp/Qwen3-1.7B-Q4_K_M.gguf
VLM,QWEN,qwen3-vlm,D:/tmp/model/llama.cpp/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-VL-2b-Instruct-Q4_K_M.gguf
EMBEDDING,QWEN,qwen3-embedding,D:/tmp/model/llama.cpp/Qwen3-Embedding-0.6B-Q4_K_M.gguf
RERANKING,QWEN,qwen3-reranking,D:/tmp/model/llama.cpp/Qwen3-Reranker-0.6B-Q4_K_M.gguf
"@

export CAIWEI_CONTEXT_INFO="
CLS,YOLO,yolo26n-cls,yolo26n-cls.onnx
DET,YOLO,yolo26n-det,yolo26n-det.onnx
ASR,QWEN,qwen3-asr,/data/caiwei/deps/llama.cpp/Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-Q4_K_M.gguf|/data/caiwei/deps/llama.cpp/Qwen3-ASR-0.6B/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf
LLM,QWEN,qwen3-llm,/data/caiwei/deps/llama.cpp/Qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf
VLM,QWEN,qwen3-vlm,/data/caiwei/deps/llama.cpp/Qwen3-VL-2B-InstructQwen3-VL-2B-Instruct-Q4_K_M.gguf|/data/caiwei/deps/llama.cpp/Qwen3-VL-2B-Instruct/mmproj-Qwen3-VL-2b-Instruct-Q4_K_M.gguf
EMBEDDING,QWEN,qwen3-embedding,/data/caiwei/deps/llama.cpp/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B-Q4_K_M.gguf
RERANKING,QWEN,qwen3-reranking,/data/caiwei/deps/llama.cpp/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B-Q4_K_M.gguf
"

export CAIWEI_CONTEXT_INFO="
CLS,YOLO,yolo26n-cls,yolo26n-cls-rk3588-f16.rknn
DET,YOLO,yolo26n-det,yolo26n-det-rk3588-f16.rknn
ASR,QWEN,qwen3-asr,/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.rknn|/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.weight|/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.embed.bin|/data/model/Qwen3-ASR-0.6B/llm/Qwen3-ASR-0.6B.tokenizer.gguf|/data/model/Qwen3-ASR-0.6B/asr/Qwen3-ASR-0.6B.rknn|/data/model/Qwen3-ASR-0.6B/asr/Qwen3-ASR-0.6B.weight
LLM,QWEN,qwen3-llm,/data/model/Qwen3-1.7B/Qwen3-1.7B.rknn|/data/model/Qwen3-1.7B/Qwen3-1.7B.weight|/data/model/Qwen3-1.7B/Qwen3-1.7B.embed.bin|/data/model/Qwen3-1.7B/Qwen3-1.7B.tokenizer.gguf
VLM,QWEN,qwen3-vlm,/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.rknn|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.weight|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.embed.bin|/data/model/Qwen3-VL-2B-Instruct/llm/Qwen3-VL-2B-Instruct.tokenizer.gguf|/data/model/Qwen3-VL-2B-Instruct/vlm/Qwen3-VL-2B-Instruct.rknn|/data/model/Qwen3-VL-2B-Instruct/vlm/Qwen3-VL-2B-Instruct.weight
EMBEDDING,QWEN,qwen3-embedding,/data/model/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B.rknn|/data/model/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B.weight|/data/model/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B.embed.bin|/data/model/Qwen3-Embedding-0.6B/Qwen3-Embedding-0.6B.tokenizer.gguf
RERANKING,QWEN,qwen3-reranking,/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.rknn|/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.weight|/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.embed.bin|/data/model/Qwen3-Reranker-0.6B/Qwen3-Reranker-0.6B.tokenizer.gguf
"
"""

if __name__ == "__main__":
    # test_cls()
    # test_det()
    # test_asr()
    # test_llm()
    # test_vlm()
    # test_media()
    for i in range(100):
        test_llm()

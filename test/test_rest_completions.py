from openai import OpenAI


def print_resp(stream, resp):
    usage = None
    if stream:
        full_text = ""
        for chunk in resp:
            if not chunk.choices:
                continue
            for choice in chunk.choices:
                delta = choice.delta
                if delta.content:
                    print(delta.content, end="", flush=True)
                    full_text += delta.content
            if chunk.usage:
                usage = chunk.usage
        print("")
        print(f"full_text = {full_text}")
    else:
        for choice in resp.choices:
            print(f"choice[{choice.index}] = {choice.message.content}")
        usage = resp.usage
    if usage:
        print(f"usage = {usage}")


def test_cls():
    stream = True
    client = OpenAI(api_key="acgist", base_url="http://127.0.0.1:8888/v1")
    resp = client.chat.completions.create(
        model="yolo26n-cls",
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


def test_det():
    stream = True
    client = OpenAI(api_key="acgist", base_url="http://127.0.0.1:8888/v1")
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


def test_seg():
    stream = True
    client = OpenAI(api_key="acgist", base_url="http://127.0.0.1:8888/v1")
    resp = client.chat.completions.create(
        model="yolo26n-seg",
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


def test_pose():
    stream = True
    client = OpenAI(api_key="acgist", base_url="http://127.0.0.1:8888/v1")
    resp = client.chat.completions.create(
        model="yolo26n-pose",
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
    stream = True
    client = OpenAI(api_key="acgist", base_url="http://127.0.0.1:8888/v1")
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
    stream = True
    client = OpenAI(api_key="acgist", base_url="http://127.0.0.1:8888/v1")
    resp = client.chat.completions.create(
        model="qwen3-llm",
        stream=stream,
        messages=[{"role": "user", "content": "碧螺萧萧"}],
    )
    print_resp(stream, resp)


def test_vlm():
    stream = True
    client = OpenAI(api_key="acgist", base_url="http://127.0.0.1:8888/v1")
    resp = client.chat.completions.create(
        model="qwen3-vlm",
        stream=stream,
        messages=[
            {"role": "system", "content": "简单描述图片"},
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
    stream = True
    client = OpenAI(api_key="acgist", base_url="http://127.0.0.1:8888/v1")
    resp = client.chat.completions.create(
        # model="qwen3-asr",
        model="qwen3-vlm",
        # model="yolo26n-det",
        stream=stream,
        extra_body={
            # "media_url": "./caiwei.mp3",
            "media_url": "./caiwei.mp4",
            "media_type": "file",
            "asr_samples": 16000 * 5,
            "drop_on_full": False,
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
SEG,YOLO,yolo26n-seg,yolo26n-seg.onnx
POSE,YOLO,yolo26n-pose,yolo26n-pose.onnx
ASR,QWEN,qwen3-asr,D:/tmp/model/llama.cpp/Qwen3-ASR-0.6B-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf
LLM,QWEN,qwen3-llm,D:/tmp/model/llama.cpp/Qwen3-0.6B-Q8_0.gguf
VLM,QWEN,qwen3-vlm,D:/tmp/model/llama.cpp/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-VL-2B-Instruct-Q4_K_M.gguf
EMBEDDING,QWEN,qwen3-embedding,D:/tmp/model/llama.cpp/Qwen3-Embedding-0.6B-Q4_K_M.gguf
RERANKING,QWEN,qwen3-reranking,D:/tmp/model/llama.cpp/Qwen3-Reranker-0.6B-Q4_K_M.gguf
"@

export CAIWEI_CONTEXT_INFO="
CLS,YOLO,yolo26n-cls,yolo26n-cls.onnx
DET,YOLO,yolo26n-det,yolo26n-det.onnx
SEG,YOLO,yolo26n-seg,yolo26n-seg.onnx
POSE,YOLO,yolo26n-pose,yolo26n-pose.onnx
ASR,QWEN,qwen3-asr,D:/tmp/model/llama.cpp/Qwen3-ASR-0.6B-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-ASR-0.6b-Q4_K_M.gguf
LLM,QWEN,qwen3-llm,D:/tmp/model/llama.cpp/Qwen3-0.6B-Q8_0.gguf
VLM,QWEN,qwen3-vlm,D:/tmp/model/llama.cpp/Qwen3-VL-2B-Instruct-Q4_K_M.gguf|D:/tmp/model/llama.cpp/mmproj-Qwen3-VL-2B-Instruct-Q4_K_M.gguf
EMBEDDING,QWEN,qwen3-embedding,D:/tmp/model/llama.cpp/Qwen3-Embedding-0.6B-Q4_K_M.gguf
RERANKING,QWEN,qwen3-reranking,D:/tmp/model/llama.cpp/Qwen3-Reranker-0.6B-Q4_K_M.gguf
"
"""

# test_cls()
# test_det()
# test_seg()
# test_pose()
# test_asr()
# test_llm()
# test_vlm()
test_media()

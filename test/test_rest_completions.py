from openai import OpenAI

client = OpenAI(api_key="acgist", base_url="http://192.168.1.100:8000/v1")
resp = client.chat.completions.create(
    model="Qwen3-4B",
    # model="Qwen3-ASR-0.6B",
    # model="Qwen3-VL-2B-Instruct",
    # messages=[{"role": "user", "content": "碧螺萧萧"}],
    messages=[
        # {"role": "system", "content": "简单描述图片"},
        {
            "role": "user",
            "content": [
                {"type":"text", "text":"简单描述图片"},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data":"base64",
                        "format": "wav",
                    },
                },
                # {
                #     "type": "image_url",
                #     "image_url": {"url": "data:image/jpeg;base64,...."},
                # },
            ],
        },
    ],
)
print(resp)

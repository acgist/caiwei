import os
import time
import json
import torch
import asyncio

from clip import ffmpeg_clips
from config import cfg
from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration


def save_jsonl(data: list, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as stream:
        for x in data:
            stream.write(f"{json.dumps(x, ensure_ascii=False)}\n")


async def clips(file: str, output: str, jsonl: list):
    async for _, _, path in ffmpeg_clips(file, output):
        output_text = fit(path)
        if output_text is None:
            continue
        print(f"{path} = {output_text}")
        media_type = None
        if path.lower().endswith(".mp4"):
            media_type = "video"
        elif path.lower().endswith(tuple((".jpg", ".jpeg", ".png"))):
            media_type = "image"
        else:
            print(f"不支持的文件类型: {path}")
            continue
        jsonl.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": cfg["vlm"]["prompt"].strip(),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": f"{media_type}",
                                f"{media_type}": f"{path}",
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{output_text}",
                            }
                        ],
                    },
                ]
            }
        )


model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")


def fit(path: str):
    media_type = None
    if path.lower().endswith(".mp4"):
        media_type = "video"
    elif path.lower().endswith(tuple((".jpg", ".jpeg", ".png"))):
        media_type = "image"
    else:
        print(f"不支持的文件类型: {path}")
        return None
    messages = [
        {
            "role": "system",
            "content": cfg["vlm"]["prompt"].strip(),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": f"{media_type}",
                    f"{media_type}": f"{path}",
                },
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


jsonl = []
if __name__ == "__main__":
    index = 0
    timestamp = int(time.time())
    for filename in os.listdir("video"):
        if filename.lower().endswith(".mp4"):
            output = f"clips-{timestamp}-{index}"
            index += 1
        elif filename.lower().endswith(tuple((".jpg", ".jpeg", ".png"))):
            output = f"clips-{timestamp}"
        else:
            print(f"不支持的文件类型: {filename}")
            continue
        os.makedirs(output, exist_ok=True)
        asyncio.run(clips(f"video/{filename}", output, jsonl))
    save_jsonl(jsonl, f"sft-{timestamp}.jsonl")

"""
export PYTHONPATH="."; python lora/sft.py
"""

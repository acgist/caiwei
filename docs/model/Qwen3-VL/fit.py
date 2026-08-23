import torch

from peft import PeftModel
from config import cfg
from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    model, "Qwen3-VL-4B-Instruct-lora/v0-20260715-094947/checkpoint-100"
)
model.eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

messages = [
    {
        "role": "system",
        "content": cfg["vlm"]["prompt"].strip(),
    },
    {
        "role": "user",
        "content": [
            # {
            #     "type": "image",
            #     "image": "image.jpg",
            # },
            {
                "type": "video",
                "video": "video.mp4",
            },
            # {"type": "text", "text": "描述视频内容"},
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
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output_text)
print(output_text[0])

"""
export PYTHONPATH="."; python lora/fit.py
"""

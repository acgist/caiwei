import torch

from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.float16,
    device_map="auto",
    max_new_tokens=1024,
    max_inference_batch_size=32,
)
results = model.transcribe(
    audio="../../../test/caiwei.mp3",
    language=None,
)
print(f"language:\n{results[0].language}")
print(f"text:\n{results[0].text}")

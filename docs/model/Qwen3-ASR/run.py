import time
import torch

from qwen_asr import Qwen3ASRModel


def get_memory():
    reserved = 0
    allocated = 0
    gpu_count = torch.cuda.device_count()
    for gpu_id in range(gpu_count):
        reserved += torch.cuda.memory_reserved(gpu_id)
        allocated += torch.cuda.memory_allocated(gpu_id)
    print(f"cuda memory reserved: {(reserved / (1024**3)):.2f} GB")
    print(f"cuda memory allocated: {(allocated / (1024**3)):.2f} GB")


tt = time.time()
model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype="auto",
    device_map="auto",
    max_new_tokens=256,
    max_inference_batch_size=32,
)
print(f"dtype: {model.dtype}")
print(f"load model: {(time.time() - tt):.2f} s")
tt = time.time()
results = model.transcribe(
    audio="audio.mp3",
    language=None,
)
print(f"transcribe: {(time.time() - tt):.2f} s")
get_memory()
print(f"language:\n{results[0].language}")
print(f"text:\n{results[0].text}")
input("按回车键退出...")

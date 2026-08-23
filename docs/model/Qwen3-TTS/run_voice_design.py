import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


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
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    dtype=torch.float16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
)
print(f"load model: {(time.time() - tt):.2f} s")
print(f"supported speakers: {model.get_supported_speakers()}")
print(f"supported languages: {model.get_supported_languages()}")
tt = time.time()
wavs, sr = model.generate_voice_design(
    text="请PSG LGD挑选英雄",
    language="Chinese",
    instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
)
print(f"generate_voice_design: {(time.time() - tt):.2f} s")
get_memory()
sf.write("voice_design.wav", wavs[0], sr)

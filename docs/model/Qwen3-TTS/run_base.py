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
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    dtype=torch.float16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
)
print(f"load model: {(time.time() - tt):.2f} s")
print(f"supported speakers: {model.get_supported_speakers()}")
print(f"supported languages: {model.get_supported_languages()}")
tt = time.time()
ref_audio = "audio.mp3"
ref_text = "请PSG LGD挑选英雄"
wavs, sr = model.generate_voice_clone(
    text="为什么不Ban猛犸",
    language="Chinese",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
print(f"generate_voice_clone: {(time.time() - tt):.2f} s")
get_memory()
sf.write("voice_clone.wav", wavs[0], sr)

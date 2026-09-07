import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    dtype=torch.float16,
    device_map="cuda:0",
)
wavs, sr = model.generate_custom_voice(
    text="请PSG LGD挑选英雄",
    speaker="Vivian",
    language="Chinese",
    instruct="用特别愤怒的语气",
)
sf.write("voice_custom.wav", wavs[0], sr)

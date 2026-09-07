import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    dtype=torch.float16,
    device_map="cuda:0",
)
wavs, sr = model.generate_voice_design(
    text="请PSG LGD战队挑选英雄",
    language="Chinese",
    instruct="史诗般恢宏的女声",
)
sf.write("voice_design.wav", wavs[0], sr)

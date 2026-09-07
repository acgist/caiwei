import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    dtype=torch.float16,
    device_map="cuda:0",
)
wavs, sr = model.generate_voice_clone(
    text="咕咕嘎嘎，今晚的饭真是好吃。",
    language="Chinese",
    ref_text="一名极限求生专家，为了挑战自身极限，我来到世界之巅，珠穆朗玛峰。",
    ref_audio="../../../test/caiwei.mp3",
)
sf.write("voice_clone.wav", wavs[0], sr)

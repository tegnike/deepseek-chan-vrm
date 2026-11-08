"""Qwen3-TTS を PyTorch で動かす（Windows / Linux 用）。NVIDIA GPU があれば使い、なければ CPU で動きます。"""
import numpy as np

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"


def load(reference, reference_text):
    """参照音声を読み込み、generate(text, seed, max_new_tokens) -> (wav, sample_rate) を返します。"""
    import torch
    from qwen_tts import Qwen3TTSModel

    if torch.cuda.is_available():
        device, dtype = "cuda:0", torch.bfloat16
    else:
        print("NVIDIA GPU が見つからないので CPU で動かします。生成にかなり時間がかかります。", flush=True)
        device, dtype = "cpu", torch.float32
    model = Qwen3TTSModel.from_pretrained(
        MODEL, revision=MODEL_REVISION, device_map=device, dtype=dtype, attn_implementation="sdpa",
    )
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(reference), ref_text=reference_text, x_vector_only_mode=False,
    )

    def generate(text, seed, max_new_tokens):
        torch.manual_seed(seed)
        wavs, sample_rate = model.generate_voice_clone(
            text=text, language="Chinese", voice_clone_prompt=prompt,
            non_streaming_mode=True, max_new_tokens=max_new_tokens,
        )
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1), sample_rate

    return generate

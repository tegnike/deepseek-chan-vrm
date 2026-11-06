"""DeepSeek-chan の声で、原稿を1行ずつ独立して生成する（Apple Silicon の Mac + Metal）。

- 参照は同梱の中国語音声 reference.wav と、その台本に固定します。
- 言語は chinese に固定します。日本語の原稿を渡すと、中国語訛り風の日本語になります。
- 前の行の出力を次の行に渡しません。行ごとにシードも同じ値へ戻します。
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

GGUF_REPO = "Serveurperso/Qwen3-TTS-GGUF"
GGUF_REVISION = "b7ee2e8c7459c3bea99da23e3d178125a7d1713c"
REFERENCE_SHA256 = "d14c15bb975fab6dee2af4252598b922f98a0c76876763ff65b58fbf8793b464"
REFERENCE_TEXT = "你好呀！今天也来找我聊天啦？我刚刚学会了一点新本领，想第一个分享给你。准备好了吗？那我们开始吧！"
SAMPLE_RATE = 24000
FRAMES_PER_SECOND = 12.5


def parse_lines(value):
    return {int(number) for number in value.split(",")}


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True, help="1行1セリフのUTF-8テキスト")
    parser.add_argument("--out", type=Path, default=Path("take-01"), help="新しい出力フォルダ")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--lines", type=parse_lines, help="生成する行番号。例: 3,6,7（既定は全行）")
    parser.add_argument("--quant", choices=["Q4_K_M", "Q8_0", "BF16"], default="Q4_K_M")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="200で約16秒。止まらない生成を打ち切る")
    parser.add_argument("--gap", type=float, default=0.18, help="joined.wav の行間の無音（秒）")
    args = parser.parse_args()

    reference = here / "reference.wav"
    if hashlib.sha256(reference.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("reference.wav が同梱のものと違います")
    lines = [line.strip() for line in args.script.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError("原稿が空です")
    args.out.mkdir(parents=True, exist_ok=False)

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ["GGML_BACKEND"] = "MTL0"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import numpy as np
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from qwentts_cpp import QwenTTS

    talker = hf_hub_download(GGUF_REPO, f"qwen-talker-1.7b-base-{args.quant}.gguf", revision=GGUF_REVISION)
    codec = hf_hub_download(GGUF_REPO, f"qwen-tokenizer-12hz-{args.quant}.gguf", revision=GGUF_REVISION)
    tts = QwenTTS(talker, codec, log_level="warning")
    audio, sr = sf.read(reference, dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"参照音声のサンプルレートが想定外です: {sr}")
    voice = tts.extract_voice_ref(audio)

    limit_seconds = args.max_new_tokens / FRAMES_PER_SECOND
    rows, parts = [], []
    for number, text in enumerate(lines, 1):
        if args.lines and number not in args.lines:
            continue
        started = time.perf_counter()
        chunks = []
        options = dict(
            text=text, lang="chinese", ref_spk_emb=voice.ref_spk_emb, ref_codes=voice.ref_codes,
            ref_text=REFERENCE_TEXT, seed=args.seed, max_new_tokens=args.max_new_tokens,
            repetition_penalty=1.05, temperature=0.9, top_k=50, top_p=1.0,
            subtalker_do_sample=True, subtalker_temperature=0.9, subtalker_top_k=50, subtalker_top_p=1.0,
        )
        for data, sr in tts.stream(first_chunk_frames=4, codec_chunk_sec=0.32, **options):
            data = np.asarray(data, dtype=np.float32).reshape(-1)
            if data.size:
                chunks.append(data)
        if not chunks:
            raise RuntimeError(f"{number}行目の音声が空です")
        data = np.concatenate(chunks)
        if sr != SAMPLE_RATE or not np.isfinite(data).all():
            raise RuntimeError(f"{number}行目の音声が不正です")
        name = f"line{number:02d}.wav"
        sf.write(args.out / name, data, sr, subtype="PCM_16")
        duration = len(data) / sr
        truncated = duration >= limit_seconds - 0.1
        rows.append(dict(
            file=name, text=text, duration_s=round(duration, 3),
            generation_s=round(time.perf_counter() - started, 3), hit_token_limit=truncated,
        ))
        parts.append(data)
        note = "  ※上限で打ち切り。--seed を変えて作り直してください" if truncated else ""
        print(f"{name}  {duration:5.2f}s  生成 {rows[-1]['generation_s']:5.2f}s  {text}{note}", flush=True)
    tts.close()
    if not parts:
        raise ValueError("--lines に該当する行がありません")

    silence = np.zeros(round(args.gap * SAMPLE_RATE), dtype=np.float32)
    joined = [piece for part in parts for piece in (part, silence)][:-1]
    sf.write(args.out / "joined.wav", np.concatenate(joined), SAMPLE_RATE, subtype="PCM_16")
    manifest = dict(
        seed=args.seed, language="chinese", quant=args.quant, gguf_repo=GGUF_REPO, gguf_revision=GGUF_REVISION,
        reference_sha256=REFERENCE_SHA256, max_new_tokens=args.max_new_tokens, gap_s=args.gap, lines=rows,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"DONE: {args.out / 'joined.wav'}")


if __name__ == "__main__":
    main()

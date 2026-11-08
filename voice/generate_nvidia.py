"""DeepSeek-chan の声で、原稿を1行ずつ独立して生成する（Windows / Linux の PyTorch 版）。

使い方は generate_mac.py と同じです。前の行の出力は次の行に渡さず、行ごとにシードも同じ値へ戻します。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from pytorch_backend import MODEL, MODEL_REVISION, load

REFERENCE_TEXT = "你好呀！今天也来找我聊天啦？我刚刚学会了一点新本领，想第一个分享给你。准备好了吗？那我们开始吧！"
FRAMES_PER_SECOND = 12.5


def parse_lines(value):
    return {int(number) for number in value.split(",")}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True, help="1行1セリフのUTF-8テキスト")
    parser.add_argument("--out", type=Path, default=Path("take-01"), help="新しい出力フォルダ")
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--lines", type=parse_lines, help="生成する行番号。例: 3,6,7（既定は全行）")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="200で約16秒。止まらない生成を打ち切る")
    parser.add_argument("--gap", type=float, default=0.18, help="joined.wav の行間の無音（秒）")
    args = parser.parse_args()

    lines = [line.strip() for line in args.script.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError("原稿が空です")
    args.out.mkdir(parents=True, exist_ok=False)

    generate = load(here / "reference.wav", REFERENCE_TEXT)
    limit_seconds = args.max_new_tokens / FRAMES_PER_SECOND
    rows, parts, rate = [], [], None
    for number, text in enumerate(lines, 1):
        if args.lines and number not in args.lines:
            continue
        started = time.perf_counter()
        data, rate = generate(text, args.seed, args.max_new_tokens)
        name = f"line{number:02d}.wav"
        sf.write(args.out / name, data, rate, subtype="PCM_16")
        duration = len(data) / rate
        truncated = duration >= limit_seconds - 0.1
        rows.append(dict(
            file=name, text=text, duration_s=round(duration, 3),
            generation_s=round(time.perf_counter() - started, 3), hit_token_limit=truncated,
        ))
        parts.append(data)
        note = "  ※上限で打ち切り。--seed を変えて作り直してください" if truncated else ""
        print(f"{name}  {duration:5.2f}s  生成 {rows[-1]['generation_s']:5.2f}s  {text}{note}", flush=True)
    if not parts:
        raise ValueError("--lines に該当する行がありません")

    silence = np.zeros(round(args.gap * rate), dtype=np.float32)
    joined = [piece for part in parts for piece in (part, silence)][:-1]
    sf.write(args.out / "joined.wav", np.concatenate(joined), rate, subtype="PCM_16")
    manifest = dict(
        seed=args.seed, language="chinese", model=MODEL, model_revision=MODEL_REVISION,
        max_new_tokens=args.max_new_tokens, gap_s=args.gap, lines=rows,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"DONE: {args.out / 'joined.wav'}")


if __name__ == "__main__":
    main()

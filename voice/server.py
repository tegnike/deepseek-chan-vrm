"""AITuberKitのGSVI設定から呼べる、中国語訛り風TTSのローカルサーバー（Apple Silicon）。

- POST /tts に {"text": "..."} を受け取り、WAVを返します。
- 1リクエストを1行として独立に生成します。前の出力は次へ渡さず、シードも毎回戻します。
- 生成が止まらず上限まで伸びた行は、文末の読点を句点に替え、シードを変えて作り直します。
"""
import argparse
import io
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["GGML_BACKEND"] = "MTL0"
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from qwentts_cpp import QwenTTS

GGUF_REPO = "Serveurperso/Qwen3-TTS-GGUF"
GGUF_REVISION = "b7ee2e8c7459c3bea99da23e3d178125a7d1713c"
REFERENCE_TEXT = "你好呀！今天也来找我聊天啦？我刚刚学会了一点新本领，想第一个分享给你。准备好了吗？那我们开始吧！"
SEED = 20261030
FRAMES_PER_SECOND = 12.5
MAX_NEW_TOKENS = 200  # 約16秒
ATTEMPTS = 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("reference.wav"), help="参照音声（24 kHz、モノラルのWAV）")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    talker = hf_hub_download(GGUF_REPO, "qwen-talker-1.7b-base-Q4_K_M.gguf", revision=GGUF_REVISION)
    codec = hf_hub_download(GGUF_REPO, "qwen-tokenizer-12hz-Q4_K_M.gguf", revision=GGUF_REVISION)
    tts = QwenTTS(talker, codec, log_level="warning")
    audio, _ = sf.read(args.reference, dtype="float32")
    voice = tts.extract_voice_ref(audio)

    def generate(text, seed, tokens):
        chunks = []
        for data, sample_rate in tts.stream(
            text=text, lang="chinese",
            ref_spk_emb=voice.ref_spk_emb, ref_codes=voice.ref_codes,
            ref_text=REFERENCE_TEXT, seed=seed, max_new_tokens=tokens,
            repetition_penalty=1.05, temperature=0.9, top_k=50, top_p=1.0,
            subtalker_do_sample=True, subtalker_temperature=0.9,
            subtalker_top_k=50, subtalker_top_p=1.0,
            first_chunk_frames=4, codec_chunk_sec=0.32,
        ):
            chunks.append(np.asarray(data, dtype=np.float32).reshape(-1))
        return np.concatenate(chunks), sample_rate

    def synthesize(text):
        # 普通の文は1文字0.3秒以内に収まるので、上限は文の長さに合わせる
        tokens = min(MAX_NEW_TOKENS, max(50, len(text) * 5))
        for attempt in range(ATTEMPTS):
            line = text if attempt == 0 else text.rstrip("、,") + "。"
            wav, sample_rate = generate(line, SEED + attempt, tokens)
            seconds = len(wav) / sample_rate
            if seconds < tokens / FRAMES_PER_SECOND - 0.1:
                break
            print(f"上限で打ち切り（{attempt + 1}回目）: {text}", flush=True)
        output = io.BytesIO()
        sf.write(output, wav, sample_rate, format="WAV", subtype="PCM_16")
        return output.getvalue(), seconds

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            text = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["text"]
            started = time.perf_counter()
            body, seconds = synthesize(text)
            print(f"{seconds:5.2f}s  生成 {time.perf_counter() - started:5.2f}s  {text}", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    print(f"READY http://127.0.0.1:{args.port}/tts", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

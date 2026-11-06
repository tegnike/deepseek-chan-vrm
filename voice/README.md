# DeepSeek-chan の声の作り方（Qwen3-TTS）

DeepSeek-chan の声を、音声合成モデル [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) で作る手順です。中国語のセリフと、中国語訛り風の日本語を作れます。

> [!NOTE]
> この声は合成音声です。実在の人の声を学習したものではなく、DeepSeek の公式の声でもありません。

## サンプルボイス

GitHub のファイル画面で再生できます。どれも同じ自己紹介です。

| 言語 | 構成 | 音声 |
| --- | --- | --- |
| 中国語訛り風の日本語 | Mac・Metal・Q4 | [intro-mac-q4.mp3](samples/intro-mac-q4.mp3) |
| 中国語訛り風の日本語 | NVIDIA GPU・PyTorch・BF16 | [intro-pytorch-bf16.mp3](samples/intro-pytorch-bf16.mp3) |
| 中国語 | Mac・Metal・Q4 | [intro-chinese-mac-q4.mp3](samples/intro-chinese-mac-q4.mp3) |
| 中国語 | NVIDIA GPU・PyTorch・BF16 | [intro-chinese-pytorch-bf16.mp3](samples/intro-chinese-pytorch-bf16.mp3) |

声のもとになる参照音声は [reference.wav](reference.wav) です。

## 作り方の概要

1. **声を作る**：VoiceDesign モデルに、声の指示文と中国語のセリフを渡して参照音声を作ります。この工程は一度だけで、同梱の `reference.wav` がその結果です。
2. **声を固定する**：Base モデルの音声クローンに、`reference.wav` とその台本を渡します。追加学習はしません。
3. **原稿を読ませる**：言語を `Chinese` に固定し、原稿を1行ずつ独立して生成します。
4. **仕上げる**：行をつなげたあと、全体の音量を揃えます。

中国語の原稿を渡すと、中国語で話します。**日本語の原稿をそのまま渡すと、中国語訛り風の日本語になります。** 翻訳や、訛りをカタカナで書く必要はありません。

訛りを出すポイントは次の2つです。指示文で「訛って」と頼んでも、自然な日本語のままでした。

- **中国語の参照音声で日本語を読ませる。**
- **1行ずつ短く生成する。** 長文を一括で生成すると、後半ほど日本語が流暢に聞こえました。前の行の出力は次の行に渡さず、シードも行ごとに同じ値へ戻します。なぜ分割で訛りが残るのかは分かっていません。

「中国語訛り風」は聞こえ方の話で、実際の中国語話者の発音を再現するものではありません。語尾の崩れや読み間違いも起きます。

## Mac で生成する

macOS 14 以降の Apple Silicon で動きます。初回は約1.4 GBのモデルを取得します。

```bash
cd voice
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python qwentts-cpp-python==0.5.0 numpy==2.5.4 soundfile==0.14.0 huggingface-hub==2.4.0
```

原稿を UTF-8 のテキストで用意します。1行に1セリフです。

```text
こんにちは、ディープシークちゃんです！
推論やプログラミングが得意な、ディープシークのファンアバターです。
青いロングヘアと小さなクジラが、私のトレードマークです。
```

```bash
.venv/bin/python generate_mac.py --script lines.txt --out take-01

# 一部の行だけ、シードを変えて作り直す
.venv/bin/python generate_mac.py --script lines.txt --lines 2,3 --seed 20261031 --out take-02
```

出力先に、行ごとの `lineNN.wav`、行をつないだ `joined.wav`、設定と所要時間を記録した `manifest.json` ができます。M2 Pro（メモリ32 GiB）では、音声の長さの約半分の時間で生成できました（5.28秒の音声で2.63秒）。

### 設定

`generate_mac.py` の既定値です。

| 項目 | 値 |
| --- | --- |
| モデル | [`Serveurperso/Qwen3-TTS-GGUF`](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF) の 1.7B Base（リビジョン `b7ee2e8c7459c3bea99da23e3d178125a7d1713c`） |
| 実装 | `qwentts-cpp-python==0.5.0`（`GGML_BACKEND=MTL0`） |
| 精度 | talker、codec とも `Q4_K_M` |
| 言語 | `chinese` |
| シード | `20261030`（行ごと） |
| `max_new_tokens` | `200`（約16秒） |
| サンプリング | `temperature=0.9`、`top_k=50`、`top_p=1.0`、`repetition_penalty=1.05` |
| subtalker | `do_sample=True`、`temperature=0.9`、`top_k=50`、`top_p=1.0` |
| 行間の無音 | 0.18秒 |

## NVIDIA GPU で生成する

Windows、Python 3.12.9、RTX 4070（VRAM 12 GB）で作ったときの設定です。バージョンは `qwen-tts==0.1.1`、`transformers==4.57.3`、`torch==2.14.1+cu130` です。7行の生成に約81秒かかりました。

```python
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
)
prompt = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text="你好呀！今天也来找我聊天啦？我刚刚学会了一点新本领，想第一个分享给你。准备好了吗？那我们开始吧！",
    x_vector_only_mode=False,
)
for line in lines:
    torch.manual_seed(20261030)
    wavs, sr = model.generate_voice_clone(
        text=line, language="Chinese", voice_clone_prompt=prompt,
        non_streaming_mode=True, max_new_tokens=2048,
    )
```

モデルのリビジョンは `fd4b254389122332181a7c3db7f27e918eec64e3` です。同じシードでも、Mac と GPU では別の音になります。

## 参照音声の作り方

`reference.wav` は次の条件で作りました。同じ声を使うなら、この工程は不要です。

| 項目 | 値 |
| --- | --- |
| モデル | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`（リビジョン `5ecdb67327fd37bb2e042aab12ff7391903235d3`） |
| API | `generate_voice_design()` |
| 言語 | `Chinese` |
| シード | `20261029` |
| `max_new_tokens` | `1024` |

指示文（`instruct`）：

```text
年轻成年女性，二十岁左右，二次元动画女主角的可爱声线，音调偏高但不尖锐，音色明亮清甜，咬字清晰，情绪活泼好奇，语调有自然的起伏，像亲切地和熟悉的朋友聊天。标准普通话，录音干净，没有背景音乐。
```

日本語の意味：20歳前後の成人女性。アニメのヒロインのような可愛い声で、高めだが尖らず、明るく澄んだ甘い音色。発音は明瞭で、活発さと好奇心があり、親しい友人と話すように自然な抑揚を付ける。標準中国語、背景音楽のないきれいな録音。

セリフ（参照音声の台本）：

```text
你好呀！今天也来找我聊天啦？我刚刚学会了一点新本领，想第一个分享给你。准备好了吗？那我们开始吧！
```

日本語の意味：やっほー！ 今日もおしゃべりしに来てくれたの？ さっき新しいことを覚えたから、最初にあなたに教えたくて。準備はいい？ それじゃ、始めよう！

**指示文とシードは、声の ID ではありません。** セリフを変えて同じ指示文を渡しても、同じ声になるとは限らないので、できあがった WAV を参照として保存して使います。

`reference.wav` は 8.4秒、24 kHz、モノラル、16bit PCM です。SHA-256 は `d14c15bb975fab6dee2af4252598b922f98a0c76876763ff65b58fbf8793b464` です。

## 仕上げ

### 音量

行をつないだあとの全体に、FFmpeg の `loudnorm` を2パスでかけます。サンプルボイスは `I=-18`、`TP=-1.5`、`LRA=11` です。動画では -14 LUFS に揃えました。話速とピッチは変えません。動画の尺に合わせて `atempo=1.12` で速めても、声質の劣化は気になりませんでした。

### 読み間違いを直す

崩れもこの声の表現なので、意味が通るものはそのまま残します。言葉が変わってしまう読み間違いは、音声用の原稿で直します。字幕は別に持ちます。

| 崩れ方 | 直し方 | 例 |
| --- | --- | --- |
| 漢字を中国語読みする | その語だけかなにする | 「青い髪」→「あおいかみ」、「尻尾」→「しっぽ」 |
| 語がつながって崩れる | 読点を足す | 「ニケのおごり」→「ニケの、おごり」 |
| 言い回しごと崩れる | 言い回しを変える | 「ちょっと、考える」→「うーん、かんがえる」 |
| 別の語に聞こえる | シードを変えて探す | 複数のテイクを作って選ぶ |

かなにしても、訛りに目立つ差はありませんでした。「你好！」のような中国語の語は、日本語の原稿に混ぜても中国語として読めました。

音声認識は訛りのある声を誤認識することがあるので、テイクは耳で選びます。

### 生成が止まらないとき

ある1行が約164秒まで伸びたことがあります。`max_new_tokens=200` で打ち切れます。上限に達した行は、画面の表示と `manifest.json` の `hit_token_limit` で分かるので、シードを変えて作り直してください。

## 確認状況

- `generate_mac.py` は、2026-11-06 に新しい環境でモデルを取得し直して実行しました。同じ条件の7行は、2026-10-30 に作った音声と PCM で完全に一致しました。
- GPU 用のコードは最初の制作で使ったものの抜粋で、抜粋の形では実行していません。
- 中国語を母語とする人による発音の確認はしていません。

## 利用条件

`voice/` の音声、手順書、スクリプトは、モデルと同じ [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.ja) で公開しています。クレジットの表記方法は、[リポジトリの README](../README.md#クレジット) をご覧ください。

音声合成モデルと実装は、それぞれの配布元の条件に従います。Qwen3-TTS のモデルと、GGUF 版は Apache-2.0、`qwentts-cpp-python` は MIT です。

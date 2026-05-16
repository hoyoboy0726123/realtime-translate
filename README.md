# Realtime Translate · 即時雙語翻譯

Real-time bilingual speech translation with live subtitles, recording, and
post-session meeting summaries.

即時雙語語音翻譯應用，提供即時字幕、錄音，以及會後的會議摘要。

**[English](#english) · [繁體中文](#繁體中文)**

---

## English

Speak in language A or B — the screen shows live subtitles in **both** languages
at once. The two subtitle panes are *language-fixed*: one pane is always
language A, the other always language B, no matter who is speaking. Subtitles
roll like a marquee (newest line enters, older lines scroll away). The layout
can be **vertical (top/bottom)** or **horizontal (left/right)** — toggle it from
the control bar.

Every session is recorded. Afterwards a recording can be **analysed** —
speaker diarization, re-transcription, translation, and a local-LLM meeting
summary.

- **Frontend** — Next.js (App Router + TypeScript)
- **Backend** — Python FastAPI with WebSocket streaming
- **Engines** — cloud / local / mock, switchable in the admin page

### Architecture

```
browser mic ──16kHz PCM16──▶ WebSocket ──▶ FastAPI ──▶ translation engine
     ▲                                                       │
     └──────────── bilingual subtitles (partial / final) ◀────┘
                                                             │
                                                    SQLite transcript log
                                                             │
                                          recording (.wav) ──▶ analysis
                                       (diarization · translation · summary)
```

### Engines

| Engine  | Description                                                   | Needs                          |
| ------- | ------------------------------------------------------------- | ------------------------------ |
| `cloud` | OpenAI Realtime Translation API (`gpt-realtime-translate`)    | `OPENAI_API_KEY`               |
| `local` | On-device Whisper (ASR) + NLLB-200 (translation)              | platform install — see below   |
| `mock`  | No key / no model — replays a sample dialogue to check the UI | nothing                        |

The local engine and recording analysis use a **dual backend**, auto-detected:

- **Apple Silicon Mac** → MLX-Whisper + mlx-lm (native acceleration)
- **Windows / Linux** → faster-whisper (CTranslate2) + llama-cpp-python (NVIDIA CUDA supported)

> The cloud engine works best as a **Chinese-primary** setup. OpenAI's
> translation API is a continuous one-directional interpreter and cannot
> cleanly produce fully symmetric bidirectional subtitles — use the local
> engine for that.

### Requirements

**Software versions — read this before installing.**

| Component      | Version            | Notes                                                                                                          |
| -------------- | ------------------ | -------------------------------------------------------------------------------------------------------------- |
| Python         | **3.10 or 3.11**   | **Do NOT use 3.12+** — the local engine's `torch 2.2` has no 3.12 wheels. Create the virtualenv with **Python 3.11**. |
| Node.js        | **20 or newer**    | frontend build / dev server                                                                                    |
| `torch`        | pinned **2.2.x**   | do not bump — `transformers 4.44` and NLLB depend on it                                                        |
| `transformers` | pinned **4.44.x**  | do not bump — newer releases require torch ≥ 2.4                                                               |

The version pins are deliberate and the `requirements*.txt` files already
enforce them — the one thing you must get right yourself is creating the
virtualenv with **Python 3.11**.

**Recommended hardware.** The `cloud` and `mock` engines run on almost anything;
the **local engine** is the demanding part (Whisper + NLLB + a 7B LLM).
Reference machine, verified: **MacBook Pro M4 Pro · 12-core · 24 GB RAM**.

*macOS — Apple Silicon*

| Use case                 | Minimum                        | Recommended                                            |
| ------------------------ | ------------------------------ | ------------------------------------------------------ |
| cloud / mock only        | any Mac · 8 GB RAM             | —                                                      |
| local engine + analysis  | Apple Silicon (M1) · 16 GB RAM | **Apple Silicon Pro/Max · 24 GB RAM** (M4 Pro — the reference machine) |
| free disk space          | ~20 GB (models + recordings)   | —                                                      |

*Windows / Linux*

| Use case                  | Minimum                                          | Recommended                                         |
| ------------------------- | ------------------------------------------------ | --------------------------------------------------- |
| cloud / mock only         | any modern PC · 8 GB RAM                         | —                                                   |
| local engine — CPU only   | 4-core CPU · 16 GB RAM (live subtitles will lag) | —                                                   |
| local engine — NVIDIA GPU | GPU 6 GB VRAM · 16 GB RAM                        | **NVIDIA GPU ≥ 8 GB VRAM · 8-core CPU · 32 GB RAM**  |
| free disk space           | ~20 GB (models + recordings)                     | —                                                   |

On Windows / Linux an **NVIDIA GPU is strongly recommended** for the local
engine — CPU-only works but live translation may not keep up with real time.

### Install & run

Backend runs on `:8000`, frontend on `:3000`. The three methods below each
install the `cloud` and `mock` engines; the `local` engine is added separately
(see [Local engine](#local-engine)):

- **Method 1 — pip + venv** — standard setup, most control.
- **Method 2 — uv** — fastest dependency install.
- **Method 3 — Docker** — one command, no local Python / Node setup; **Windows / Linux only** (`cloud` + `mock`).

#### Method 1 — native (pip + venv)

```bash
# backend
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in OPENAI_API_KEY for the cloud engine
uvicorn app.main:app --reload --port 8000

# frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

#### Method 2 — uv

[uv](https://docs.astral.sh/uv/) is Astral's fast Python package manager.

```bash
cd backend
uv sync                       # creates .venv from pyproject.toml
cp .env.example .env
uv run uvicorn app.main:app --reload --port 8000

cd ../frontend
npm install
npm run dev
```

#### Method 3 — Docker

> **Recommended for Windows / Linux only.** On macOS use Method 1 or 2 instead —
> Docker Desktop on a Mac runs a CPU-only Linux VM with no access to Apple
> Silicon acceleration, so a Mac gains nothing from the container and loses the
> native local engine.

```bash
cp .env.example .env          # fill in OPENAI_API_KEY for the cloud engine
docker compose up --build
```

The image covers the `cloud` and `mock` engines only. The `local` engine needs
hardware acceleration — on **any** OS it must be run natively (Method 1 or 2).

### Local engine

The local engine does ASR with Whisper and translation with NLLB-200, and
powers recording analysis (speaker diarization + meeting summary). Install the
extra for your platform — the backend auto-detects which one to use.

**Apple Silicon Mac** — ASR and the summary LLM run on Apple's native MLX:

```bash
cd backend
pip install -r requirements-local-mac.txt      # or: uv sync --extra local-mac
```

**Windows / Linux** — ASR uses faster-whisper (CTranslate2), the summary LLM
uses llama-cpp-python (GGUF). Both support NVIDIA CUDA; with a GPU, install the
matching CUDA build of `torch` and `llama-cpp-python` (see their docs).

```bash
cd backend
pip install -r requirements-local-windows.txt  # or: uv sync --extra local-win
```

**Both platforms:**

- First use auto-downloads the models (Whisper, NLLB ≈ 2.4 GB, Qwen2.5-7B ≈ 4–5 GB).
- **Speaker-diarization models must be downloaded manually** into
  `backend/data/diarization/`:
  ```bash
  cd backend/data/diarization
  curl -L -o seg.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  tar xjf seg.tar.bz2 && rm seg.tar.bz2
  curl -L -o emb.onnx "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
  ```
- Force a backend with env var `TRANSLATE_LOCAL_BACKEND=mlx` (Apple) or `ct2`
  (faster-whisper / llama.cpp).
- `transformers` is pinned to **4.44.x** — newer releases require torch ≥ 2.4,
  but this project keeps torch at 2.2.x.

### Usage

1. Open http://localhost:3000
2. Go to **後台設定 / Settings** — choose the engine and lock the two languages
   (12 languages supported; any pair).
3. Back on the main screen, enter a meeting name and press **開始翻譯 / Start** —
   allow microphone access.
4. Speak — both panes show live subtitles. Use the control-bar button to switch
   between vertical and horizontal layout.
5. Open **翻譯記錄 / History** to browse past sessions, export Markdown / JSON,
   and **分析錄音 / Analyze recording** (speaker-labelled transcript + summary).

> The browser microphone only works on `localhost` or over HTTPS.

### Environment variables

| Variable                  | Purpose                                       | Default                  |
| ------------------------- | --------------------------------------------- | ------------------------ |
| `OPENAI_API_KEY`          | OpenAI key for the `cloud` engine             | (none)                   |
| `OPENAI_REALTIME_MODEL`   | OpenAI Realtime model name                    | `gpt-realtime-translate` |
| `CORS_ORIGINS`            | Allowed frontend origins (comma-separated)    | `http://localhost:3000`  |
| `NEXT_PUBLIC_API_BASE`    | Backend address the frontend connects to      | `http://localhost:8000`  |
| `TRANSLATE_LOCAL_BACKEND` | Force the local backend: `mlx` or `ct2`       | auto-detect              |

### Project structure

```
realtime-translate/
├── docker-compose.yml
├── backend/
│   ├── pyproject.toml                  # uv dependency definitions
│   ├── requirements.txt                # base deps (cloud + mock)
│   ├── requirements-local-mac.txt      # local deps — Apple Silicon / MLX
│   ├── requirements-local-windows.txt  # local deps — Windows·Linux / CTranslate2
│   ├── Dockerfile
│   └── app/
│       ├── main.py            # FastAPI entry point
│       ├── ws.py              # /ws/translate WebSocket + audio recording
│       ├── config.py          # settings (persisted as JSON)
│       ├── db.py              # SQLite transcripts + diarized segments
│       ├── languages.py       # supported languages + per-engine codes
│       ├── nllb.py            # NLLB translation (shared)
│       ├── backends/          # cross-platform ASR + LLM (MLX / CTranslate2)
│       ├── engines/           # cloud / local / mock translation engines
│       ├── postprocess/       # recording analysis: diarize / translate / summarize
│       └── routers/           # settings / transcripts API
└── frontend/
    ├── Dockerfile
    ├── app/                   # main screen, /settings, /history
    ├── components/            # RollingSubtitles
    ├── lib/                   # WebSocket / audio capture / API
    └── public/pcm-worklet.js  # mic downsample to 16 kHz PCM16
```

### Notes & limitations

- Subtitles have a `partial` (updating) and `final` (settled) state — only
  `final` lines are written to the transcript database.
- The `cloud` engine depends on OpenAI's online service; the `local` engine
  runs fully on-device with no network.
- The Docker image does not bundle the local-engine dependencies (size / GPU) —
  run the local engine natively.
- Speaker diarization from a single mixed microphone is best-effort
  (~20–40 % error); speaker labels are hints, not guarantees. Per-speaker
  microphones give far better attribution.

> An earlier version used Meta SeamlessStreaming, but it produced wrong output
> on the Apple MPS backend and was ~26x too slow on CPU — hence Whisper.

---

## 繁體中文

說語言 A 或 B，畫面會**同時**顯示兩種語言的即時字幕。兩個字幕窗格「語言固定」：
不論誰說哪種語言，一格永遠是語言 A、另一格永遠是語言 B。字幕以跑馬燈呈現
（新句子進場、舊句子捲出）。版面可選**上下**或**左右**，用控制列的按鈕切換。

每場會議都會錄音。結束後可對錄音做**分析** —— 講者辨識、重新轉錄、翻譯，以及
由本機 LLM 產生的會議摘要。

- **前端** — Next.js（App Router + TypeScript）
- **後端** — Python FastAPI，WebSocket 串流
- **翻譯引擎** — 雲端 / 地端 / 示範，可在後台切換

### 架構

```
瀏覽器麥克風 ──16kHz PCM16──▶ WebSocket ──▶ FastAPI ──▶ 翻譯引擎
     ▲                                                      │
     └──────────── 雙語字幕（partial / final）◀──────────────┘
                                                            │
                                                   SQLite 翻譯記錄
                                                            │
                                          錄音檔（.wav）──▶ 錄音分析
                                       （講者辨識 · 翻譯 · 會議摘要）
```

### 翻譯引擎

| 引擎    | 說明                                                       | 需求                       |
| ------- | ---------------------------------------------------------- | -------------------------- |
| `cloud` | OpenAI Realtime 翻譯 API（`gpt-realtime-translate`）       | `OPENAI_API_KEY`           |
| `local` | 地端 Whisper（語音辨識）+ NLLB-200（翻譯）                 | 依平台安裝，見下方         |
| `mock`  | 免金鑰、免模型，重播範例對話，用來檢視 UI                  | 無                         |

地端引擎與錄音分析採**雙後端**，程式自動偵測平台：

- **Apple Silicon Mac** → MLX-Whisper + mlx-lm（Apple 原生加速）
- **Windows / Linux** → faster-whisper（CTranslate2）+ llama-cpp-python（支援 NVIDIA CUDA）

> 雲端引擎適合作為「**以中文為主**」的設定。OpenAI 翻譯 API 是連續單向口譯模型，
> 無法做到乾淨的對稱雙向字幕 —— 需要完整雙向請用地端引擎。

### 系統需求

**軟體版本 —— 安裝前請務必先看。**

| 元件           | 版本               | 說明                                                                                       |
| -------------- | ------------------ | ------------------------------------------------------------------------------------------ |
| Python         | **3.10 或 3.11**   | **請勿用 3.12 以上** —— 地端引擎的 `torch 2.2` 沒有 3.12 的 wheel。建立虛擬環境請用 **Python 3.11**。 |
| Node.js        | **20 以上**        | 前端建置 / 開發伺服器                                                                      |
| `torch`        | 鎖定 **2.2.x**     | 請勿升級 —— `transformers 4.44` 與 NLLB 依賴它                                              |
| `transformers` | 鎖定 **4.44.x**    | 請勿升級 —— 更新版需要 torch ≥ 2.4                                                          |

版本鎖定是刻意的，`requirements*.txt` 已經寫死；你唯一要自己確保的，是用
**Python 3.11** 建立虛擬環境。

**建議硬體規格。** `cloud` 與 `mock` 引擎幾乎什麼機器都能跑；**地端引擎**才是吃資源的
部分（要同時跑 Whisper + NLLB + 7B LLM）。實測基準機：
**MacBook Pro M4 Pro · 12 核 · 24 GB RAM**。

*macOS — Apple Silicon*

| 使用情境              | 最低                            | 建議                                              |
| --------------------- | ------------------------------- | ------------------------------------------------- |
| 只用 cloud / mock     | 任何 Mac · 8 GB RAM             | —                                                 |
| 地端引擎 + 錄音分析   | Apple Silicon（M1）· 16 GB RAM  | **Apple Silicon Pro/Max · 24 GB RAM**（M4 Pro，即基準機） |
| 可用磁碟空間          | 約 20 GB（模型 + 錄音）         | —                                                 |

*Windows / Linux*

| 使用情境                | 最低                                       | 建議                                              |
| ----------------------- | ------------------------------------------ | ------------------------------------------------- |
| 只用 cloud / mock       | 任何近代 PC · 8 GB RAM                     | —                                                 |
| 地端引擎 — 純 CPU       | 4 核 CPU · 16 GB RAM（即時字幕會延遲）     | —                                                 |
| 地端引擎 — NVIDIA GPU   | GPU 6 GB VRAM · 16 GB RAM                  | **NVIDIA GPU ≥ 8 GB VRAM · 8 核 CPU · 32 GB RAM** |
| 可用磁碟空間            | 約 20 GB（模型 + 錄音）                    | —                                                 |

Windows / Linux 上**強烈建議用 NVIDIA GPU** 跑地端引擎 —— 純 CPU 雖然能跑，但
即時翻譯可能跟不上實際語速。

### 安裝與啟動

後端跑在 `:8000`、前端跑在 `:3000`。下列三種方式都會安裝 `cloud` 與 `mock`
引擎；`local` 引擎另外安裝（見[地端引擎](#地端引擎)）：

- **方式一 — pip + venv** —— 標準安裝，最可控。
- **方式二 — uv** —— 依賴安裝最快。
- **方式三 — Docker** —— 一行指令，免裝 Python / Node；**僅限 Windows / Linux**（`cloud` + `mock`）。

#### 方式一：原生安裝（pip + venv）

```bash
# 後端
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # 使用 cloud 引擎時填入 OPENAI_API_KEY
uvicorn app.main:app --reload --port 8000

# 前端（另開終端機）
cd frontend
npm install
npm run dev
```

開啟 http://localhost:3000

#### 方式二：uv

[uv](https://docs.astral.sh/uv/) 是 Astral 的高速 Python 套件管理器。

```bash
cd backend
uv sync                       # 依 pyproject.toml 建立 .venv
cp .env.example .env
uv run uvicorn app.main:app --reload --port 8000

cd ../frontend
npm install
npm run dev
```

#### 方式三：Docker

> **僅建議在 Windows / Linux 使用。** macOS 請改用方式一或方式二 —— Docker
> Desktop 在 Mac 上是跑一個純 CPU 的 Linux 虛擬機，無法存取 Apple Silicon
> 加速，Mac 用容器不但沒有好處，還會失去原生地端引擎。

```bash
cp .env.example .env          # 使用 cloud 引擎時填入 OPENAI_API_KEY
docker compose up --build
```

容器映像僅涵蓋 `cloud` 與 `mock` 引擎。`local` 引擎需要硬體加速，**任何作業系統**
都必須以方式一或方式二原生執行。

### 地端引擎

地端引擎用 Whisper 做語音辨識、NLLB-200 做翻譯，並支援錄音分析（講者辨識＋會議
摘要）。請依平台安裝對應的依賴 —— 後端會自動偵測該用哪個後端。

**Apple Silicon Mac** —— 語音辨識與摘要 LLM 走 Apple 原生的 MLX：

```bash
cd backend
pip install -r requirements-local-mac.txt      # 或： uv sync --extra local-mac
```

**Windows / Linux** —— 語音辨識走 faster-whisper（CTranslate2）、摘要 LLM 走
llama-cpp-python（GGUF）。兩者都支援 NVIDIA CUDA；有顯卡時請額外安裝對應 CUDA
版的 `torch` 與 `llama-cpp-python`（見各自官方文件）。

```bash
cd backend
pip install -r requirements-local-windows.txt  # 或： uv sync --extra local-win
```

**兩平台共通：**

- 首次使用會自動下載模型（Whisper、NLLB ≈ 2.4 GB、Qwen2.5-7B ≈ 4–5 GB）。
- **講者辨識模型需手動下載**到 `backend/data/diarization/`：
  ```bash
  cd backend/data/diarization
  curl -L -o seg.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  tar xjf seg.tar.bz2 && rm seg.tar.bz2
  curl -L -o emb.onnx "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
  ```
- 強制指定後端：環境變數 `TRANSLATE_LOCAL_BACKEND=mlx`（Apple）或 `ct2`
  （faster-whisper / llama.cpp）。
- `transformers` 鎖定在 **4.44.x** —— 更新版需要 torch ≥ 2.4，而本專案的 torch
  須維持在 2.2.x。

### 使用方式

1. 開啟 http://localhost:3000
2. 進入 **後台設定**：選擇翻譯引擎、鎖定兩種語言（支援 12 種語言、任意配對）。
3. 回到主畫面，輸入會議名稱，按 **開始翻譯**，允許瀏覽器使用麥克風。
4. 開始說話 —— 兩個窗格即時顯示雙語字幕。用控制列按鈕切換上下／左右版面。
5. 到 **翻譯記錄** 瀏覽歷史、匯出 Markdown / JSON，並可 **分析錄音**
   （產生帶講者標籤的逐句記錄與會議摘要）。

> 瀏覽器麥克風需在 `localhost` 或 HTTPS 環境下才能使用。

### 環境變數

| 變數                      | 用途                                  | 預設值                   |
| ------------------------- | ------------------------------------- | ------------------------ |
| `OPENAI_API_KEY`          | `cloud` 引擎用的 OpenAI 金鑰          | （無）                   |
| `OPENAI_REALTIME_MODEL`   | OpenAI Realtime 模型名稱              | `gpt-realtime-translate` |
| `CORS_ORIGINS`            | 允許的前端來源（逗號分隔）            | `http://localhost:3000`  |
| `NEXT_PUBLIC_API_BASE`    | 前端連線的後端位址                    | `http://localhost:8000`  |
| `TRANSLATE_LOCAL_BACKEND` | 強制地端後端：`mlx` 或 `ct2`          | 自動偵測                 |

### 限制與備註

- 字幕有 `partial`（更新中）與 `final`（定稿）兩種狀態，只有 `final` 會寫入
  翻譯記錄資料庫。
- `cloud` 引擎依賴 OpenAI 線上服務；`local` 引擎完全在裝置上執行，不需網路。
- Docker 映像不含地端引擎依賴（體積／GPU 考量），地端引擎請原生執行。
- 單一麥克風混音的講者辨識為盡力而為（約 20–40% 誤差），講者標籤僅供參考；
  每位講者各用一支麥克風會準確得多。

> 早期版本曾採用 Meta SeamlessStreaming，但它在 Apple MPS 後端會產生錯誤輸出、
> 在 CPU 上又慢約 26 倍，因此改用 Whisper。

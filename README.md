# Realtime Translate · 即時雙語翻譯

即時雙語語音翻譯應用。講者說中文時，畫面上方顯示中文字幕、下方顯示翻譯後的英文；
講者改說英文時，下方顯示英文、上方同步翻成中文。字幕以**垂直跑馬燈**呈現 —— 新句子從
底部進場、舊句子往上推出，固定顯示約 3 行大字。所有翻譯都會被記錄下來，方便日後製作
摘要或會議記錄。

- **前端**：Next.js (App Router + TypeScript)
- **後端**：Python FastAPI，WebSocket 串流
- **翻譯引擎**：雲端 / 地端 / 示範 三種可切換

---

## 架構

```
瀏覽器麥克風 ──16kHz PCM16──▶ WebSocket ──▶ FastAPI ──▶ 翻譯引擎
     ▲                                                      │
     └──────────── 雙語字幕（partial / final）◀──────────────┘
                                                            │
                                                   SQLite 翻譯記錄
```

兩個字幕窗格「語言固定」：不論誰說哪種語言，上方永遠是語言 A、下方永遠是語言 B，畫面穩定。

## 翻譯引擎

| 引擎    | 說明                                                          | 需求                                   |
| ------- | ------------------------------------------------------------- | -------------------------------------- |
| `cloud` | OpenAI Realtime API（`gpt-realtime-translate`）               | `OPENAI_API_KEY`                       |
| `local` | 地端 Whisper（語音辨識）+ **NLLB-200**（翻譯）                | 見下方平台安裝說明                     |
| `mock`  | 免金鑰示範，重播範例對話，用來檢視 UI / 跑馬燈 / 記錄         | 無                                     |

地端引擎與「錄音分析」採**雙後端**，程式會自動偵測平台：

- **Apple Silicon Mac** → MLX-Whisper + mlx-lm（Apple 原生加速）
- **Windows / Linux** → faster-whisper（CTranslate2）+ llama-cpp-python（支援 NVIDIA CUDA）

可用環境變數 `TRANSLATE_LOCAL_BACKEND=mlx|ct2` 強制指定後端。

引擎與語言對在後台 `/settings` 頁面切換。預設為 `mock`，可直接體驗。

---

## 安裝與啟動

需求：Python 3.10+、Node.js 20+。後端跑在 `:8000`，前端跑在 `:3000`。
以下三種方式擇一即可。

### 方式一：原生安裝（pip + venv）

**後端**

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # 使用 cloud 引擎時填入 OPENAI_API_KEY
uvicorn app.main:app --reload --port 8000
```

**前端**（另開一個終端機）

```bash
cd frontend
npm install
npm run dev
```

開啟 http://localhost:3000

### 方式二：uv 安裝

[uv](https://docs.astral.sh/uv/) 是 Astral 推出的高速 Python 套件管理器。

**後端**

```bash
cd backend
uv sync                       # 依 pyproject.toml 建立 .venv 並安裝依賴
cp .env.example .env
uv run uvicorn app.main:app --reload --port 8000
```

**前端**（前端為 Node 專案，仍使用 npm）

```bash
cd frontend
npm install
npm run dev
```

### 方式三：Docker（容器安裝）

一次啟動整個堆疊，最省事：

```bash
cp .env.example .env          # 使用 cloud 引擎時填入 OPENAI_API_KEY
docker compose up --build
```

開啟 http://localhost:3000

容器映像涵蓋 `cloud` 與 `mock` 引擎。`local` 引擎需要硬體加速,建議在主機上原生
執行(方式一或方式二),不要放進容器。

---

## 啟用地端引擎（依平台安裝）

地端引擎用 Whisper 做語音辨識、**NLLB-200** 做翻譯,並提供「錄音分析」(講者辨識
＋會議摘要)。請依你的平台擇一安裝 —— 程式會自動偵測並使用對應後端。

### Apple Silicon Mac

語音辨識與摘要 LLM 走 Apple 原生的 **MLX**(快)。

```bash
cd backend
pip install -r requirements-local-mac.txt      # 或： uv sync --extra local-mac
```

### Windows / Linux

語音辨識走 **faster-whisper**(CTranslate2)、摘要 LLM 走 **llama-cpp-python**
(GGUF)。兩者都可用 NVIDIA CUDA 加速 —— 有顯卡時請額外安裝對應的 CUDA 版
`torch` 與 `llama-cpp-python`(見各自官方文件)。

```bash
cd backend
pip install -r requirements-local-windows.txt  # 或： uv sync --extra local-win
```

### 共通說明

- 首次使用會自動下載模型(Whisper、NLLB ≈ 2.4 GB、Qwen2.5-7B ≈ 4–5 GB)。
- **講者辨識模型**需手動下載到 `backend/data/diarization/`:
  ```bash
  cd backend/data/diarization
  # 語音分段模型
  curl -L -o seg.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  tar xjf seg.tar.bz2 && rm seg.tar.bz2
  # 語者嵌入模型
  curl -L -o emb.onnx "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
  ```
- 安裝後到後台把引擎切換為「地端」即可,可在設定頁更換各模型。
- 強制指定後端：環境變數 `TRANSLATE_LOCAL_BACKEND=mlx`(Apple)或 `ct2`
  (faster-whisper / llama.cpp)。
- `transformers` 鎖定在 4.44.x —— 更新版需要 torch ≥ 2.4,而本專案的 torch 須維持
  在 2.2.x。

> 早期版本曾採用 Meta SeamlessStreaming,但它在 Apple Silicon 的 MPS 後端上會產生
> 錯誤輸出、在 CPU 上又慢約 26 倍,因此改用 Whisper。

---

## 使用方式

1. 開啟 http://localhost:3000
2. 進入 **後台設定**：選擇翻譯引擎、鎖定兩種語言。
3. 回到主畫面，輸入會議名稱，按 **開始翻譯**，允許瀏覽器使用麥克風。
4. 開始說話 —— 上下窗格即時顯示雙語字幕。
5. 到 **翻譯記錄** 頁面瀏覽歷史，可匯出 Markdown / JSON。

> 瀏覽器麥克風需在 `localhost` 或 HTTPS 環境下才能使用。

## 環境變數

| 變數                    | 用途                              | 預設值                  |
| ----------------------- | --------------------------------- | ----------------------- |
| `OPENAI_API_KEY`        | `cloud` 引擎用的 OpenAI 金鑰      | （無）                  |
| `OPENAI_REALTIME_MODEL` | OpenAI Realtime 模型名稱          | `gpt-realtime-translate`|
| `CORS_ORIGINS`          | 允許的前端來源（逗號分隔）        | `http://localhost:3000` |
| `NEXT_PUBLIC_API_BASE`  | 前端連線的後端位址（建置時內嵌） | `http://localhost:8000` |

## 專案結構

```
realtime-translate/
├── docker-compose.yml        # 一鍵啟動整個堆疊
├── backend/
│   ├── pyproject.toml        # uv 依賴定義
│   ├── requirements.txt      # pip 依賴定義
│   ├── requirements-local-mac.txt     # 地端依賴（Apple Silicon / MLX）
│   ├── requirements-local-windows.txt # 地端依賴（Windows·Linux / CTranslate2）
│   ├── Dockerfile
│   └── app/
│       ├── main.py           # FastAPI 進入點
│       ├── ws.py             # /ws/translate WebSocket
│       ├── config.py         # 設定（持久化為 JSON）
│       ├── db.py             # SQLite 翻譯記錄
│       ├── languages.py      # 支援語言與各引擎代碼對應
│       ├── routers/          # settings / transcripts API
│       └── engines/          # cloud / local / mock 翻譯引擎
└── frontend/
    ├── Dockerfile
    ├── app/                  # 主畫面、/settings、/history
    ├── components/           # RollingSubtitles 跑馬燈字幕
    ├── lib/                  # WebSocket / 音訊擷取 / API
    └── public/pcm-worklet.js # 麥克風降採樣至 16kHz PCM16
```

## 限制與備註

- 字幕的 `partial`（即時更新中）與 `final`（句子定稿）兩種狀態：只有 `final` 會寫入
  翻譯記錄資料庫。
- `cloud` 引擎依賴 OpenAI Realtime API 的線上服務；`local` 引擎完全在裝置上執行，
  不需要網路。
- Docker 映像不含地端引擎依賴（體積與 GPU 考量），地端引擎請原生執行。

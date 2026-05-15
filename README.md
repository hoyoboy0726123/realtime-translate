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
| `local` | Meta **SeamlessStreaming** 地端逐字同步翻譯                   | `fairseq2` / `simuleval`、MPS 或 CUDA  |
| `mock`  | 免金鑰示範，重播範例對話，用來檢視 UI / 跑馬燈 / 記錄         | 無                                     |

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

容器映像涵蓋 `cloud` 與 `mock` 引擎。`local`（SeamlessStreaming）引擎需要 Apple
Silicon 的 MPS 或 NVIDIA CUDA —— Docker 在 macOS 上無法存取 MPS，因此若要使用地端
引擎，請在 Mac 上以方式一或方式二原生執行後端。

---

## 啟用地端 SeamlessStreaming 引擎

地端引擎使用 Meta 的 `fairseq2` + `simuleval` 串流堆疊，需額外安裝（建議在
MacBook Pro M 系列上原生執行以使用 MPS 加速）：

```bash
cd backend

# pip：
pip install -r requirements-local.txt
pip install "git+https://github.com/facebookresearch/seamless_communication.git"

# 或 uv：
uv sync --extra local
uv pip install "git+https://github.com/facebookresearch/seamless_communication.git"
```

- `seamless_communication` 未發佈於 PyPI，需從 git 安裝。
- 首次使用會自動下載 SeamlessStreaming 權重（`seamless_streaming_unity` 約 2 GB
  與 `seamless_streaming_monotonic_decoder`）。
- 安裝後到後台把引擎切換為「地端」，並可調整音訊區塊大小與 `decision_threshold`
  以權衡延遲與品質。

### 為什麼 Mac 上地端引擎不能用容器

地端 SeamlessStreaming 要靠硬體加速才能達到「即時逐字」。在 MacBook Pro（Apple
Silicon）上，加速來自 **MPS（Apple Metal）**，而 Docker Desktop 在 macOS 是把
Linux 容器跑在一個輕量虛擬機裡 —— **那個虛擬機無法存取 Mac 的 GPU / MPS**。因此
即使把地端引擎放進容器，PyTorch 也只能退回 CPU，SeamlessStreaming 會慢到失去即時
意義。

結論與建議組合：

| 情境                         | 建議做法                                              |
| ---------------------------- | ----------------------------------------------------- |
| 雲端 / 示範引擎               | 直接用 Docker（方式三）最省事                         |
| 在 Mac 上使用地端引擎         | 後端**原生執行**（方式一或方式二），不要放進容器      |
| Linux 主機 + NVIDIA GPU       | 可在容器內用地端引擎，須加 `--gpus all` 並裝 NVIDIA Container Toolkit |

也就是說，「容器不能跑地端引擎」這句話只對 **Mac** 成立；在有 NVIDIA 顯卡的 Linux
主機上，容器是可以把 GPU 通進去的。你使用的是 Mac，所以實務上：要地端翻譯就原生跑後端。

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
│   ├── requirements-local.txt# 地端 SeamlessStreaming 額外依賴
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

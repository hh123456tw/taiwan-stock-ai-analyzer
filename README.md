# 台股 AI 分析系統

台灣股市自動分析與模擬交易系統，整合多模型 ML 預測、技術指標、市場情緒分析，並透過 Telegram 推送訊號。

---

## 系統架構

```
資料來源 (yfinance / twstock / jiansoft)
    ↓
特徵工程
├── technical_indicators.py     技術指標 (RSI, MACD, MA...)
├── market_sentiment_engine.py  市場情緒分析
└── news_sentiment_analyzer.py  新聞情緒 (NLP)
    ↓
預測模型 (可獨立使用)
├── lstm_stock_predictor.py     LSTM 深度學習
├── xgb_stock_predictor.py      XGBoost
├── rf_stock_predictor.py       Random Forest
├── lightgbm_stock_predictor.py LightGBM
└── arima_stock_predictor.py    ARIMA 時間序列
    ↓
決策整合
├── improved_dynamic_weight_learner.py  動態權重 (各模型表現自適應)
└── realtime_trading_system/core/ai_decision_engine.py
        權重：60% AI預測 + 25% 技術指標 + 15% 市場情緒
    ↓
交易訊號 / 通知
└── realtime_trading_system/telegram_notifier.py → Telegram Bot
```

---

## 主要入口

| 用途 | 檔案 |
|------|------|
| 實時模擬交易（盤中執行） | `realtime_trading_system/realtime_trading_main.py` |
| 終極交易系統（動態權重版） | `ultimate_trading_system.py` |
| 盤前分析 | `pre_market_analysis.py` |
| 盤後分析 | `enhanced_real_post_market_analysis.py` |
| 盤前盤後自動排程（Ollama 分析） | `台股盤後盤前自動分析系統_2025-07-31.py` |
| 智能選股 | `smart_stock_screener.py` |
| 單支股票訓練 | `single_stock_trainer.py` |
| 批次訓練所有股票 | `batch_model_trainer.py` |
| 模型效能分析 | `model_performance_analysis.py` |
| 自動再訓練排程 | `retraining_scheduler.py` |

---

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

> TensorFlow 和 PyTorch 擇一安裝即可，不需要同時裝。
> TA-Lib 可選，安裝較複雜，不裝的話技術指標會 fallback 到純 Python 實作。

### 2. 設定 config

```bash
cp config.example.yaml config.yaml
# 編輯 config.yaml，填入 Telegram Bot Token 和 Ollama 位址
```

### 3. 確認 Ollama 服務

系統使用本地 Ollama 做 AI 分析，預設模型是 `phi4:14b`：

```bash
ollama pull phi4:14b         # 高品質分析
ollama pull llama3.1:8b      # 備用，速度較快
```

### 4. 執行

```bash
# 盤前分析
python pre_market_analysis.py

# 盤後分析
python enhanced_real_post_market_analysis.py

# 實時模擬交易（09:00–13:30 執行）
python realtime_trading_system/realtime_trading_main.py

# 自動盤前盤後排程（08:30 / 15:00 自動觸發）
python 台股盤後盤前自動分析系統_2025-07-31.py
```

### 5. 設為系統服務（可選）

```bash
sudo cp taiwan-auto-analysis.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable taiwan-auto-analysis.service
sudo systemctl start taiwan-auto-analysis.service
```

---

## 模型訓練

系統需要先訓練模型才能預測。訓練結果存放在 `models/` 目錄。

```bash
# 訓練單支股票
python single_stock_trainer.py --stock 2330

# 批次訓練所有股票池
python batch_model_trainer.py

# 自動排程（夜間訓練）
python retraining_scheduler.py
```

訓練好的模型不進 git（檔案太大），每台機器需要自己訓練。

---

## 目錄結構

```
taiwan-stock-ai-analyzer/
├── realtime_trading_system/        實時交易系統（核心）
│   ├── core/
│   │   ├── ai_decision_engine.py   AI 決策引擎
│   │   ├── market_scanner_fixed.py 市場掃描器
│   │   ├── technical_indicators.py 技術指標
│   │   └── market_sentiment_engine.py
│   ├── config/trading_config.py    交易參數設定
│   └── telegram_notifier.py        TG 通知
├── backtest/                       回測模組
├── data_sources/                   資料來源介接
├── scripts/                        排程腳本
├── models/                         訓練好的模型（不進 git）
├── logs/                           日誌（不進 git）
├── data/                           SQLite DB（不進 git）
├── config.yaml                     實際設定（不進 git）
├── config.example.yaml             設定範本
└── requirements.txt
```

---

## 注意事項

- 台股交易單位：1 張 = 1000 股，計算成本請注意
- 系統為**模擬交易**，不接實際券商 API，不會真的下單
- 預測結果僅供參考，不構成投資建議
- Ollama 服務需要自行在本地或遠端部署

---

## 依賴外部服務

| 服務 | 用途 | 必要 |
|------|------|------|
| Ollama (phi4:14b) | AI 自然語言分析報告 | 否（沒有則跳過 LLM 分析） |
| Telegram Bot | 推送交易訊號 | 否（沒有則只輸出 console） |
| jiansoft API | 基本面資料 | 否 |
| yfinance | 美股 / 歷史價格 | 是 |
| twstock | 台股即時報價 | 是 |

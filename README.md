# 📈 台股突破訊號掃描器

台股技術指標掃描工具，支援手機瀏覽器，資料來源為 FinMind API。

## 功能
- 自選清單 / 台灣50 / 中型100 掃描
- 五大技術訊號評分：價格突破、量能放大、MACD金叉、RSI強勢、均線多頭
- 互動式線圖：K線、成交量、MACD、RSI
- 手機優化介面，支援向上滑動關閉線圖

## 部署到 Render

1. Fork 或 Clone 此 Repository 到你的 GitHub
2. 前往 [render.com](https://render.com) 登入
3. New → Web Service → 選擇此 Repository
4. 設定如下：
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
5. 點 Deploy，等待約 3 分鐘完成

## 本地執行

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

瀏覽器開啟 http://localhost:8000

## 資料來源

[FinMind](https://finmindtrade.com) 免費開放資料，需申請免費 Token。

## 免責聲明

本工具僅供技術分析參考，不構成任何投資建議。

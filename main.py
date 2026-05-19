from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import json
import os

LC_JS_PATH = "static/lc.js"
LC_CDN_URL = "https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"

def download_lc_if_needed():
    if os.path.exists(LC_JS_PATH) and os.path.getsize(LC_JS_PATH) > 50000:
        print("✅ LightweightCharts 已存在，跳過下載")
        return
    print("📦 下載 LightweightCharts...")
    try:
        r = requests.get(LC_CDN_URL, timeout=30)
        r.raise_for_status()
        os.makedirs("static", exist_ok=True)
        with open(LC_JS_PATH, "wb") as f:
            f.write(r.content)
        print(f"✅ 下載完成 ({len(r.content)} bytes)")
    except Exception as e:
        print(f"⚠️ 下載失敗: {e}，將使用 CDN fallback")

@asynccontextmanager
async def lifespan(app):
    download_lc_if_needed()
    yield

app = FastAPI(title="台股突破訊號掃描器", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── 股票名稱對照表 ──────────────────────────────────────
STOCK_NAMES = {
    '2330':'台積電','2317':'鴻海','2454':'聯發科','2412':'中華電','2308':'台達電',
    '3008':'大立光','2382':'廣達','1303':'南亞','2881':'富邦金','2002':'中鋼',
    '2886':'兆豐金','2882':'國泰金','2884':'玉山金','2303':'聯電','3045':'台灣大',
    '2207':'和泰車','5880':'合庫金','2891':'中信金','2885':'元大金','2890':'永豐金',
    '1301':'台塑','2357':'華碩','1101':'台泥','2327':'國巨','2345':'智邦',
    '3711':'日月光','2379':'瑞昱','4904':'遠傳','2395':'研華','2371':'大同',
    '2408':'南科','2376':'技嘉','2609':'陽明','2615':'萬海','2801':'彰銀',
    '1590':'亞德客','2887':'台新金','2888':'新光金','2889':'國票金','6505':'台塑化',
    '1216':'統一','1326':'台化','2474':'可成','2059':'川湖','3443':'創意',
    '6669':'緯穎','3231':'緯創','8046':'南電','2049':'上銀','6770':'力積電',
    '5269':'祥碩','3034':'聯詠','3533':'嘉澤','2385':'群光','2368':'金像電',
    '4938':'和碩','2301':'光寶科','3481':'群創','5483':'中美晶','3044':'健鼎',
    '6116':'彩晶','3376':'新日興','2347':'聯強','1402':'遠東新','3293':'鈊象',
}

TW50 = [
    '2330','2317','2454','2412','2308','3008','2382','1303','2881','2002',
    '2886','2882','2884','2303','3045','2207','5880','2891','2885','2890',
    '1301','2357','1101','2327','2345','3711','2379','4904','2395','2371',
    '2408','2376','2609','2615','2801','1590','2887','2888','2889','6505',
    '1216','1326','2474','2059','3443','6669','3231','8046','2049','6770',
]
MID100 = [
    '5269','3034','3533','2385','2368','4938','2301','3481','5483','3044',
    '6116','3376','2347','1402','3293','6278','2474','2492','6669','3231',
    '8046','2059','3443','1477','2014','3703','4966','4763','6415','5871',
]

# ── 資料模型 ────────────────────────────────────────────
class ScanRequest(BaseModel):
    token: str
    mode: str = "custom"           # custom | tw50 | mid100 | all
    stocks: Optional[List[str]] = None
    price_breakout_pct: float = 0.97
    vol_ratio_threshold: float = 1.5
    rsi_low: float = 55
    rsi_high: float = 75

class StockResult(BaseModel):
    code: str
    name: str
    price: float
    change_pct: float
    vol_ratio: float
    rsi: float
    ma5: float
    ma10: float
    ma20: float
    macd_hist: float
    price_breakout: bool
    volume_surge: bool
    macd_positive: bool
    macd_golden_cross: bool
    rsi_strong: bool
    ma_bullish: bool
    score: int
    score_pct: float
    strength: str

# ── 技術指標計算 ────────────────────────────────────────
def fetch_finmind(stock_id: str, token: str, days: int = 120):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    resp = requests.get(
        'https://api.finmindtrade.com/api/v4/data',
        params={
            'dataset': 'TaiwanStockPrice',
            'data_id': stock_id,
            'start_date': start_date,
            'end_date': end_date,
            'token': token,
        },
        timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get('data'):
        raise ValueError(f"無資料：{data.get('msg','')}")
    df = pd.DataFrame(data['data']).sort_values('date').reset_index(drop=True)
    for col in ['open','close','max','min']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Trading_Volume'] = pd.to_numeric(df['Trading_Volume'], errors='coerce').fillna(0)
    df = df.dropna(subset=['close'])
    if len(df) < 20:
        raise ValueError('資料筆數不足')
    return df

def calc_rsi(closes: pd.Series, period=14) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.where(loss != 0, other=1e-9)
    return float((100 - 100 / (1 + rs)).iloc[-1])

def calc_macd(closes: pd.Series):
    e12 = closes.ewm(span=12, adjust=False).mean()
    e26 = closes.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    sig  = line.ewm(span=9, adjust=False).mean()
    hist = line - sig
    return float(hist.iloc[-1]), float(hist.iloc[-2]) if len(hist) > 1 else 0.0

def safe_float(v, decimals=2):
    """把 NaN/Inf 換成 0，避免 JSON 序列化失敗"""
    import math
    if v is None: return 0.0
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, decimals)
    except: return 0.0

def analyze(stock_id: str, df: pd.DataFrame, cfg: ScanRequest) -> dict:
    closes  = df['close']
    volumes = df['Trading_Volume']
    highs   = df['max']

    price      = float(closes.iloc[-1])
    prev_price = float(closes.iloc[-2]) if len(closes) > 1 else price
    change_pct = (price - prev_price) / prev_price * 100

    ma5  = float(closes.tail(5).mean())
    ma10 = float(closes.tail(10).mean())
    ma20 = float(closes.tail(20).mean())

    max20      = float(highs.tail(20).max())
    vol_avg20  = float(volumes.tail(20).mean())
    vol_now    = float(volumes.iloc[-1])
    vol_ratio  = vol_now / vol_avg20 if vol_avg20 > 0 else 1.0

    rsi              = calc_rsi(closes)
    hist_now, hist_prev = calc_macd(closes)

    signals = {
        'price_breakout':    price >= max20 * cfg.price_breakout_pct,
        'volume_surge':      vol_ratio >= cfg.vol_ratio_threshold,
        'macd_golden_cross': hist_now > 0 and hist_prev <= 0,
        'macd_positive':     hist_now > 0,
        'rsi_strong':        cfg.rsi_low <= rsi <= cfg.rsi_high,
        'ma_bullish':        ma5 > ma10 > ma20,
    }
    main = ['price_breakout','volume_surge','macd_positive','rsi_strong','ma_bullish']
    score     = sum(signals[k] for k in main)
    score_pct = score / len(main) * 100
    strength  = 'strong' if score >= 4 else ('medium' if score == 3 else 'weak')

    return {
        'code': stock_id,
        'name': STOCK_NAMES.get(stock_id, stock_id),
        'price': safe_float(price, 2),
        'change_pct': safe_float(change_pct, 2),
        'vol_ratio': safe_float(vol_ratio, 2),
        'rsi': safe_float(rsi, 1),
        'ma5': safe_float(ma5, 2),
        'ma10': safe_float(ma10, 2),
        'ma20': safe_float(ma20, 2),
        'macd_hist': safe_float(hist_now, 4),
        'score': score,
        'score_pct': safe_float(score_pct, 1),
        'strength': strength,
        **{k: signals[k] for k in signals},
        '_df': df,
    }

# ── API 路由 ────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")

@app.get("/api/verify")
async def verify_token(token: str):
    try:
        end = datetime.today().strftime('%Y-%m-%d')
        start = (datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')
        resp = requests.get(
            'https://api.finmindtrade.com/api/v4/data',
            params={'dataset':'TaiwanStockPrice','data_id':'2330',
                    'start_date':start,'end_date':end,'token':token},
            timeout=10
        )
        data = resp.json()
        if data.get('data') or data.get('msg') == 'success':
            return {"ok": True}
        return {"ok": False, "msg": data.get('msg','Token 無效')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan")
async def scan(req: ScanRequest):
    if req.mode == 'custom':
        codes = req.stocks or []
    elif req.mode == 'tw50':
        codes = TW50
    elif req.mode == 'mid100':
        codes = MID100
    else:
        codes = list(set(TW50 + MID100))

    if not codes:
        raise HTTPException(status_code=400, detail="請提供股票代號")

    results, errors = [], []
    for code in codes:
        try:
            df  = fetch_finmind(code, req.token)
            r   = analyze(code, df, req)
            results.append({k: v for k, v in r.items() if k != '_df'})
        except Exception as e:
            errors.append({'code': code, 'error': str(e)})

    return {
        'results': sorted(results, key=lambda x: x['score'], reverse=True),
        'errors':  errors,
        'scanned': len(codes),
        'success': len(results),
    }

def clean(lst):
    """把 NaN / Inf / None 換成 None，確保 JSON 序列化不崩潰"""
    import math
    result = []
    for v in lst:
        try:
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                result.append(None)
            else:
                result.append(round(float(v), 4))
        except Exception:
            result.append(None)
    return result

@app.get("/api/chart/{code}")
async def get_chart(code: str, token: str):
    import traceback
    try:
        df = fetch_finmind(code, token, days=120)
        cols = df.columns.tolist()
        print(f"[chart] {code} 欄位: {cols}")

        df = df.tail(60).copy().reset_index(drop=True)

        # 安全取欄位——FinMind 欄位名稱保護
        def get_col(name, fallback=None):
            if name in df.columns:
                return pd.to_numeric(df[name], errors='coerce')
            if fallback and fallback in df.columns:
                return pd.to_numeric(df[fallback], errors='coerce')
            return pd.Series([None] * len(df))

        close_s  = get_col('close')
        open_s   = get_col('open').fillna(close_s)
        high_s   = get_col('max', 'high').fillna(close_s)
        low_s    = get_col('min', 'low').fillna(close_s)
        vol_s    = get_col('Trading_Volume', 'volume').fillna(0)

        # 均線
        ma5  = close_s.rolling(5,  min_periods=1).mean()
        ma10 = close_s.rolling(10, min_periods=1).mean()
        ma20 = close_s.rolling(20, min_periods=1).mean()

        # MACD
        e12  = close_s.ewm(span=12, adjust=False).mean()
        e26  = close_s.ewm(span=26, adjust=False).mean()
        ml   = e12 - e26
        sig  = ml.ewm(span=9, adjust=False).mean()
        macd = (ml - sig)

        # RSI
        delta = close_s.diff()
        gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs    = gain / loss.where(loss != 0, other=1e-9)
        rsi   = 100 - 100 / (1 + rs)

        # 布林通道 (20日, 2倍標準差)
        bb_mid   = close_s.rolling(20, min_periods=1).mean()
        bb_std   = close_s.rolling(20, min_periods=1).std().fillna(0)
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        return {
            'dates':    df['date'].tolist(),
            'open':     clean(open_s.tolist()),
            'high':     clean(high_s.tolist()),
            'low':      clean(low_s.tolist()),
            'close':    clean(close_s.tolist()),
            'vol':      clean(vol_s.tolist()),
            'ma5':      clean(ma5.tolist()),
            'ma10':     clean(ma10.tolist()),
            'ma20':     clean(ma20.tolist()),
            'macd':     clean(macd.tolist()),
            'rsi':      clean(rsi.tolist()),
            'bb_upper': clean(bb_upper.tolist()),
            'bb_mid':   clean(bb_mid.tolist()),
            'bb_lower': clean(bb_lower.tolist()),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

@app.get("/api/intraday/{code}")
async def get_intraday(code: str, token: str):
    """抓取當日盤中分鐘資料（FinMind TaiwanStockPriceTick 或 TaiwanStockPriceMinute）"""
    import traceback
    try:
        today = datetime.today().strftime('%Y-%m-%d')
        # 先試分鐘線
        resp = requests.get(
            'https://api.finmindtrade.com/api/v4/data',
            params={
                'dataset': 'TaiwanStockPriceMinute',
                'data_id': code,
                'start_date': today,
                'end_date': today,
                'token': token,
            },
            timeout=15
        )
        data = resp.json()
        rows = data.get('data', [])

        # 非交易日或盤前：退回前一個交易日
        if not rows:
            yesterday = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
            resp2 = requests.get(
                'https://api.finmindtrade.com/api/v4/data',
                params={
                    'dataset': 'TaiwanStockPriceMinute',
                    'data_id': code,
                    'start_date': yesterday,
                    'end_date': yesterday,
                    'token': token,
                },
                timeout=15
            )
            data2 = resp2.json()
            rows = data2.get('data', [])
            date_label = yesterday
        else:
            date_label = today

        if not rows:
            return {'date': date_label, 'times': [], 'open': [], 'high': [],
                    'low': [], 'close': [], 'vol': [], 'vwap': [], 'msg': '無盤中資料'}

        df = pd.DataFrame(rows)
        for col in ['open','close','max','min','volume','Trading_Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        close_col = 'close'
        vol_col   = 'volume' if 'volume' in df.columns else 'Trading_Volume'
        high_col  = 'max'    if 'max'    in df.columns else 'high'
        low_col   = 'min'    if 'min'    in df.columns else 'low'
        open_col  = 'open'

        # 時間欄位處理
        time_col = 'datetime' if 'datetime' in df.columns else 'date'
        df = df.sort_values(time_col).reset_index(drop=True)

        close_s = df[close_col].fillna(method='ffill').fillna(0)
        vol_s   = df[vol_col].fillna(0) if vol_col in df.columns else pd.Series([0]*len(df))

        # VWAP（成交量加權平均價）
        cum_vol = vol_s.cumsum()
        cum_pv  = (close_s * vol_s).cumsum()
        vwap    = (cum_pv / cum_vol.where(cum_vol > 0, other=1)).round(2)

        # 時間標籤
        if 'datetime' in df.columns:
            times = df['datetime'].astype(str).str[-8:].str[:5].tolist()  # HH:MM
        else:
            times = df['date'].astype(str).tolist()

        return {
            'date':   date_label,
            'times':  times,
            'open':   clean(df[open_col].tolist() if open_col in df.columns else close_s.tolist()),
            'high':   clean(df[high_col].tolist() if high_col in df.columns else close_s.tolist()),
            'low':    clean(df[low_col].tolist()  if low_col  in df.columns else close_s.tolist()),
            'close':  clean(close_s.tolist()),
            'vol':    clean(vol_s.tolist()),
            'vwap':   clean(vwap.tolist()),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

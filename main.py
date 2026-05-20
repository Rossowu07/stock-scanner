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

def fetch_institutional(stock_id: str, token: str, days: int = 30) -> pd.DataFrame:
    """三大法人買賣超"""
    end_date   = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    resp = requests.get(
        'https://api.finmindtrade.com/api/v4/data',
        params={
            'dataset':    'TaiwanStockInstitutionalInvestors',
            'data_id':    stock_id,
            'start_date': start_date,
            'end_date':   end_date,
            'token':      token,
        },
        timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get('data'):
        return pd.DataFrame()
    df = pd.DataFrame(data['data'])
    df['buy']  = pd.to_numeric(df.get('buy',  pd.Series(dtype=float)), errors='coerce').fillna(0)
    df['sell'] = pd.to_numeric(df.get('sell', pd.Series(dtype=float)), errors='coerce').fillna(0)
    df['net']  = df['buy'] - df['sell']
    return df.sort_values('date').reset_index(drop=True)

def fetch_margin(stock_id: str, token: str, days: int = 30) -> pd.DataFrame:
    """融資融券餘額"""
    end_date   = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    resp = requests.get(
        'https://api.finmindtrade.com/api/v4/data',
        params={
            'dataset':    'TaiwanStockMarginPurchaseShortsale',
            'data_id':    stock_id,
            'start_date': start_date,
            'end_date':   end_date,
            'token':      token,
        },
        timeout=15
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get('data'):
        return pd.DataFrame()
    df = pd.DataFrame(data['data'])
    for col in ['MarginPurchaseBuy','MarginPurchaseSell','MarginPurchaseRedeem',
                'MarginPurchaseTodayBalance','ShortSaleBuy','ShortSaleSell',
                'ShortSaleTodayBalance']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df.sort_values('date').reset_index(drop=True)

def calc_chip(inst_df: pd.DataFrame, margin_df: pd.DataFrame) -> dict:
    """計算籌碼面指標，回傳結構化結果"""
    result = {
        # 三大法人
        'foreign_5d': 0, 'foreign_20d': 0, 'foreign_today': 0,
        'trust_5d':   0, 'trust_20d':   0, 'trust_today':   0,
        'dealer_5d':  0, 'dealer_5d':   0, 'dealer_today':  0,
        'inst_total_5d': 0, 'inst_total_20d': 0,
        # 融資融券
        'margin_balance': 0, 'margin_change_5d': 0, 'margin_ratio': 0.0,
        'short_balance':  0, 'short_change_5d':  0,
        'margin_healthy': False,
        # 籌碼訊號
        'foreign_buying':  False,
        'trust_buying':    False,
        'margin_safe':     False,
        'chip_score':      0,
        'chip_available':  False,
    }

    # ── 三大法人 ────────────────────────────────────────
    if not inst_df.empty and 'name' in inst_df.columns:
        result['chip_available'] = True
        for name_key, prefix in [
            ('外資', 'foreign'), ('外資自營商', 'foreign'),
            ('投信', 'trust'), ('自營商', 'dealer'),
        ]:
            sub = inst_df[inst_df['name'] == name_key].copy()
            if sub.empty:
                continue
            net = sub['net']
            if prefix == 'foreign':
                result['foreign_today']  = safe_float(net.iloc[-1], 0)
                result['foreign_5d']     = safe_float(net.tail(5).sum(), 0)
                result['foreign_20d']    = safe_float(net.tail(20).sum(), 0)
            elif prefix == 'trust':
                result['trust_today']    = safe_float(net.iloc[-1], 0)
                result['trust_5d']       = safe_float(net.tail(5).sum(), 0)
                result['trust_20d']      = safe_float(net.tail(20).sum(), 0)
            elif prefix == 'dealer':
                result['dealer_today']   = safe_float(net.iloc[-1], 0)
                result['dealer_5d']      = safe_float(net.tail(5).sum(), 0)

        # 三大法人合計
        dates = inst_df['date'].unique()
        totals = []
        for d in sorted(dates):
            day_net = inst_df[inst_df['date'] == d]['net'].sum()
            totals.append(day_net)
        totals_s = pd.Series(totals)
        result['inst_total_5d']  = safe_float(totals_s.tail(5).sum(),  0)
        result['inst_total_20d'] = safe_float(totals_s.tail(20).sum(), 0)

        result['foreign_buying'] = result['foreign_5d'] > 0
        result['trust_buying']   = result['trust_5d']   > 0

    # ── 融資融券 ────────────────────────────────────────
    if not margin_df.empty:
        result['chip_available'] = True
        if 'MarginPurchaseTodayBalance' in margin_df.columns:
            mb = margin_df['MarginPurchaseTodayBalance']
            result['margin_balance']   = safe_float(mb.iloc[-1], 0)
            result['margin_change_5d'] = safe_float(mb.iloc[-1] - mb.iloc[-6] if len(mb) >= 6 else 0, 0)
        if 'ShortSaleTodayBalance' in margin_df.columns:
            sb = margin_df['ShortSaleTodayBalance']
            result['short_balance']   = safe_float(sb.iloc[-1], 0)
            result['short_change_5d'] = safe_float(sb.iloc[-1] - sb.iloc[-6] if len(sb) >= 6 else 0, 0)
        # 融資健康：餘額減少（散戶出場）或融券增加（空頭回補動能）
        result['margin_healthy'] = result['margin_change_5d'] <= 0
        result['margin_safe']    = result['margin_healthy']

    # ── 籌碼評分（3分滿分）────────────────────────────
    chip_score = 0
    if result['foreign_buying']:  chip_score += 1
    if result['trust_buying']:    chip_score += 1
    if result['margin_safe']:     chip_score += 1
    result['chip_score'] = chip_score
    return result

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

    # 布林通道分析
    bb_mid   = closes.rolling(20, min_periods=1).mean()
    bb_std   = closes.rolling(20, min_periods=1).std().fillna(0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = ((bb_upper - bb_lower) / bb_mid * 100)  # 通道寬度 %

    bb_mid_now   = float(bb_mid.iloc[-1])
    bb_upper_now = float(bb_upper.iloc[-1])
    bb_lower_now = float(bb_lower.iloc[-1])
    bb_width_now = float(bb_width.iloc[-1])
    bb_width_avg = float(bb_width.tail(20).mean()) if len(bb_width) >= 5 else bb_width_now

    # 價格在通道中的位置 0~100%（0=下軌, 100=上軌）
    bb_range     = bb_upper_now - bb_lower_now
    bb_position  = ((price - bb_lower_now) / bb_range * 100) if bb_range > 0 else 50
    bb_expanding = bb_width_now > bb_width_avg  # 通道是否擴張

    signals = {
        'price_breakout':    price >= max20 * cfg.price_breakout_pct,
        'volume_surge':      vol_ratio >= cfg.vol_ratio_threshold,
        'macd_golden_cross': hist_now > 0 and hist_prev <= 0,
        'macd_positive':     hist_now > 0,
        'rsi_strong':        cfg.rsi_low <= rsi <= cfg.rsi_high,
        'ma_bullish':        ma5 > ma10 > ma20,
    }
    main = ['price_breakout','volume_surge','macd_positive','rsi_strong','ma_bullish']
    tech_score = sum(signals[k] for k in main)
    score_pct  = tech_score / len(main) * 100
    strength   = 'strong' if tech_score >= 4 else ('medium' if tech_score == 3 else 'weak')

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
        'score':      tech_score,
        'score_pct':  safe_float(score_pct, 1),
        'strength':   strength,
        'bb_upper':   safe_float(bb_upper_now, 2),
        'bb_mid':     safe_float(bb_mid_now, 2),
        'bb_lower':   safe_float(bb_lower_now, 2),
        'bb_position':safe_float(bb_position, 1),
        'bb_width':   safe_float(bb_width_now, 1),
        'bb_expanding': bb_expanding,
        **{k: signals[k] for k in signals},
        '_df': df,
        # 籌碼面預留欄位（掃描後由 scan endpoint 填入）
        'chip_score': 0, 'chip_available': False,
        'foreign_5d': 0, 'foreign_20d': 0, 'foreign_today': 0,
        'trust_5d': 0, 'trust_20d': 0, 'trust_today': 0,
        'dealer_5d': 0, 'dealer_today': 0,
        'inst_total_5d': 0, 'inst_total_20d': 0,
        'margin_balance': 0, 'margin_change_5d': 0,
        'short_balance': 0,  'short_change_5d': 0,
        'foreign_buying': False, 'trust_buying': False, 'margin_safe': False,
        'total_score': tech_score, 'total_score_pct': safe_float(score_pct, 1),
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
            df = fetch_finmind(code, req.token)
            r  = analyze(code, df, req)

            # 抓籌碼資料（獨立 try，失敗不影響技術面結果）
            try:
                inst_df   = fetch_institutional(code, req.token, days=35)
                margin_df = fetch_margin(code, req.token, days=35)
                chip      = calc_chip(inst_df, margin_df)
                r.update(chip)
            except Exception:
                pass  # 籌碼資料失敗靜默處理

            # 合併總分（技術5 + 籌碼3 = 滿分8）
            chip_score = r.get('chip_score', 0)
            tech_score = r.get('score', 0)
            total      = tech_score + chip_score
            total_pct  = round(total / 8 * 100, 1)
            # 重新判斷強度（以8分為基準）
            if total >= 6:   strength = 'strong'
            elif total >= 4: strength = 'medium'
            else:            strength = 'weak'
            r['total_score']     = total
            r['total_score_pct'] = total_pct
            r['strength']        = strength

            results.append({k: v for k, v in r.items() if k != '_df'})
        except Exception as e:
            errors.append({'code': code, 'error': str(e)})

    return {
        'results': sorted(results, key=lambda x: x['total_score'], reverse=True),
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

class StrategyRequest(BaseModel):
    stock: dict  # 前端傳來的完整股票資料

@app.post("/api/strategy")
async def get_strategy(req: StrategyRequest):
    """根據技術面 + 籌碼面資料，產生短中長線操作建議"""
    import traceback
    try:
        r = req.stock
        code  = r.get('code','')
        name  = r.get('name','')
        price = r.get('price', 0)

        # 收集各面向分數
        tech_score  = r.get('score', 0)           # 技術面 /5
        chip_score  = r.get('chip_score', 0)       # 籌碼面 /3
        total_score = r.get('total_score', 0)      # 總分 /8
        chip_avail  = r.get('chip_available', False)

        # 技術指標
        ma_bullish      = r.get('ma_bullish', False)
        macd_positive   = r.get('macd_positive', False)
        macd_golden     = r.get('macd_golden_cross', False)
        rsi             = r.get('rsi', 50)
        rsi_strong      = r.get('rsi_strong', False)
        price_breakout  = r.get('price_breakout', False)
        volume_surge    = r.get('volume_surge', False)
        bb_position     = r.get('bb_position', 50)
        bb_expanding    = r.get('bb_expanding', False)
        bb_upper        = r.get('bb_upper', 0)
        bb_lower        = r.get('bb_lower', 0)
        bb_mid          = r.get('bb_mid', 0)

        # 籌碼指標
        foreign_buying  = r.get('foreign_buying', False)
        trust_buying    = r.get('trust_buying', False)
        margin_safe     = r.get('margin_safe', False)
        foreign_5d      = r.get('foreign_5d', 0)
        foreign_20d     = r.get('foreign_20d', 0)
        trust_5d        = r.get('trust_5d', 0)
        margin_change   = r.get('margin_change_5d', 0)

        # ── 短線建議（1–5 日）────────────────────────
        short_signals, short_risks = [], []
        if macd_golden:     short_signals.append('MACD 剛發生金叉，短線動能轉強')
        if volume_surge:    short_signals.append(f'成交量放大至均量 {r.get("vol_ratio",1):.1f} 倍，有資金進場')
        if price_breakout:  short_signals.append('突破近20日高點，短線有突破動能')
        if rsi_strong:      short_signals.append(f'RSI {rsi:.0f} 在強勢區間，多頭動能足夠')
        if bb_position > 70 and bb_expanding:
            short_signals.append('布林通道擴張且偏上軌，短線趨勢向上')

        if rsi > 75:        short_risks.append(f'RSI {rsi:.0f} 已過熱，短線回調風險')
        if bb_position > 90: short_risks.append(f'股價貼近布林上軌 {bb_upper:.2f}，短線壓力大')
        if not volume_surge and price_breakout:
            short_risks.append('突破時量能不足，需確認是否為假突破')

        if len(short_signals) >= 3:
            short_action = '積極'
            short_note   = f'多項短線訊號同時出現，可考慮短線進場，留意 {bb_upper:.2f} 壓力'
        elif len(short_signals) >= 2:
            short_action = '中性偏多'
            short_note   = '短線訊號尚可，建議小量試單，嚴設停損'
        elif short_risks:
            short_action = '謹慎'
            short_note   = '短線風險訊號較多，建議觀望或減碼'
        else:
            short_action = '觀望'
            short_note   = '短線訊號不明確，等待更好進場時機'

        # ── 中線建議（1–3 個月）──────────────────────
        mid_signals, mid_risks = [], []
        if ma_bullish:       mid_signals.append('均線多頭排列（MA5>MA10>MA20），中線趨勢向上')
        if macd_positive:    mid_signals.append('MACD 柱狀維持正值，中線多頭動能持續')
        if foreign_buying and chip_avail:
            mid_signals.append(f'外資近5日累積買超 {foreign_5d:,.0f} 張，法人持續佈局')
        if trust_buying and chip_avail:
            mid_signals.append(f'投信近5日買超 {trust_5d:,.0f} 張，機構資金進場')
        if margin_safe and chip_avail:
            mid_signals.append('融資餘額下降，籌碼趨於健康，主力控盤跡象')
        if bb_expanding and bb_position > 50:
            mid_signals.append('布林通道擴張且股價在中軌以上，中線趨勢確立')

        if foreign_20d < 0 and chip_avail:
            mid_risks.append(f'外資近20日累積賣超 {abs(foreign_20d):,.0f} 張，中線偏空')
        if not ma_bullish:
            mid_risks.append('均線尚未完整多頭排列，中線趨勢未確立')
        if margin_change > 0 and chip_avail:
            mid_risks.append('融資餘額上升，散戶追高風險，籌碼不穩')

        if len(mid_signals) >= 4:
            mid_action = '積極佈局'
            mid_note   = f'技術面與籌碼面雙重確認，中線趨勢明確向上，可分批建立部位，支撐參考 MA20 {bb_mid:.2f}'
        elif len(mid_signals) >= 2:
            mid_action = '逢低布局'
            mid_note   = '中線訊號偏正向，建議分批進場，以 MA20 或布林中軌為支撐判斷'
        elif mid_risks:
            mid_action = '減碼觀察'
            mid_note   = '中線風險因素較多，建議降低持倉比重，等待籌碼面改善'
        else:
            mid_action = '中性觀察'
            mid_note   = '中線方向尚未明確，持續追蹤法人動向'

        # ── 長線建議（3個月以上）──────────────────────
        long_signals, long_risks = [], []
        if foreign_20d > 0 and chip_avail:
            long_signals.append(f'外資近20日持續買超 {foreign_20d:,.0f} 張，長線法人看多')
        if ma_bullish and macd_positive:
            long_signals.append('技術面趨勢結構完整，均線多頭 + MACD 正值')
        if margin_safe and chip_avail:
            long_signals.append('籌碼面健康，融資不追高，有助長線走揚')
        if total_score >= 6:
            long_signals.append(f'綜合評分 {total_score}/8，技術面與籌碼面均強')

        if not chip_avail:
            long_risks.append('籌碼資料未能取得，長線判斷依賴技術面為主')
        if rsi > 75:
            long_risks.append('目前 RSI 偏高，長線進場位置不理想，等待回調')
        if bb_position > 85:
            long_risks.append(f'股價偏高，距布林下軌支撐 {bb_lower:.2f} 較遠，進場風險高')

        if len(long_signals) >= 3 and len(long_risks) == 0:
            long_action = '長線買進'
            long_note   = f'多項長線正向訊號，適合分批建立長期部位，支撐參考 {bb_lower:.2f}（布林下軌）'
        elif len(long_signals) >= 2:
            long_action = '分批建倉'
            long_note   = '長線條件尚可，建議小量分批建倉，持續追蹤外資與法人動向'
        elif long_risks:
            long_action = '等待時機'
            long_note   = '目前進場位置風險偏高，建議等待回調至較低位置再考慮長線佈局'
        else:
            long_action = '持續觀察'
            long_note   = '長線訊號尚不明確，持續追蹤基本面與籌碼面變化'

        # ── 整體風險提示 ──────────────────────────────
        risk_warnings = []
        if not chip_avail:
            risk_warnings.append('⚠️ 籌碼面資料取得失敗，以上建議僅基於技術面分析')
        if rsi > 80:
            risk_warnings.append(f'⚠️ RSI {rsi:.0f} 極度過熱，任何操作需謹慎')
        if bb_position > 95:
            risk_warnings.append('⚠️ 股價已突破布林上軌，極端強勢但回調風險極高')
        risk_warnings.append('⚠️ 以上為技術面與籌碼面分析，不構成投資建議，操作請自行判斷並設定停損')

        return {
            'code': code, 'name': name, 'price': price,
            'total_score': total_score, 'tech_score': tech_score, 'chip_score': chip_score,
            'short': {
                'action':  short_action,
                'note':    short_note,
                'signals': short_signals,
                'risks':   short_risks,
            },
            'mid': {
                'action':  mid_action,
                'note':    mid_note,
                'signals': mid_signals,
                'risks':   mid_risks,
            },
            'long': {
                'action':  long_action,
                'note':    long_note,
                'signals': long_signals,
                'risks':   long_risks,
            },
            'risk_warnings': risk_warnings,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


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
    """
    盤中/當日走勢：使用 Yahoo Finance 1分鐘線（免費，近5日可用）
    收盤後即可取得當日完整分鐘資料。
    """
    import traceback
    try:
        ticker = f"{code}.TW"
        # Yahoo Finance：interval=1m 最多取近7天，period=5d 取近5個交易日
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?interval=1m&range=1d&includePrePost=false"
        )
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        result = data.get("chart", {}).get("result", [])
        if not result:
            raise ValueError("Yahoo Finance 無資料回應")

        r      = result[0]
        meta   = r.get("meta", {})
        ts     = r.get("timestamp", [])
        quotes = r.get("indicators", {}).get("quote", [{}])[0]

        if not ts:
            raise ValueError("今日尚無分鐘資料（可能尚未開盤或非交易日）")

        from datetime import timezone, timedelta as td
        tz_offset = td(hours=8)  # 台灣 UTC+8

        times  = []
        opens, highs, lows, closes, vols = [], [], [], [], []

        raw_close = quotes.get("close", [])
        raw_open  = quotes.get("open",  [])
        raw_high  = quotes.get("high",  [])
        raw_low   = quotes.get("low",   [])
        raw_vol   = quotes.get("volume",[])

        for i, t in enumerate(ts):
            c = raw_close[i] if i < len(raw_close) else None
            if c is None:
                continue
            dt_local = datetime.fromtimestamp(t, tz=timezone.utc).astimezone(
                timezone(tz_offset)
            )
            times.append(dt_local.strftime("%H:%M"))
            opens.append(round(float(raw_open[i]),  2) if i < len(raw_open)  and raw_open[i]  else round(float(c), 2))
            highs.append(round(float(raw_high[i]),  2) if i < len(raw_high)  and raw_high[i]  else round(float(c), 2))
            lows.append( round(float(raw_low[i]),   2) if i < len(raw_low)   and raw_low[i]   else round(float(c), 2))
            closes.append(round(float(c), 2))
            vols.append(  int(raw_vol[i])               if i < len(raw_vol)   and raw_vol[i]   else 0)

        if not closes:
            raise ValueError("資料過濾後為空")

        # VWAP
        cum_pv  = 0.0
        cum_vol = 0
        vwap    = []
        for c, v in zip(closes, vols):
            cum_pv  += c * v
            cum_vol += v
            vwap.append(round(cum_pv / cum_vol, 2) if cum_vol > 0 else c)

        # 取得日期標籤
        trade_date = datetime.fromtimestamp(
            ts[0], tz=timezone.utc
        ).astimezone(timezone(tz_offset)).strftime("%Y-%m-%d")

        regular_hours = meta.get("regularMarketTime")
        market_state  = meta.get("marketState", "")
        status_note   = "（盤中即時）" if market_state == "REGULAR" else "（收盤後）"

        return {
            "date":   trade_date,
            "status": status_note,
            "times":  times,
            "open":   opens,
            "high":   highs,
            "low":    lows,
            "close":  closes,
            "vol":    vols,
            "vwap":   vwap,
            "msg":    "",
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")



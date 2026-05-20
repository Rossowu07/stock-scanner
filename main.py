from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import requests, pandas as pd, traceback, math, os
from datetime import datetime, timedelta
from typing import List, Optional

# ── LightweightCharts 本地快取 ──────────────────────────
LC_JS_PATH = "static/lc.js"
LC_CDN_URL = "https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"

def download_lc_if_needed():
    if os.path.exists(LC_JS_PATH) and os.path.getsize(LC_JS_PATH) > 50000:
        print("✅ LightweightCharts 已存在"); return
    try:
        r = requests.get(LC_CDN_URL, timeout=30); r.raise_for_status()
        os.makedirs("static", exist_ok=True)
        open(LC_JS_PATH, "wb").write(r.content)
        print(f"✅ LightweightCharts 下載完成 ({len(r.content)} bytes)")
    except Exception as e:
        print(f"⚠️ 下載失敗: {e}")

@asynccontextmanager
async def lifespan(app):
    download_lc_if_needed()
    yield

app = FastAPI(title="台股突破訊號掃描器", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── 常數 ────────────────────────────────────────────────
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
TW50 = ['2330','2317','2454','2412','2308','3008','2382','1303','2881','2002',
        '2886','2882','2884','2303','3045','2207','5880','2891','2885','2890',
        '1301','2357','1101','2327','2345','3711','2379','4904','2395','2371',
        '2408','2376','2609','2615','2801','1590','2887','2888','2889','6505',
        '1216','1326','2474','2059','3443','6669','3231','8046','2049','6770']
MID100 = ['5269','3034','3533','2385','2368','4938','2301','3481','5483','3044',
          '6116','3376','2347','1402','3293','6278','2474','2492','6669','3231',
          '8046','2059','3443','1477','2014','3703','4966','4763','6415','5871']

# ── 工具函數 ─────────────────────────────────────────────
def safe_float(v, decimals=2):
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f): return 0.0
        return round(f, decimals)
    except: return 0.0

def clean(lst):
    result = []
    for v in lst:
        try:
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                result.append(None)
            else:
                result.append(round(float(v), 4))
        except: result.append(None)
    return result

# ── 資料模型 ─────────────────────────────────────────────
class ScanRequest(BaseModel):
    token: str
    mode: str = "custom"
    stocks: Optional[List[str]] = None
    price_breakout_pct: float = 0.97
    vol_ratio_threshold: float = 1.5
    rsi_low: float = 55
    rsi_high: float = 75

class StrategyRequest(BaseModel):
    stock: dict

# ── FinMind 資料抓取 ──────────────────────────────────────
def finmind_get(dataset, stock_id, start, end, token):
    """用 urllib GET 呼叫 FinMind，手動組 URL 避免 requests 編碼問題"""
    import json as _json, urllib.request, urllib.parse, urllib.error
    # 只對 token 做 encoding（含特殊字元），其他參數直接拼接
    safe_token = urllib.parse.quote(token, safe='')
    url = (
        f"https://api.finmindtrade.com/api/v4/data"
        f"?dataset={dataset}"
        f"&data_id={stock_id}"
        f"&start_date={start}"
        f"&end_date={end}"
        f"&token={safe_token}"
    )
    print(f"[finmind] GET dataset={dataset} data_id={stock_id} start={start} end={end} token_len={len(token)}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"[finmind] ERROR {e.code}: {body[:300]}")
        raise Exception(f"FinMind HTTP {e.code}: {body[:200]}")


def fetch_finmind(stock_id, token, days=120):
    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    data  = finmind_get('TaiwanStockPrice', stock_id, start, end, token)
    if not data.get('data'): raise ValueError(f"無資料：{data.get('msg','')}")
    df = pd.DataFrame(data['data']).sort_values('date').reset_index(drop=True)
    for col in ['open','close','max','min']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Trading_Volume'] = pd.to_numeric(df['Trading_Volume'], errors='coerce').fillna(0)
    df = df.dropna(subset=['close'])
    if len(df) < 20: raise ValueError('資料筆數不足')
    return df

def fetch_institutional(stock_id, token, days=35):
    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    data  = finmind_get('TaiwanStockInstitutionalInvestorsBuySell', stock_id, start, end, token)
    print(f"[inst] {stock_id} msg={data.get('msg')} count={len(data.get('data',[]))}")
    if not data.get('data'): return pd.DataFrame()
    df = pd.DataFrame(data['data'])
    print(f"[inst] {stock_id} cols={df.columns.tolist()}")
    df.columns = [c.lower() for c in df.columns]
    if 'name' in df.columns:
        print(f"[inst] {stock_id} names={df['name'].unique().tolist()}")
    for col in ['buy','sell']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0
    df['net'] = df['buy'] - df['sell']
    return df.sort_values('date').reset_index(drop=True)

def fetch_margin(stock_id, token, days=35):
    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    data  = finmind_get('TaiwanStockMarginPurchaseShortSale', stock_id, start, end, token)
    print(f"[margin] {stock_id} msg={data.get('msg')} count={len(data.get('data',[]))}")
    if not data.get('data'): return pd.DataFrame()
    df = pd.DataFrame(data['data'])
    print(f"[margin] {stock_id} cols={df.columns.tolist()}")
    for col in df.columns:
        if col not in ['date','stock_id']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df.sort_values('date').reset_index(drop=True)

# ── 技術指標計算 ──────────────────────────────────────────
def calc_rsi(closes, period=14):
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs    = gain / loss.where(loss != 0, other=1e-9)
    return float((100 - 100 / (1 + rs)).iloc[-1])

def calc_macd(closes):
    e12  = closes.ewm(span=12, adjust=False).mean()
    e26  = closes.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    sig  = line.ewm(span=9, adjust=False).mean()
    hist = line - sig
    return float(hist.iloc[-1]), float(hist.iloc[-2]) if len(hist) > 1 else 0.0

def calc_chip(inst_df, margin_df):
    r = {
        'foreign_5d':0,'foreign_20d':0,'foreign_today':0,
        'trust_5d':0,'trust_20d':0,'trust_today':0,
        'dealer_5d':0,'dealer_today':0,
        'inst_total_5d':0,'inst_total_20d':0,
        'margin_balance':0,'margin_change_5d':0,
        'short_balance':0,'short_change_5d':0,
        'margin_healthy':False,'foreign_buying':False,
        'trust_buying':False,'margin_safe':False,
        'chip_score':0,'chip_available':False,
    }

    # ── 三大法人 ──────────────────────────────────────────
    if not inst_df.empty and 'name' in inst_df.columns:
        r['chip_available'] = True

        def classify(name):
            """用關鍵字判斷法人類型，不依賴精確名稱"""
            n = str(name).lower()
            # 自營商避險/自行買賣都算自營商，但要先判斷避免外資自營商誤判
            if '外資' in n or 'foreign' in n:
                return 'foreign'
            if '投信' in n or 'trust' in n or 'investment' in n:
                return 'trust'
            if '自營' in n or 'dealer' in n:
                return 'dealer'
            return None

        for name_val, grp in inst_df.groupby('name'):
            kind = classify(name_val)
            net  = grp['net']
            print(f"[chip] name='{name_val}' -> kind={kind} rows={len(grp)}")
            if kind == 'foreign':
                r['foreign_today'] = safe_float(net.iloc[-1], 0)
                r['foreign_5d']   += safe_float(net.tail(5).sum(), 0)
                r['foreign_20d']  += safe_float(net.tail(20).sum(), 0)
            elif kind == 'trust':
                r['trust_today']   = safe_float(net.iloc[-1], 0)
                r['trust_5d']     += safe_float(net.tail(5).sum(), 0)
                r['trust_20d']    += safe_float(net.tail(20).sum(), 0)
            elif kind == 'dealer':
                r['dealer_today']  = safe_float(net.iloc[-1], 0)
                r['dealer_5d']    += safe_float(net.tail(5).sum(), 0)

        # 三大合計：以日期彙總
        daily = inst_df.groupby('date')['net'].sum()
        r['inst_total_5d']  = safe_float(daily.tail(5).sum(),  0)
        r['inst_total_20d'] = safe_float(daily.tail(20).sum(), 0)
        r['foreign_buying'] = r['foreign_5d'] > 0
        r['trust_buying']   = r['trust_5d']   > 0

    # ── 融資融券 ──────────────────────────────────────────
    if not margin_df.empty:
        r['chip_available'] = True
        # 建立小寫對應表
        col_map = {c.lower(): c for c in margin_df.columns}
        print(f"[chip] margin cols={list(col_map.keys())}")

        # 融資餘額（官方欄位：MarginPurchaseTodayBalance）
        for try_col in ['marginpurchasetodaybalance', 'marginpurchase_today_balance',
                        'margin_purchase_today_balance']:
            orig = col_map.get(try_col)
            if orig:
                mb = margin_df[orig]
                r['margin_balance']   = safe_float(mb.iloc[-1], 0)
                r['margin_change_5d'] = safe_float(
                    mb.iloc[-1] - mb.iloc[-6] if len(mb) >= 6 else 0, 0)
                print(f"[chip] margin_balance={r['margin_balance']} change5d={r['margin_change_5d']}")
                break

        # 融券餘額（官方欄位：ShortSaleTodayBalance）
        for try_col in ['shortsaletodaybalance', 'short_sale_today_balance',
                        'shortsale_today_balance']:
            orig = col_map.get(try_col)
            if orig:
                sb = margin_df[orig]
                r['short_balance']   = safe_float(sb.iloc[-1], 0)
                r['short_change_5d'] = safe_float(
                    sb.iloc[-1] - sb.iloc[-6] if len(sb) >= 6 else 0, 0)
                break

        r['margin_healthy'] = r['margin_change_5d'] <= 0
        r['margin_safe']    = r['margin_healthy']

    r['chip_score'] = sum([r['foreign_buying'], r['trust_buying'], r['margin_safe']])
    return r

def analyze(stock_id, df, cfg):
    closes  = df['close']
    volumes = df['Trading_Volume']
    highs   = df['max']

    price      = float(closes.iloc[-1])
    prev_price = float(closes.iloc[-2]) if len(closes) > 1 else price
    change_pct = (price - prev_price) / prev_price * 100

    ma5  = float(closes.tail(5).mean())
    ma10 = float(closes.tail(10).mean())
    ma20 = float(closes.tail(20).mean())

    max20     = float(highs.tail(20).max())
    vol_avg20 = float(volumes.tail(20).mean())
    vol_now   = float(volumes.iloc[-1])
    vol_ratio = vol_now / vol_avg20 if vol_avg20 > 0 else 1.0

    rsi              = calc_rsi(closes)
    hist_now, hist_prev = calc_macd(closes)

    # 布林通道
    bb_mid_s   = closes.rolling(20, min_periods=1).mean()
    bb_std_s   = closes.rolling(20, min_periods=1).std().fillna(0)
    bb_upper_s = bb_mid_s + 2 * bb_std_s
    bb_lower_s = bb_mid_s - 2 * bb_std_s
    bb_width_s = ((bb_upper_s - bb_lower_s) / bb_mid_s * 100)

    bb_mid_now   = float(bb_mid_s.iloc[-1])
    bb_upper_now = float(bb_upper_s.iloc[-1])
    bb_lower_now = float(bb_lower_s.iloc[-1])
    bb_width_now = float(bb_width_s.iloc[-1])
    bb_width_avg = float(bb_width_s.tail(20).mean()) if len(bb_width_s) >= 5 else bb_width_now
    bb_range     = bb_upper_now - bb_lower_now
    bb_position  = ((price - bb_lower_now) / bb_range * 100) if bb_range > 0 else 50
    bb_expanding = bb_width_now > bb_width_avg

    signals = {
        'price_breakout':    price >= max20 * cfg.price_breakout_pct,
        'volume_surge':      vol_ratio >= cfg.vol_ratio_threshold,
        'macd_golden_cross': hist_now > 0 and hist_prev <= 0,
        'macd_positive':     hist_now > 0,
        'rsi_strong':        cfg.rsi_low <= rsi <= cfg.rsi_high,
        'ma_bullish':        ma5 > ma10 > ma20,
    }
    main       = ['price_breakout','volume_surge','macd_positive','rsi_strong','ma_bullish']
    tech_score = sum(signals[k] for k in main)
    score_pct  = tech_score / len(main) * 100
    strength   = 'strong' if tech_score >= 4 else ('medium' if tech_score == 3 else 'weak')

    return {
        'code': stock_id, 'name': STOCK_NAMES.get(stock_id, stock_id),
        'price': safe_float(price, 2), 'change_pct': safe_float(change_pct, 2),
        'vol_ratio': safe_float(vol_ratio, 2), 'rsi': safe_float(rsi, 1),
        'ma5': safe_float(ma5, 2), 'ma10': safe_float(ma10, 2), 'ma20': safe_float(ma20, 2),
        'macd_hist': safe_float(hist_now, 4),
        'score': tech_score, 'score_pct': safe_float(score_pct, 1), 'strength': strength,
        'bb_upper': safe_float(bb_upper_now, 2), 'bb_mid': safe_float(bb_mid_now, 2),
        'bb_lower': safe_float(bb_lower_now, 2), 'bb_position': safe_float(bb_position, 1),
        'bb_width': safe_float(bb_width_now, 1), 'bb_expanding': bb_expanding,
        **{k: signals[k] for k in signals},
        # 籌碼預留欄位
        'chip_score':0,'chip_available':False,
        'foreign_5d':0,'foreign_20d':0,'foreign_today':0,
        'trust_5d':0,'trust_20d':0,'trust_today':0,
        'dealer_5d':0,'dealer_today':0,
        'inst_total_5d':0,'inst_total_20d':0,
        'margin_balance':0,'margin_change_5d':0,
        'short_balance':0,'short_change_5d':0,
        'foreign_buying':False,'trust_buying':False,'margin_safe':False,
        'total_score': tech_score, 'total_score_pct': safe_float(score_pct, 1),
        '_df': df,
    }

# ── API 路由 ──────────────────────────────────────────────
@app.get("/")
async def root(): return FileResponse("static/index.html")

@app.get("/api/verify")
async def verify_token(token: str):
    try:
        print(f"[verify] token_len={len(token)} token_preview={token[:20]}...{token[-10:]}")
        end   = datetime.today().strftime('%Y-%m-%d')
        start = (datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')
        data  = finmind_get('TaiwanStockPrice', '2330', start, end, token)
        if data.get('data') or data.get('msg') == 'success':
            return {"ok": True}
        return {"ok": False, "msg": data.get('msg','Token 無效')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan")
async def scan(req: ScanRequest):
    if req.mode == 'custom':   codes = req.stocks or []
    elif req.mode == 'tw50':   codes = TW50
    elif req.mode == 'mid100': codes = MID100
    else:                      codes = list(set(TW50 + MID100))
    if not codes: raise HTTPException(status_code=400, detail="請提供股票代號")

    results, errors = [], []
    for code in codes:
        try:
            df = fetch_finmind(code, req.token)
            r  = analyze(code, df, req)
            try:
                chip = calc_chip(
                    fetch_institutional(code, req.token),
                    fetch_margin(code, req.token)
                )
                r.update(chip)
            except: pass

            ts   = r.get('score', 0) + r.get('chip_score', 0)
            tpct = round(ts / 8 * 100, 1)
            r['total_score']     = ts
            r['total_score_pct'] = tpct
            r['strength'] = 'strong' if ts >= 6 else ('medium' if ts >= 4 else 'weak')
            results.append({k: v for k, v in r.items() if k != '_df'})
        except Exception as e:
            errors.append({'code': code, 'error': str(e)})

    return {
        'results': sorted(results, key=lambda x: x['total_score'], reverse=True),
        'errors': errors, 'scanned': len(codes), 'success': len(results),
    }

@app.get("/api/chart/{code}")
async def get_chart(code: str, token: str):
    try:
        df = fetch_finmind(code, token, days=120)
        df = df.tail(60).copy().reset_index(drop=True)

        def get_col(name, fallback=None):
            if name in df.columns:     return pd.to_numeric(df[name], errors='coerce')
            if fallback and fallback in df.columns: return pd.to_numeric(df[fallback], errors='coerce')
            return pd.Series([None]*len(df))

        close_s = get_col('close')
        open_s  = get_col('open').fillna(close_s)
        high_s  = get_col('max','high').fillna(close_s)
        low_s   = get_col('min','low').fillna(close_s)
        vol_s   = get_col('Trading_Volume','volume').fillna(0)

        ma5  = close_s.rolling(5,  min_periods=1).mean()
        ma10 = close_s.rolling(10, min_periods=1).mean()
        ma20 = close_s.rolling(20, min_periods=1).mean()

        e12  = close_s.ewm(span=12, adjust=False).mean()
        e26  = close_s.ewm(span=26, adjust=False).mean()
        ml   = e12 - e26
        macd = ml - ml.ewm(span=9, adjust=False).mean()

        delta  = close_s.diff()
        gain   = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss   = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rsi    = 100 - 100 / (1 + gain / loss.where(loss != 0, other=1e-9))

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

@app.get("/api/fundamental/{code}")
async def get_fundamental(code: str, token: str):
    """抓取基本面資料：月營收、綜合損益表、股利政策"""
    try:
        from datetime import timezone
        today  = datetime.today()
        y      = today.year
        # 月營收：近12個月
        rev_start = (today - timedelta(days=400)).strftime('%Y-%m-%d')
        rev_end   = today.strftime('%Y-%m-%d')
        # 財報：近2年
        fin_start = f"{y-2}-01-01"
        fin_end   = today.strftime('%Y-%m-%d')
        # 股利：近5年
        div_start = f"{y-5}-01-01"
        div_end   = today.strftime('%Y-%m-%d')

        result = {
            'revenue': [], 'financial': [],
            'dividend': [], 'per_pbr': {},
        }

        # ── 月營收 ──────────────────────────────────────
        try:
            d = finmind_get('TaiwanStockMonthRevenue', code, rev_start, rev_end, token)
            print(f"[fund] revenue msg={d.get('msg')} count={len(d.get('data',[]))}")
            if d.get('data'):
                df = pd.DataFrame(d['data']).sort_values('date')
                for col in ['revenue','last_month_revenue','last_year_month_revenue']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                rows = []
                for _, row in df.tail(13).iterrows():
                    rev    = safe_float(row.get('revenue', 0), 0)
                    lm_rev = safe_float(row.get('last_month_revenue', 0), 0)
                    ly_rev = safe_float(row.get('last_year_month_revenue', 0), 0)
                    mom    = round((rev - lm_rev) / lm_rev * 100, 1) if lm_rev > 0 else 0
                    yoy    = round((rev - ly_rev) / ly_rev * 100, 1) if ly_rev > 0 else 0
                    rows.append({
                        'date':  str(row.get('date','')),
                        'revenue': rev,
                        'mom': mom,  # 月增率
                        'yoy': yoy,  # 年增率
                    })
                result['revenue'] = rows
        except Exception as e:
            print(f"[fund] revenue error: {e}")

        # ── 綜合損益表（EPS、營業利益率） ───────────────
        try:
            d = finmind_get('TaiwanStockFinancialStatements', code, fin_start, fin_end, token)
            print(f"[fund] financial msg={d.get('msg')} count={len(d.get('data',[]))}")
            if d.get('data'):
                df = pd.DataFrame(d['data'])
                df['value'] = pd.to_numeric(df.get('value', pd.Series(dtype=float)), errors='coerce').fillna(0)
                # 取出各季的關鍵指標
                dates = sorted(df['date'].unique())[-8:] if 'date' in df.columns else []
                rows = []
                for dt in dates:
                    sub  = df[df['date'] == dt]
                    def get_val(typ):
                        r = sub[sub['type'] == typ]
                        return safe_float(r['value'].iloc[0], 2) if not r.empty else None
                    eps         = get_val('EPS')
                    op_income   = get_val('營業利益')
                    revenue     = get_val('營業收入')
                    net_income  = get_val('本期淨利（淨損）')
                    op_margin   = round(op_income / revenue * 100, 1) if op_income and revenue and revenue != 0 else None
                    net_margin  = round(net_income / revenue * 100, 1) if net_income and revenue and revenue != 0 else None
                    rows.append({
                        'date': str(dt), 'eps': eps,
                        'op_margin': op_margin, 'net_margin': net_margin,
                        'op_income': op_income, 'revenue': revenue,
                    })
                result['financial'] = [r for r in rows if r['eps'] is not None or r['op_margin'] is not None]
        except Exception as e:
            print(f"[fund] financial error: {e}")

        # ── 股利政策 ─────────────────────────────────────
        try:
            d = finmind_get('TaiwanStockDividend', code, div_start, div_end, token)
            print(f"[fund] dividend msg={d.get('msg')} count={len(d.get('data',[]))}")
            if d.get('data'):
                df = pd.DataFrame(d['data']).sort_values('date')
                for col in ['CashDividend','StockDividend','CashEarningsDistribution',
                            'StockEarningsDistribution']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                rows = []
                for _, row in df.tail(5).iterrows():
                    cash  = safe_float(row.get('CashDividend', 0), 2)
                    stock = safe_float(row.get('StockDividend', 0), 2)
                    total = round(cash + stock, 2)
                    rows.append({
                        'date':  str(row.get('date','')),
                        'cash':  cash,
                        'stock': stock,
                        'total': total,
                    })
                result['dividend'] = rows
        except Exception as e:
            print(f"[fund] dividend error: {e}")

        # ── PER / PBR（從技術面股價資料推算） ────────────
        try:
            d = finmind_get('TaiwanStockPER', code,
                (today - timedelta(days=10)).strftime('%Y-%m-%d'),
                today.strftime('%Y-%m-%d'), token)
            if d.get('data'):
                latest = d['data'][-1]
                result['per_pbr'] = {
                    'date': latest.get('date',''),
                    'per':  safe_float(latest.get('PER', 0), 1),
                    'pbr':  safe_float(latest.get('PBR', 0), 2),
                    'dividend_yield': safe_float(latest.get('dividend_yield', 0), 2),
                }
        except Exception as e:
            print(f"[fund] per_pbr error (non-critical): {e}")

        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


async def get_chip(code: str, token: str):
    """獨立抓取最新籌碼資料（三大法人 + 融資融券）"""
    try:
        inst_df   = fetch_institutional(code, token, days=35)
        margin_df = fetch_margin(code, token, days=35)
        chip      = calc_chip(inst_df, margin_df)
        return chip
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

@app.get("/api/intraday/{code}")
async def get_intraday(code: str, token: str):
    try:
        from datetime import timezone
        url  = (f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.TW"
                f"?interval=1m&range=1d&includePrePost=false")
        resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("chart",{}).get("result",[])
        if not result: raise ValueError("Yahoo Finance 無資料")

        r      = result[0]
        meta   = r.get("meta",{})
        ts     = r.get("timestamp",[])
        quotes = r.get("indicators",{}).get("quote",[{}])[0]
        if not ts: raise ValueError("今日尚無分鐘資料")

        tz8 = timezone(timedelta(hours=8))
        times, opens, highs, lows, closes, vols = [], [], [], [], [], []
        for i, t in enumerate(ts):
            c = quotes.get("close",[None]*len(ts))[i]
            if c is None: continue
            dt = datetime.fromtimestamp(t, tz=timezone.utc).astimezone(tz8)
            times.append(dt.strftime("%H:%M"))
            opens.append(round(float(quotes.get("open", [c]*len(ts))[i] or c), 2))
            highs.append(round(float(quotes.get("high", [c]*len(ts))[i] or c), 2))
            lows.append( round(float(quotes.get("low",  [c]*len(ts))[i] or c), 2))
            closes.append(round(float(c), 2))
            vols.append(int(quotes.get("volume",[0]*len(ts))[i] or 0))

        if not closes: raise ValueError("資料過濾後為空")

        cum_pv, cum_vol, vwap = 0.0, 0, []
        for c, v in zip(closes, vols):
            cum_pv += c*v; cum_vol += v
            vwap.append(round(cum_pv/cum_vol, 2) if cum_vol > 0 else c)

        trade_date  = datetime.fromtimestamp(ts[0], tz=timezone.utc).astimezone(tz8).strftime("%Y-%m-%d")
        status_note = "（盤中即時）" if meta.get("marketState") == "REGULAR" else "（收盤後）"

        return {"date":trade_date,"status":status_note,"times":times,
                "open":opens,"high":highs,"low":lows,"close":closes,"vol":vols,"vwap":vwap,"msg":""}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

@app.post("/api/strategy")
async def get_strategy(req: StrategyRequest):
    try:
        r = req.stock
        code, name, price = r.get('code',''), r.get('name',''), r.get('price',0)
        tech_score   = r.get('score', 0)
        chip_score   = r.get('chip_score', 0)
        total_score  = r.get('total_score', tech_score)
        chip_avail   = r.get('chip_available', False)
        ma_bullish   = r.get('ma_bullish', False)
        macd_positive= r.get('macd_positive', False)
        macd_golden  = r.get('macd_golden_cross', False)
        rsi          = r.get('rsi', 50)
        rsi_strong   = r.get('rsi_strong', False)
        price_break  = r.get('price_breakout', False)
        vol_surge    = r.get('volume_surge', False)
        bb_pos       = r.get('bb_position', 50)
        bb_expanding = r.get('bb_expanding', False)
        bb_upper     = r.get('bb_upper', 0)
        bb_lower     = r.get('bb_lower', 0)
        bb_mid       = r.get('bb_mid', 0)
        foreign_buy  = r.get('foreign_buying', False)
        trust_buy    = r.get('trust_buying', False)
        margin_safe  = r.get('margin_safe', False)
        foreign_5d   = r.get('foreign_5d', 0)
        foreign_20d  = r.get('foreign_20d', 0)
        trust_5d     = r.get('trust_5d', 0)
        vol_ratio    = r.get('vol_ratio', 1)
        margin_chg   = r.get('margin_change_5d', 0)

        # 短線
        ss, sr = [], []
        if macd_golden:  ss.append('MACD 剛發生金叉，短線動能轉強')
        if vol_surge:    ss.append(f'成交量放大至均量 {vol_ratio:.1f} 倍，有資金進場')
        if price_break:  ss.append('突破近20日高點，短線有突破動能')
        if rsi_strong:   ss.append(f'RSI {rsi:.0f} 在強勢區間，多頭動能足夠')
        if bb_pos > 70 and bb_expanding: ss.append('布林通道擴張且偏上軌，短線趨勢向上')
        if rsi > 75:     sr.append(f'RSI {rsi:.0f} 已過熱，短線回調風險')
        if bb_pos > 90:  sr.append(f'股價貼近布林上軌 {bb_upper:.2f}，短線壓力大')
        if not vol_surge and price_break: sr.append('突破時量能不足，留意是否假突破')
        sa = '積極' if len(ss)>=3 else ('中性偏多' if len(ss)>=2 else ('謹慎' if sr else '觀望'))
        sn = (f'多項短線訊號同時出現，可考慮短線進場，留意 {bb_upper:.2f} 壓力' if len(ss)>=3
              else '短線訊號尚可，建議小量試單，嚴設停損' if len(ss)>=2
              else '短線風險訊號較多，建議觀望或減碼' if sr else '短線訊號不明確，等待更好進場時機')

        # 中線
        ms, mr = [], []
        if ma_bullish:   ms.append('均線多頭排列（MA5>MA10>MA20），中線趨勢向上')
        if macd_positive:ms.append('MACD 柱狀維持正值，中線多頭動能持續')
        if foreign_buy and chip_avail: ms.append(f'外資近5日累積買超 {foreign_5d:,.0f} 張，法人持續佈局')
        if trust_buy and chip_avail:   ms.append(f'投信近5日買超 {trust_5d:,.0f} 張，機構資金進場')
        if margin_safe and chip_avail: ms.append('融資餘額下降，籌碼轉健康，主力控盤跡象')
        if bb_expanding and bb_pos>50: ms.append('布林通道擴張且股價在中軌以上，中線趨勢確立')
        if foreign_20d < 0 and chip_avail: mr.append(f'外資近20日賣超 {abs(foreign_20d):,.0f} 張，中線偏空')
        if not ma_bullish: mr.append('均線尚未完整多頭排列，中線趨勢未確立')
        if margin_chg > 0 and chip_avail: mr.append('融資餘額上升，散戶追高風險，籌碼不穩')
        ma = ('積極佈局' if len(ms)>=4 else '逢低布局' if len(ms)>=2 else '減碼觀察' if mr else '中性觀察')
        mn = (f'技術面與籌碼面雙重確認，中線趨勢明確，可分批建立部位，支撐參考 MA20 {bb_mid:.2f}' if len(ms)>=4
              else '中線訊號偏正向，建議分批進場，以 MA20 或布林中軌為支撐判斷' if len(ms)>=2
              else '中線風險因素較多，建議降低持倉比重' if mr else '中線方向尚未明確，持續追蹤法人動向')

        # 長線
        ls, lr = [], []
        if foreign_20d > 0 and chip_avail: ls.append(f'外資近20日持續買超 {foreign_20d:,.0f} 張，長線法人看多')
        if ma_bullish and macd_positive:   ls.append('技術面趨勢結構完整，均線多頭 + MACD 正值')
        if margin_safe and chip_avail:     ls.append('籌碼面健康，融資不追高，有助長線走揚')
        if total_score >= 6:               ls.append(f'綜合評分 {total_score}/8，技術面與籌碼面均強')
        if not chip_avail: lr.append('籌碼資料未能取得，長線判斷依賴技術面為主')
        if rsi > 75:       lr.append(f'目前 RSI 偏高，長線進場位置不理想，等待回調')
        if bb_pos > 85:    lr.append(f'股價偏高，距布林下軌 {bb_lower:.2f} 較遠，進場風險高')
        la = ('長線買進' if len(ls)>=3 and not lr else '分批建倉' if len(ls)>=2 else '等待時機' if lr else '持續觀察')
        ln = (f'多項長線正向訊號，適合分批建立長期部位，支撐參考 {bb_lower:.2f}（布林下軌）' if len(ls)>=3 and not lr
              else '長線條件尚可，建議小量分批建倉，持續追蹤法人動向' if len(ls)>=2
              else '目前進場位置風險偏高，建議等待回調再考慮長線佈局' if lr else '長線訊號尚不明確，持續追蹤')

        warnings = []
        if not chip_avail: warnings.append('⚠️ 籌碼面資料取得失敗，建議僅基於技術面參考')
        if rsi > 80:       warnings.append(f'⚠️ RSI {rsi:.0f} 極度過熱，任何操作需謹慎')
        if bb_pos > 95:    warnings.append('⚠️ 股價已突破布林上軌，回調風險極高')
        warnings.append('⚠️ 以上為技術面與籌碼面分析，不構成投資建議，操作請自行判斷並設定停損')

        return {
            'code':code,'name':name,'price':price,
            'total_score':total_score,'tech_score':tech_score,'chip_score':chip_score,
            'short':{'action':sa,'note':sn,'signals':ss,'risks':sr},
            'mid':  {'action':ma,'note':mn,'signals':ms,'risks':mr},
            'long': {'action':la,'note':ln,'signals':ls,'risks':lr},
            'risk_warnings': warnings,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

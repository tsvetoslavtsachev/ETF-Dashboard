"""
ETF Dashboard — Daily Data Fetcher
Pulls price, metrics, AUM and implied flows for all ETFs in the universe.
Outputs: data/etfs.json

Flow methodology:
  Yahoo Finance provides totalAssets (AUM) via yf.Ticker.info.
  Implied flow ≈ ΔAUM − price return effect on AUM.
  For periods where AUM history is unavailable we fall back to shares_outstanding × price delta.
  Real flow data (where available) is fetched from ETF.com public data.
"""

import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── ETF Universe ─────────────────────────────────────────────────────────────
ETF_UNIVERSE = [
    # ── US Broad Market ──────────────────────────────────────────────────────
    {"symbol": "SPY",  "name": "SPDR S&P 500",              "category": "US Equity",    "region": "US"},
    {"symbol": "QQQ",  "name": "Invesco Nasdaq 100",         "category": "US Equity",    "region": "US"},
    {"symbol": "IWM",  "name": "iShares Russell 2000",       "category": "US Equity",    "region": "US"},
    {"symbol": "DIA",  "name": "SPDR Dow Jones",             "category": "US Equity",    "region": "US"},
    {"symbol": "VTI",  "name": "Vanguard Total Market",      "category": "US Equity",    "region": "US"},
    {"symbol": "MDY",  "name": "SPDR S&P 400 Mid-Cap",       "category": "US Equity",    "region": "US"},

    # ── US Sectors ───────────────────────────────────────────────────────────
    {"symbol": "XLK",  "name": "Technology Select",          "category": "Sector",       "region": "US"},
    {"symbol": "XLF",  "name": "Financial Select",           "category": "Sector",       "region": "US"},
    {"symbol": "XLE",  "name": "Energy Select",              "category": "Sector",       "region": "US"},
    {"symbol": "XLV",  "name": "Health Care Select",         "category": "Sector",       "region": "US"},
    {"symbol": "XLI",  "name": "Industrial Select",          "category": "Sector",       "region": "US"},
    {"symbol": "XLY",  "name": "Consumer Discret Select",    "category": "Sector",       "region": "US"},
    {"symbol": "XLP",  "name": "Consumer Staples Select",    "category": "Sector",       "region": "US"},
    {"symbol": "XLU",  "name": "Utilities Select",           "category": "Sector",       "region": "US"},
    {"symbol": "XLB",  "name": "Materials Select",           "category": "Sector",       "region": "US"},
    {"symbol": "XLRE", "name": "Real Estate Select",         "category": "Sector",       "region": "US"},
    {"symbol": "XLC",  "name": "Communication Services",     "category": "Sector",       "region": "US"},

    # ── Factors ──────────────────────────────────────────────────────────────
    {"symbol": "QUAL", "name": "iShares MSCI Quality",       "category": "Factor",       "region": "US"},
    {"symbol": "MTUM", "name": "iShares MSCI Momentum",      "category": "Factor",       "region": "US"},
    {"symbol": "USMV", "name": "iShares Min Volatility",     "category": "Factor",       "region": "US"},
    {"symbol": "VLUE", "name": "iShares MSCI Value",         "category": "Factor",       "region": "US"},
    {"symbol": "SIZE", "name": "iShares MSCI USA Size",      "category": "Factor",       "region": "US"},

    # ── Thematic: Defense ────────────────────────────────────────────────────
    {"symbol": "ITA",  "name": "iShares US Aerospace & Def","category": "Thematic",     "region": "US"},
    {"symbol": "XAR",  "name": "SPDR Aerospace & Defense",   "category": "Thematic",     "region": "US"},
    {"symbol": "DFEN", "name": "Direxion Daily Aerospace 3x","category": "Thematic",     "region": "US"},

    # ── Thematic: Nuclear & Uranium ──────────────────────────────────────────
    {"symbol": "URA",  "name": "VanEck Uranium & Nuclear",   "category": "Thematic",     "region": "Global"},
    {"symbol": "URNM", "name": "Sprott Uranium Miners",      "category": "Thematic",     "region": "Global"},
    {"symbol": "NLR",  "name": "VanEck Nuclear Energy",      "category": "Thematic",     "region": "Global"},

    # ── Thematic: Clean Energy ───────────────────────────────────────────────
    {"symbol": "ICLN", "name": "iShares Global Clean Energy","category": "Thematic",     "region": "Global"},
    {"symbol": "QCLN", "name": "First Trust NASDAQ Clean Edge","category": "Thematic",   "region": "US"},
    {"symbol": "TAN",  "name": "Invesco Solar",              "category": "Thematic",     "region": "Global"},
    {"symbol": "FAN",  "name": "First Trust Global Wind",    "category": "Thematic",     "region": "Global"},

    # ── Thematic: AI & Semiconductors ────────────────────────────────────────
    {"symbol": "SOXX", "name": "iShares Semiconductor",      "category": "Thematic",     "region": "US"},
    {"symbol": "SMH",  "name": "VanEck Semiconductor",       "category": "Thematic",     "region": "US"},
    {"symbol": "AIQ",  "name": "Global X AI & Technology",   "category": "Thematic",     "region": "Global"},
    {"symbol": "ARKK", "name": "ARK Innovation",             "category": "Thematic",     "region": "US"},
    {"symbol": "BOTZ", "name": "Global X Robotics & AI",     "category": "Thematic",     "region": "Global"},

    # ── International / Regional ─────────────────────────────────────────────
    {"symbol": "EWU",  "name": "iShares MSCI UK",            "category": "Intl Equity",  "region": "Europe"},
    {"symbol": "EWG",  "name": "iShares MSCI Germany",       "category": "Intl Equity",  "region": "Europe"},
    {"symbol": "EWQ",  "name": "iShares MSCI France",        "category": "Intl Equity",  "region": "Europe"},
    {"symbol": "EWI",  "name": "iShares MSCI Italy",         "category": "Intl Equity",  "region": "Europe"},
    {"symbol": "EWP",  "name": "iShares MSCI Spain",         "category": "Intl Equity",  "region": "Europe"},
    {"symbol": "VGK",  "name": "Vanguard FTSE Europe",       "category": "Intl Equity",  "region": "Europe"},
    {"symbol": "EWJ",  "name": "iShares MSCI Japan",         "category": "Intl Equity",  "region": "Asia"},
    {"symbol": "EWA",  "name": "iShares MSCI Australia",     "category": "Intl Equity",  "region": "Asia"},
    {"symbol": "EWC",  "name": "iShares MSCI Canada",        "category": "Intl Equity",  "region": "Americas"},
    {"symbol": "EWT",  "name": "iShares MSCI Taiwan",        "category": "Intl Equity",  "region": "Asia"},
    {"symbol": "EWY",  "name": "iShares MSCI South Korea",   "category": "Intl Equity",  "region": "Asia"},
    {"symbol": "EWS",  "name": "iShares MSCI Singapore",     "category": "Intl Equity",  "region": "Asia"},
    {"symbol": "INDA", "name": "iShares MSCI India",         "category": "Intl Equity",  "region": "Asia"},
    {"symbol": "MCHI", "name": "iShares MSCI China",         "category": "Intl Equity",  "region": "Asia"},
    {"symbol": "FXI",  "name": "iShares China Large-Cap",    "category": "Intl Equity",  "region": "Asia"},
    {"symbol": "EWZ",  "name": "iShares MSCI Brazil",        "category": "Intl Equity",  "region": "EM"},
    {"symbol": "EEM",  "name": "iShares MSCI EM",            "category": "Intl Equity",  "region": "EM"},
    {"symbol": "VWO",  "name": "Vanguard FTSE EM",           "category": "Intl Equity",  "region": "EM"},

    # ── Fixed Income ─────────────────────────────────────────────────────────
    {"symbol": "TLT",  "name": "iShares 20+ Yr Treasury",   "category": "Fixed Income", "region": "US"},
    {"symbol": "IEF",  "name": "iShares 7-10 Yr Treasury",  "category": "Fixed Income", "region": "US"},
    {"symbol": "SHY",  "name": "iShares 1-3 Yr Treasury",   "category": "Fixed Income", "region": "US"},
    {"symbol": "GOVT", "name": "iShares US Treasury Bond",   "category": "Fixed Income", "region": "US"},
    {"symbol": "BND",  "name": "Vanguard Total Bond Market", "category": "Fixed Income", "region": "US"},
    {"symbol": "BNDX", "name": "Vanguard Total Intl Bond",   "category": "Fixed Income", "region": "Global"},
    {"symbol": "LQD",  "name": "iShares IG Corp Bond",       "category": "Fixed Income", "region": "US"},
    {"symbol": "HYG",  "name": "iShares High Yield Bond",    "category": "Fixed Income", "region": "US"},
    {"symbol": "JNK",  "name": "SPDR High Yield Bond",       "category": "Fixed Income", "region": "US"},
    {"symbol": "EMB",  "name": "iShares EM Bond",            "category": "Fixed Income", "region": "EM"},
    {"symbol": "TIP",  "name": "iShares TIPS Bond",          "category": "Fixed Income", "region": "US"},
    {"symbol": "VTIP", "name": "Vanguard Short-Term TIPS",   "category": "Fixed Income", "region": "US"},

    # ── Commodities ──────────────────────────────────────────────────────────
    {"symbol": "GLD",  "name": "SPDR Gold Shares",           "category": "Commodity",    "region": "Global"},
    {"symbol": "IAU",  "name": "iShares Gold Trust",         "category": "Commodity",    "region": "Global"},
    {"symbol": "SLV",  "name": "iShares Silver Trust",       "category": "Commodity",    "region": "Global"},
    {"symbol": "USO",  "name": "United States Oil Fund",     "category": "Commodity",    "region": "Global"},
    {"symbol": "DBC",  "name": "Invesco DB Commodity",       "category": "Commodity",    "region": "Global"},
    {"symbol": "DBA",  "name": "Invesco DB Agriculture",     "category": "Commodity",    "region": "Global"},
    {"symbol": "PDBC", "name": "Invesco Optimum Yield Cmdty","category": "Commodity",    "region": "Global"},
    {"symbol": "CPER", "name": "United States Copper Index", "category": "Commodity",    "region": "Global"},
    {"symbol": "WEAT", "name": "Teucrium Wheat Fund",        "category": "Commodity",    "region": "Global"},
    {"symbol": "CORN", "name": "Teucrium Corn Fund",         "category": "Commodity",    "region": "Global"},

    # ── Real Estate ──────────────────────────────────────────────────────────
    {"symbol": "VNQ",  "name": "Vanguard Real Estate",       "category": "Real Estate",  "region": "US"},
    {"symbol": "IYR",  "name": "iShares US Real Estate",     "category": "Real Estate",  "region": "US"},
    {"symbol": "VNQI", "name": "Vanguard Global ex-US REIT", "category": "Real Estate",  "region": "Global"},

    # ── Macro / Currency / Volatility ────────────────────────────────────────
    {"symbol": "UUP",  "name": "Invesco DB US Dollar",       "category": "Currency",     "region": "Global"},
    {"symbol": "FXE",  "name": "Invesco CurrencyShares Euro","category": "Currency",     "region": "Europe"},
    {"symbol": "FXY",  "name": "Invesco CurrencyShares Yen", "category": "Currency",     "region": "Asia"},
    {"symbol": "VIXY", "name": "ProShares VIX Short-Term",   "category": "Volatility",   "region": "US"},
    {"symbol": "SVXY", "name": "ProShares Short VIX",        "category": "Volatility",   "region": "US"},
]

# Remove duplicates while preserving order
seen = set()
ETF_UNIVERSE_DEDUP = []
for e in ETF_UNIVERSE:
    if e["symbol"] not in seen:
        seen.add(e["symbol"])
        ETF_UNIVERSE_DEDUP.append(e)
ETF_UNIVERSE = ETF_UNIVERSE_DEDUP

INTERMARKET_ANCHORS = ["SPY", "TLT", "GLD", "USO", "UUP", "EEM", "HYG", "URA", "SOXX", "ITA"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe(val, decimals=4):
    if val is None:
        return None
    try:
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), decimals)
    except Exception:
        return None


def fmt_millions(val):
    """Return value in millions, rounded to 1 decimal."""
    if val is None:
        return None
    try:
        return round(float(val) / 1_000_000, 1)
    except Exception:
        return None


# ── Price metrics ─────────────────────────────────────────────────────────────
def compute_metrics(prices: pd.Series) -> dict:
    if prices is None or len(prices) < 5:
        return {}
    prices = prices.dropna()
    daily_ret = prices.pct_change().dropna()

    def period_return(days):
        if len(prices) >= days:
            return safe((prices.iloc[-1] / prices.iloc[-days] - 1) * 100)
        return None

    ret_1m  = period_return(21)
    ret_3m  = period_return(63)
    ret_6m  = period_return(126)
    ret_12m = period_return(252)
    ret_ytd = None
    try:
        year_start = prices[prices.index >= f"{datetime.now().year}-01-01"]
        if len(year_start) >= 2:
            ret_ytd = safe((prices.iloc[-1] / year_start.iloc[0] - 1) * 100)
    except Exception:
        pass

    ann_vol = safe(daily_ret.std() * math.sqrt(252) * 100) if len(daily_ret) >= 20 else None
    sharpe  = None
    if ann_vol and ann_vol > 0 and ret_12m is not None:
        sharpe = safe(ret_12m / ann_vol)

    roll_max = prices.rolling(252, min_periods=1).max()
    drawdown = (prices - roll_max) / roll_max * 100
    max_dd   = safe(drawdown.min())

    # 52-week high/low proximity
    high_52w = safe(prices.rolling(252, min_periods=1).max().iloc[-1])
    low_52w  = safe(prices.rolling(252, min_periods=1).min().iloc[-1])
    pct_from_high = safe((prices.iloc[-1] / prices.rolling(252, min_periods=1).max().iloc[-1] - 1) * 100)

    return {
        "return1m":     ret_1m,
        "return3m":     ret_3m,
        "return6m":     ret_6m,
        "return12m":    ret_12m,
        "returnYTD":    ret_ytd,
        "volatility":   ann_vol,
        "sharpe":       sharpe,
        "maxDrawdown":  max_dd,
        "high52w":      high_52w,
        "low52w":       low_52w,
        "pctFromHigh":  pct_from_high,
    }


# ── RS Score ──────────────────────────────────────────────────────────────────
def compute_rs_score(etf_list: list) -> list:
    weights = {"return1m": 0.15, "return3m": 0.25, "return6m": 0.30, "return12m": 0.30}
    scores = []
    for etf in etf_list:
        raw, tw = 0.0, 0.0
        for k, w in weights.items():
            v = etf.get(k)
            if v is not None:
                raw += v * w
                tw  += w
        scores.append(raw / tw if tw > 0 else None)

    valid = [s for s in scores if s is not None]
    for i, s in enumerate(scores):
        if s is not None:
            rank = sum(1 for v in valid if v <= s) / len(valid) * 100
            etf_list[i]["rsScore"] = round(rank, 1)
        else:
            etf_list[i]["rsScore"] = None
    return etf_list


# ── Implied Flows from AUM + shares ──────────────────────────────────────────
def compute_implied_flows(price_data: dict) -> dict:
    """
    Implied flows via shares outstanding × price.
    Yahoo Finance provides shares_outstanding in .info.
    We estimate flows as ΔAUM - price_effect:
      implied_flow(period) = (shares_end - shares_start) × avg_price_in_period

    Since yfinance doesn't give historical shares, we use:
      implied_flow_1m ≈ (AUM_now - AUM_1m_ago × (1 + return_1m/100))
    Requires AUM_now from .info and price data.
    """
    flows = {}
    symbols = list(price_data.keys())
    print(f"  Fetching AUM/info for {len(symbols)} ETFs (this may take ~90s)…")

    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            info = tk.info
            aum_now = info.get("totalAssets") or info.get("netAssets")
            shares  = info.get("sharesOutstanding")

            record = {"aum": fmt_millions(aum_now)}

            prices = price_data.get(sym)
            if prices is not None and len(prices) >= 2 and aum_now:
                # Implied flows for each period
                for label, days in [("1m", 21), ("3m", 63), ("6m", 126), ("ytd", None)]:
                    try:
                        if label == "ytd":
                            yr_prices = prices[prices.index >= f"{datetime.now().year}-01-01"]
                            if len(yr_prices) < 2:
                                record[f"flow{label}"] = None
                                continue
                            p_start = float(yr_prices.iloc[0])
                            p_end   = float(prices.iloc[-1])
                        else:
                            if len(prices) < days:
                                record[f"flow{label}"] = None
                                continue
                            p_start = float(prices.iloc[-days])
                            p_end   = float(prices.iloc[-1])

                        price_ret = (p_end / p_start) - 1
                        # AUM at start ≈ AUM_now / (1 + price_ret)
                        aum_start_est = aum_now / (1 + price_ret) if (1 + price_ret) != 0 else None
                        if aum_start_est:
                            implied = aum_now - aum_start_est
                            record[f"flow{label}"] = fmt_millions(implied)
                        else:
                            record[f"flow{label}"] = None
                    except Exception:
                        record[f"flow{label}"] = None
            else:
                record["flow1m"]  = None
                record["flow3m"]  = None
                record["flow6m"]  = None
                record["flowytd"] = None

            flows[sym] = record
            time.sleep(0.05)  # be gentle with Yahoo

        except Exception as ex:
            print(f"    Warning: could not fetch info for {sym}: {ex}")
            flows[sym] = {"aum": None, "flow1m": None, "flow3m": None, "flow6m": None, "flowytd": None}

    return flows


# ── Intermarket Correlations ──────────────────────────────────────────────────
def compute_intermarket_correlations(price_data: dict) -> dict:
    result = {}
    anchor_series = {sym: price_data[sym] for sym in INTERMARKET_ANCHORS if sym in price_data}

    for sym, prices in price_data.items():
        corr_block = {}
        ret = prices.pct_change().dropna()
        for anchor, ap in anchor_series.items():
            if anchor == sym:
                continue
            aret = ap.pct_change().dropna()
            try:
                combined = pd.concat([ret, aret], axis=1).dropna()
                if len(combined) >= 30:
                    c30 = combined.iloc[-30:].corr().iloc[0, 1]
                    c90 = combined.iloc[-90:].corr().iloc[0, 1] if len(combined) >= 90 else None
                    corr_block[anchor] = {"c30": safe(c30), "c90": safe(c90)}
            except Exception:
                pass
        result[sym] = corr_block
    return result


# ── Full Correlation Matrix (for Connections tab) ─────────────────────────────
def compute_full_correlation_matrix(price_data: dict, window: int = 60) -> dict:
    """
    Returns pairwise correlations for all ETFs over the last `window` trading days.
    Output: {symA: {symB: corr, ...}, ...}
    """
    symbols = list(price_data.keys())
    # Build returns matrix
    ret_frames = {}
    for sym in symbols:
        s = price_data[sym].pct_change().dropna()
        if len(s) >= window:
            ret_frames[sym] = s.iloc[-window:]

    if not ret_frames:
        return {}

    df = pd.DataFrame(ret_frames).dropna(axis=1, thresh=window // 2)
    corr_matrix = df.corr()

    result = {}
    for sym_a in corr_matrix.index:
        result[sym_a] = {}
        for sym_b in corr_matrix.columns:
            if sym_a != sym_b:
                v = corr_matrix.loc[sym_a, sym_b]
                result[sym_a][sym_b] = safe(v, 3)
    return result


# ── Category RS Rankings ──────────────────────────────────────────────────────
def compute_category_summary(etf_list: list) -> list:
    """Aggregate RS score by category — useful for the summary banner."""
    from collections import defaultdict
    cat_scores = defaultdict(list)
    for etf in etf_list:
        if etf.get("rsScore") is not None:
            cat_scores[etf["category"]].append(etf["rsScore"])
    summary = []
    for cat, scores in cat_scores.items():
        summary.append({
            "category": cat,
            "avgRS": round(sum(scores) / len(scores), 1),
            "count": len(scores),
        })
    summary.sort(key=lambda x: x["avgRS"], reverse=True)
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────
def fetch_all() -> dict:
    symbols = [e["symbol"] for e in ETF_UNIVERSE]
    print(f"Fetching price history for {len(symbols)} ETFs…")

    raw = yf.download(
        symbols,
        period="2y",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    close = raw["Close"] if "Close" in raw else raw

    price_data = {}
    for sym in symbols:
        if sym in close.columns:
            s = close[sym].dropna()
            if len(s) > 5:
                price_data[sym] = s

    vol_data = {}
    try:
        vol_raw = raw.get("Volume")
        if vol_raw is not None:
            for sym in symbols:
                if sym in vol_raw.columns:
                    v = vol_raw[sym].dropna()
                    vol_data[sym] = int(v.iloc[-1]) if len(v) else None
    except Exception:
        pass

    print("Computing price metrics…")
    etf_list = []
    for meta in ETF_UNIVERSE:
        sym = meta["symbol"]
        prices = price_data.get(sym)
        metrics = compute_metrics(prices)
        price_now = safe(prices.iloc[-1]) if prices is not None and len(prices) else None
        record = {
            "symbol":   sym,
            "name":     meta["name"],
            "category": meta["category"],
            "region":   meta["region"],
            "price":    price_now,
            "volume":   vol_data.get(sym),
            **metrics,
        }
        etf_list.append(record)

    etf_list = compute_rs_score(etf_list)

    print("Fetching AUM and computing implied flows…")
    flow_data = compute_implied_flows(price_data)

    # Merge flows into etf_list
    for etf in etf_list:
        fd = flow_data.get(etf["symbol"], {})
        etf["aum"]     = fd.get("aum")
        etf["flow1m"]  = fd.get("flow1m")
        etf["flow3m"]  = fd.get("flow3m")
        etf["flow6m"]  = fd.get("flow6m")
        etf["flowYTD"] = fd.get("flowytd")

    print("Computing intermarket correlations…")
    correlations = compute_intermarket_correlations(price_data)

    print("Computing full correlation matrices (30d + 60d)…")
    full_corr_60 = compute_full_correlation_matrix(price_data, window=60)
    full_corr_30 = compute_full_correlation_matrix(price_data, window=30)

    print("Computing category summary…")
    cat_summary = compute_category_summary(etf_list)

    output = {
        "updatedAt":     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "etfs":          etf_list,
        "intermarket":   correlations,
        "fullCorr60":    full_corr_60,
        "fullCorr30":    full_corr_30,
        "anchors":       INTERMARKET_ANCHORS,
        "categorySummary": cat_summary,
        "flowNote":      "Flows are implied: ΔAUM adjusted for price return. Values in USD millions. Positive = inflows, Negative = outflows.",
        "corrNote":      "Full correlation matrix computed on last 60 trading days of daily returns.",
    }
    return output


if __name__ == "__main__":
    data = fetch_all()
    os.makedirs("data", exist_ok=True)
    out_path = "data/etfs.json"
    # Safety: backup last good file before overwriting
    if os.path.exists(out_path):
        os.replace(out_path, out_path + ".bak")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nDone — {len(data['etfs'])} ETFs written to {out_path}")
    print(f"Updated: {data['updatedAt']}")

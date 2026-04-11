"""
ETF Dashboard — Daily Data Fetcher
Pulls price + metrics for all ETFs in the universe via yfinance.
Outputs: data/etfs.json
"""

import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# ── ETF Universe ─────────────────────────────────────────────────────────────
ETF_UNIVERSE = [
    # US Broad Market
    {"symbol": "SPY",  "name": "SPDR S&P 500",           "category": "US Equity",     "region": "US"},
    {"symbol": "QQQ",  "name": "Invesco Nasdaq 100",      "category": "US Equity",     "region": "US"},
    {"symbol": "IWM",  "name": "iShares Russell 2000",    "category": "US Equity",     "region": "US"},
    {"symbol": "DIA",  "name": "SPDR Dow Jones",          "category": "US Equity",     "region": "US"},
    {"symbol": "VTI",  "name": "Vanguard Total Market",   "category": "US Equity",     "region": "US"},
    # US Sectors
    {"symbol": "XLK",  "name": "Technology Select",       "category": "Sector",        "region": "US"},
    {"symbol": "XLF",  "name": "Financial Select",        "category": "Sector",        "region": "US"},
    {"symbol": "XLE",  "name": "Energy Select",           "category": "Sector",        "region": "US"},
    {"symbol": "XLV",  "name": "Health Care Select",      "category": "Sector",        "region": "US"},
    {"symbol": "XLI",  "name": "Industrial Select",       "category": "Sector",        "region": "US"},
    {"symbol": "XLY",  "name": "Consumer Discret Select", "category": "Sector",        "region": "US"},
    {"symbol": "XLP",  "name": "Consumer Staples Select", "category": "Sector",        "region": "US"},
    {"symbol": "XLU",  "name": "Utilities Select",        "category": "Sector",        "region": "US"},
    {"symbol": "XLB",  "name": "Materials Select",        "category": "Sector",        "region": "US"},
    {"symbol": "XLRE", "name": "Real Estate Select",      "category": "Sector",        "region": "US"},
    # International / Regional
    {"symbol": "EWU",  "name": "iShares MSCI UK",         "category": "Intl Equity",   "region": "Europe"},
    {"symbol": "EWG",  "name": "iShares MSCI Germany",    "category": "Intl Equity",   "region": "Europe"},
    {"symbol": "EWQ",  "name": "iShares MSCI France",     "category": "Intl Equity",   "region": "Europe"},
    {"symbol": "EWI",  "name": "iShares MSCI Italy",      "category": "Intl Equity",   "region": "Europe"},
    {"symbol": "EWP",  "name": "iShares MSCI Spain",      "category": "Intl Equity",   "region": "Europe"},
    {"symbol": "VGK",  "name": "Vanguard FTSE Europe",    "category": "Intl Equity",   "region": "Europe"},
    {"symbol": "EWJ",  "name": "iShares MSCI Japan",      "category": "Intl Equity",   "region": "Asia"},
    {"symbol": "EWZ",  "name": "iShares MSCI Brazil",     "category": "Intl Equity",   "region": "EM"},
    {"symbol": "EEM",  "name": "iShares MSCI EM",         "category": "Intl Equity",   "region": "EM"},
    {"symbol": "FXI",  "name": "iShares China Large-Cap", "category": "Intl Equity",   "region": "Asia"},
    # Fixed Income
    {"symbol": "TLT",  "name": "iShares 20+ Yr Treasury", "category": "Fixed Income",  "region": "US"},
    {"symbol": "IEF",  "name": "iShares 7-10 Yr Treasury","category": "Fixed Income",  "region": "US"},
    {"symbol": "SHY",  "name": "iShares 1-3 Yr Treasury", "category": "Fixed Income",  "region": "US"},
    {"symbol": "LQD",  "name": "iShares IG Corp Bond",    "category": "Fixed Income",  "region": "US"},
    {"symbol": "HYG",  "name": "iShares High Yield Bond", "category": "Fixed Income",  "region": "US"},
    {"symbol": "EMB",  "name": "iShares EM Bond",         "category": "Fixed Income",  "region": "EM"},
    # Commodities
    {"symbol": "GLD",  "name": "SPDR Gold Shares",        "category": "Commodity",     "region": "Global"},
    {"symbol": "SLV",  "name": "iShares Silver Trust",    "category": "Commodity",     "region": "Global"},
    {"symbol": "USO",  "name": "United States Oil Fund",  "category": "Commodity",     "region": "Global"},
    {"symbol": "DBA",  "name": "Invesco DB Agriculture",  "category": "Commodity",     "region": "Global"},
    {"symbol": "PDBC", "name": "Invesco Commodity",       "category": "Commodity",     "region": "Global"},
    # Real Assets / Alternatives
    {"symbol": "VNQ",  "name": "Vanguard Real Estate",    "category": "Real Estate",   "region": "US"},
    {"symbol": "IAU",  "name": "iShares Gold Trust",      "category": "Commodity",     "region": "Global"},
    # Volatility / Macro
    {"symbol": "UUP",  "name": "Invesco DB US Dollar",    "category": "Currency",      "region": "Global"},
    {"symbol": "TIP",  "name": "iShares TIPS Bond",       "category": "Fixed Income",  "region": "US"},
]

INTERMARKET_ANCHORS = ["SPY", "TLT", "GLD", "USO", "UUP", "EEM", "HYG"]


def safe(val):
    """Convert NaN/Inf to None for JSON serialisation."""
    if val is None:
        return None
    try:
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 4)
    except Exception:
        return None


def compute_metrics(prices: pd.Series) -> dict:
    """Return return/risk metrics from a price series."""
    if prices is None or len(prices) < 5:
        return {}

    prices = prices.dropna()
    daily_ret = prices.pct_change().dropna()

    def period_return(days):
        if len(prices) >= days:
            return safe((prices.iloc[-1] / prices.iloc[-days] - 1) * 100)
        return None

    # Returns
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

    # Risk
    ann_vol = safe(daily_ret.std() * math.sqrt(252) * 100) if len(daily_ret) >= 20 else None
    sharpe  = None
    if ann_vol and ann_vol > 0 and ret_12m is not None:
        sharpe = safe((ret_12m / ann_vol))

    # Max drawdown (1-year)
    roll_max = prices.rolling(252, min_periods=1).max()
    drawdown = ((prices - roll_max) / roll_max * 100)
    max_dd   = safe(drawdown.min())

    return {
        "return1m":  ret_1m,
        "return3m":  ret_3m,
        "return6m":  ret_6m,
        "return12m": ret_12m,
        "returnYTD": ret_ytd,
        "volatility": ann_vol,
        "sharpe":    sharpe,
        "maxDrawdown": max_dd,
    }


def compute_rs_score(etf_metrics: list) -> list:
    """Rank each ETF 0-100 by composite momentum (weighted average of return periods)."""
    weights = {"return1m": 0.15, "return3m": 0.25, "return6m": 0.30, "return12m": 0.30}
    scores = []
    for etf in etf_metrics:
        raw = 0.0
        tw  = 0.0
        for k, w in weights.items():
            v = etf.get(k)
            if v is not None:
                raw += v * w
                tw  += w
        scores.append(raw / tw if tw > 0 else None)

    # Percentile rank
    valid = [s for s in scores if s is not None]
    for i, s in enumerate(scores):
        if s is not None:
            rank = sum(1 for v in valid if v <= s) / len(valid) * 100
            etf_metrics[i]["rsScore"] = round(rank, 1)
        else:
            etf_metrics[i]["rsScore"] = None
    return etf_metrics


def compute_intermarket_correlations(price_data: dict) -> dict:
    """30-day & 90-day correlations of every ETF vs anchor tickers."""
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


def fetch_all() -> dict:
    symbols = [e["symbol"] for e in ETF_UNIVERSE]
    print(f"Fetching {len(symbols)} ETFs from Yahoo Finance…")

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
            price_data[sym] = close[sym].dropna()

    # Volume (last close volume)
    vol_data = {}
    try:
        vol_raw = raw["Volume"] if "Volume" in raw else None
        if vol_raw is not None:
            for sym in symbols:
                if sym in vol_raw.columns:
                    vol_data[sym] = int(vol_raw[sym].dropna().iloc[-1]) if len(vol_raw[sym].dropna()) else None
    except Exception:
        pass

    print("Computing metrics…")
    etf_list = []
    for meta in ETF_UNIVERSE:
        sym = meta["symbol"]
        prices = price_data.get(sym)
        metrics = compute_metrics(prices)

        current_price = safe(prices.iloc[-1]) if prices is not None and len(prices) else None

        record = {
            "symbol":   sym,
            "name":     meta["name"],
            "category": meta["category"],
            "region":   meta["region"],
            "price":    current_price,
            "volume":   vol_data.get(sym),
            **metrics,
        }
        etf_list.append(record)

    etf_list = compute_rs_score(etf_list)

    print("Computing intermarket correlations…")
    correlations = compute_intermarket_correlations(price_data)

    output = {
        "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "etfs": etf_list,
        "intermarket": correlations,
        "anchors": INTERMARKET_ANCHORS,
    }
    return output


if __name__ == "__main__":
    data = fetch_all()
    os.makedirs("data", exist_ok=True)
    with open("data/etfs.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Done — {len(data['etfs'])} ETFs written to data/etfs.json")
    print(f"Updated: {data['updatedAt']}")

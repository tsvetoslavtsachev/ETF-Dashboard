# ETF Dashboard

Sortable ETF dashboard with heatmap, relative strength table, exposure drill-down and intermarket correlation panel.

**Live site:** https://tsvetoslavtsachev.github.io/ETF-Dashboard/

## Structure

```
ETF-Dashboard/
├── index.html              # Dashboard UI (single-file, no build step)
├── fetch_data.py           # Data fetcher — pulls from Yahoo Finance
├── requirements.txt        # Python dependencies
├── data/
│   └── etfs.json           # Auto-generated data file (do not edit manually)
└── .github/workflows/
    ├── update-data.yml     # Daily data update + deploy (Mon–Fri 22:00 UTC)
    └── deploy-pages.yml    # Deploy on every push to main
```

## ETF Universe (40 ETFs)

| Category | ETFs |
|----------|------|
| US Broad Market | SPY, QQQ, IWM, DIA, VTI |
| US Sectors | XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE |
| International | EWU, EWG, EWQ, EWI, EWP, VGK, EWJ, EWZ, EEM, FXI |
| Fixed Income | TLT, IEF, SHY, LQD, HYG, EMB |
| Commodities | GLD, SLV, USO, DBA, PDBC, IAU |
| Real Estate | VNQ |
| Currency | UUP, TIP |

## Metrics

- **Performance:** YTD, 1M, 3M, 6M, 12M returns
- **Risk:** Annualised volatility, Sharpe ratio, Max drawdown (1Y)
- **RS Score:** Composite momentum rank 0–100 (weighted: 12M×30%, 6M×30%, 3M×25%, 1M×15%)
- **Intermarket:** 30-day rolling correlation vs SPY, TLT, GLD, USO, UUP, EEM, HYG

## Automated Updates

GitHub Actions runs every weekday at **22:00 UTC (00:00 Sofia/EET)**:
1. Fetches latest prices from Yahoo Finance
2. Computes all metrics and RS scores
3. Commits updated `data/etfs.json`
4. Re-deploys GitHub Pages

You can also trigger a manual update from **Actions → Daily ETF Data Update → Run workflow**.

## Local Usage

```bash
pip install -r requirements.txt
python fetch_data.py
# then open index.html in your browser
```

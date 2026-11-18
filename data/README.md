# Data

Every dataset used or produced by the pipeline is stored in this directory, so that all results can be reproduced without downloading any data.

| Folder | Produced by | Content |
|---|---|---|
| `raw/market/` | Step 1 (`src/data_acquisition.py`), Yahoo Finance | Daily OHLCV of SPY, IEF, GLD and SHY, adjusted for splits and dividends (`auto_adjust=True`) |
| `raw/macro/` | Step 1, FRED | Treasury yields (DGS3MO, DGS2, DGS5, DGS10, DGS30), Fed funds rate, CPI and core CPI, 5Y/10Y breakeven inflation, VIX |
| `processed/` | Step 2 (`src/data_transformation.py`) | Aligned daily panel, adjusted-price validation report, weekly decision calendar, weekly panel, in-sample feature snapshot |
| `signals/` | Step 3 (`src/agents/train.py`) and step 5 (`main.py`) | In-sample signals of every agent (`signals_<agent>_<asset>.csv`; Random Forest and SVM are out-of-fold) and out-of-sample Orchestrator signals (`oos_signals_orchestrator.csv`) |
| `results/` | Steps 3 to 6 | Training logs (`training/`), validation grid (`validation/`), target weights, daily NAVs, metrics, Sharpe-ratio test and charts (`plots/`) |

## Provenance and limitations

- Raw files are snapshots. A new download can differ, because Yahoo Finance recomputes adjusted prices after every distribution and FRED revises some series; set `pipeline.download_data: true` only to build a new dataset version.
- FRED series are stored with their observation dates. Monthly series (CPI, core CPI, Fed funds) are shifted by a conventional publication lag (`data.release_lags_days` in `config.yaml`) before use; historical vintages are not reconstructed.
- The common sample starts on 18 November 2004 (first GLD session) and ends on the last session available for every ETF.

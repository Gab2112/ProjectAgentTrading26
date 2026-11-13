# Data Provenance and Frequency

This directory contains the exact input snapshots used in the registered
Core35-Satellite65 research and its retrospective matched execution replay.
All simulation and model-validation commands in this submission operate
offline; no Yahoo Finance, FRED or Bloomberg connection is required.

| Path | Contents | Frequency and intended use |
|---|---|---|
| `raw/market/` | SPY, IEF, GLD, SHY, TLT OHLCV snapshots | Historical daily market data; original prices from the frozen archive |
| `raw/macro/` | CPI, yield, VIX and policy-rate source series | Original observed frequencies; lag/availability handled in code |
| `processed/market_daily.csv` | Date and per-ETF adjusted Open/Close fields | Aligned daily simulation inputs |
| `signals/signals_orchestrator_*.csv` | Development dates, discrete signals and confidence | 2010-2021 weekly decisions; frozen training comparison |
| `../reports/oos_signals/` | Out-of-sample weekly Orchestrator inference | 2022-2026 frozen decisions |
| `../reports/oos_final_2022_2026/` | Original daily NAV, weights, metrics and provenance | Immutable official final evaluation |

The research universe is SPY, IEF, GLD and SHY. TLT data are retained as a
historical Treasury input for audit, not as a traded instrument in the final
four-ETF candidate. Historical FRED publication vintages are not fully
point-in-time certified and remain a documented data limitation.

The authoritative source hashes for the original raw data are in
`../reports/oos_input_audit.json`, and the full ZIP is indexed by
`../PACKAGE_MANIFEST.json`. Do not regenerate or replace historical sources
inside the final package; research exports are provided under
`../research_outputs/`.

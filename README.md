# ProjectAgentTrading26 | Group 9

**LUISS Guido Carli - Data-Driven Models for Investment, 2026-27**

A weekly multi-asset trading system on four ETFs (SPY, IEF, GLD, SHY). Five agents (SMA/EMA trend, Z-score, macro yield curve, Random Forest, SVM) produce signals in {-1, 0, +1}; for each asset an Orchestrator, fitted with a genetic algorithm in a yearly walk-forward, combines them into one signal and a confidence. A portfolio rule allocates capital across the assets with active signals. The full methodology, the exact signal logic and thresholds, and the pitfall analysis are in [`docs/methodology.md`](docs/methodology.md).

## 1. Setup

Python 3.13 (tested with 3.13.5).

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Running the pipeline

From the repository root:

```bash
python main.py
```

`main.py` runs every step in order and reads all settings from `config.yaml`:

| Step | Module | Output |
|---|---|---|
| 1. Data acquisition (skipped by default: `pipeline.download_data: false`) | `src/data_acquisition.py` | `data/raw/` |
| 2. Data validation and preparation | `src/data_transformation.py` | `data/processed/` |
| 3. Agent training on the in-sample period (skipped by default: `pipeline.train_agents: false`; the saved models are reused) | `src/agents/train.py` | `src/agents/config/`, `data/signals/`, `data/results/training/` |
| 4. Selection of the allocation parameters on the validation window | `src/strategy.py` | `data/results/validation/` |
| 5. Backtests, metrics and charts: in-sample final fit (2005-2020), in-sample walk-forward (2013-2020) and, once, out-of-sample | `src/backtest.py`, `src/baselines.py`, `src/evaluator.py`, `src/plots.py` | `data/results/metrics_*.csv`, `nav_*.csv`, `target_weights_*.csv`, `signal_backtests_*.csv`, `data/results/plots/` |

The repository contains the raw data, the processed data, the trained models and every result, so the pipeline runs offline. To retrain every agent from scratch, set `pipeline.train_agents: true` (about six minutes); training is deterministic and reproduces the stored models and results exactly. Random seeds are fixed (`project.random_seed`), all paths are relative to the repository root, and no credential is required.

Unit tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## 3. Design summary

| Item | Choice |
|---|---|
| Data | Yahoo Finance daily OHLCV adjusted for splits and dividends; FRED yields, inflation, breakevens and VIX |
| Sample | Nov 2004 - Oct 2026 |
| In-sample (training) | 20 May 2005 - 31 Dec 2020; walk-forward model selection inside this window; yearly expanding walk-forward of the Orchestrator from 2013 |
| Validation | 2016 - 2020 (last five IS years): allocation parameters only |
| Out-of-sample | 1 Jan 2021 - 9 Oct 2026, evaluated once |
| Decision / execution | signals at the close of the last trading session of each week, orders at the next session's open |
| Costs | no transaction costs (optional in the project specification; `execution.commission_bps` and `execution.slippage_bps` in `config.yaml`); 1% per year borrowing fee on shorts |
| Sizing | equal weight across active assets, at most 35% per asset (selected on validation), cash for the remainder; no leverage |
| Risk-free rate | 0% (cash in the backtest also earns 0%) |
| Primary benchmark | SPY buy-and-hold (broad market); Equal-weight buy-and-hold of the four ETFs as second benchmark |

Allocation parameters selected on the validation window (using the walk-forward Orchestrator signals): `min_confidence = 0.40`, `max_weight_per_asset = 0.35`, `allow_short = false` (validation Sharpe 1.32).

## 4. Results

Daily returns without transaction costs; Sharpe ratio with a 0% risk-free rate. Source: `data/results/metrics_in_sample.csv` and `data/results/metrics_oos.csv`.

**Out-of-sample, 8 Jan 2021 - 9 Oct 2026 (301 weekly decisions)**

| Portfolio | CAGR | Volatility | Sharpe | Max drawdown | Profit factor | Beta to SPY |
|---|---:|---:|---:|---:|---:|---:|
| **Agent Strategy** | **8.29%** | **5.75%** | **1.41** | **-7.47%** | **1.31** | **0.19** |
| SPY Buy & Hold | 14.96% | 16.62% | 0.92 | -24.50% | 1.18 | 1.00 |
| Equal Weight Buy & Hold | 8.65% | 8.96% | 0.97 | -14.70% | 1.19 | 0.36 |
| 60/40 SPY-IEF (weekly) | 8.23% | 10.68% | 0.79 | -21.11% | 1.15 | 0.62 |
| Volatility Control | 7.47% | 7.21% | 1.03 | -13.77% | 1.19 | 0.28 |
| Macro Regime Long/Flat | 4.97% | 4.40% | 1.12 | -5.84% | 1.24 | 0.08 |

**In-sample, final fit, 20 May 2005 - 31 Dec 2020** (the final models applied to the data they were trained on: an optimistic upper bound, reported only to measure overfitting)

| Portfolio | CAGR | Volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Agent Strategy | 13.00% | 5.83% | 2.13 | -9.25% |
| SPY Buy & Hold | 9.80% | 19.77% | 0.57 | -55.19% |
| Equal Weight Buy & Hold | 7.34% | 8.54% | 0.87 | -17.63% |

**In-sample, walk-forward, 4 Jan 2013 - 31 Dec 2020** (first week of the Orchestrator walk-forward; still optimistic, because the rule-based agents and the allocation parameters were selected on this period)

| Portfolio | CAGR | Volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Agent Strategy | 6.31% | 5.15% | 1.21 | -9.25% |
| SPY Buy & Hold | 14.73% | 16.81% | 0.90 | -33.72% |
| Equal Weight Buy & Hold | 6.32% | 6.85% | 0.93 | -14.21% |

**Interpretation.** Out of sample the strategy has a higher Sharpe ratio and a much smaller maximum drawdown than both buy-and-hold benchmarks, with a beta to SPY of 0.20. It earns a little more than half of the return of SPY, because it holds at most 35% per asset and stays largely in cash during the 2022 bond-equity sell-off. The Sharpe-ratio advantage rests on a single OOS path of about six years and is not supported by a formal statistical test, so it is suggestive rather than conclusive. The final-fit in-sample Sharpe ratio (2.13) is far above the walk-forward (1.21) and OOS (1.41) values: the gap measures the overfitting of models evaluated on their own training data, and the walk-forward figure is the in-sample estimate consistent with the OOS result. Per-asset signal backtests (`data/results/signal_backtests_oos.csv`) show that the GLD, SHY and IEF signals beat buy-and-hold of the same asset on a risk-adjusted basis, whereas the SPY signal (Sharpe 0.52) lags SPY buy-and-hold (0.92).

**Cost sensitivity.** With 2 bp commission plus 1 bp slippage on traded notional, the same pipeline selects `min_confidence = 0.50` and yields an OOS Sharpe ratio of 1.29 (CAGR 7.33%, maximum drawdown -5.24%), so the conclusions do not depend on the cost assumption.

Charts: `data/results/plots/equity_oos_benchmarks.png`, `equity_in_sample_benchmarks.png`, `equity_*_all.png`, `signal_backtests_*.png`, `validation_grid.png`.

## 5. Repository structure

```text
project/
├── main.py                     end-to-end pipeline (steps 1-6)
├── config.yaml                 every setting of the project
├── requirements.txt            pinned dependencies (Python 3.13)
├── src/
│   ├── data_acquisition.py     download from Yahoo Finance and FRED
│   ├── data_loader.py          file readers with strict date checks
│   ├── data_transformation.py  data validation and processed datasets
│   ├── features.py             every feature definition
│   ├── macro_availability.py   point-in-time alignment of macro series
│   ├── rate_features.py        interest-rate features for the SVM
│   ├── time_split.py           IS / validation / OOS boundaries, decision calendar
│   ├── agents/                 the five agents, the Orchestrator and the training script
│   ├── strategy.py             allocation rule, signal sources, validation tuning
│   ├── baselines.py            benchmarks and baseline strategies
│   ├── portfolio.py            volatility-control optimiser and volatility cap
│   ├── regime_allocator.py     macro-regime baseline rules
│   ├── backtest.py             backtest interface (next-open execution, costs)
│   ├── execution_engine.py     daily accounting engine (cash, shorts, borrow, margin)
│   ├── evaluator.py            performance metrics
│   └── plots.py                charts
├── data/                       raw, processed, signals and results (see data/README.md)
├── docs/methodology.md         methodology and pitfall analysis
└── tests/                      unit tests
```

## 6. Limitations

FRED data are not point-in-time vintages; execution at the adjusted open without transaction costs ignores commissions, bid-ask spreads and market impact; the Orchestrator applied out of sample is a single fit on 2010-2020 and is not re-estimated during the OOS period; the rule-based agents use parameters selected on the whole IS period, so only the OOS period is fully independent; the Sharpe-ratio difference is not supported by a formal statistical test; one OOS period of about six years gives limited statistical power. See `docs/methodology.md`, section 8.

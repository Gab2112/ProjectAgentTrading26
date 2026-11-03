# ProjectAgentTrading26

**Regime-Aware Multi-Asset Allocation: Dynamic Risk-Adjusted Performance and Downside Protection**

Quantitative investment research for **Data-Driven Models for Investment**.

> **Status, 28 October 2026 (local working tree):** A1, A2 and B are implemented and the A/B modules have been consolidated; 123 unit tests passed locally after the cleanup. **B_v2** is the training-selected Long/Flat candidate; its **2017-2021 validation has been examined** in a separate, unchanged-model diagnostic. **2022+ has not been evaluated for the new Multi-Asset strategy.** Clustering and the ML/agent modules are not yet integrated into the shared allocation pipeline.

## 1. Research question and system

Can progressively adding volatility control, macro-financial regimes and forecasting improve risk-adjusted returns and downside protection over simpler portfolios?

The investable universe comprises **SPY** (US equities), **IEF** (intermediate US Treasuries), **GLD** (gold) and **SHY** (short US Treasuries).

| Level | Method | Status |
|---|---|---|
| **A1** | SPY buy-and-hold; weekly SPY/IEF 60/40; weekly four-ETF equal-weight; equal-weight buy-and-hold | Implemented |
| **A2** | Fully invested, long-only allocation close to equal-weight under an 8% estimated volatility constraint | Implemented |
| **B_v1** | Deterministic macro regimes and Long/Short momentum | Preserved experimental baseline |
| **B_v2** | Regime-conditioned **Long/Flat** momentum, with unused allocation in cash | Frozen training-selected candidate |
| **C** | Forecasting, ML and potentially timestamped news/agents | Not integrated |

**B_v2:** Each week, an ETF is Long only when its **21-day and 63-day momentum are both positive**; otherwise it is Flat. Four regimes use **VIXCLS** and **T10YIE** versus their lagged 52-week median thresholds. Each regime determines SPY/IEF/GLD/SHY budgets, and a **126-session covariance estimate** scales Long/Flat weights down if estimated annual volatility exceeds **8%**. B_v2 has **no short positions** and **no separate macro directional signal filter**. It is *not* market-neutral, and 8% is not a realized-risk guarantee.

B_v1 remains available to reproduce the research path that motivated the simpler B_v2; it is **not** the selected strategy.

## 2. Data, execution and research split

| Segment | Dates | Use |
|---|---|---|
| Warm-up | Before 20 May 2005 | Indicator history only |
| **Training** | 20 May 2005 - 31 Dec 2016 (last session: 30 Dec) | Development and B_v2 selection |
| **Validation** | 1 Jan 2017 - 31 Dec 2021 | B_v2 diagnostic **already inspected**; normal A/B runs remain training-only |
| **Final test** | From 1 Jan 2022 | Reserved for the integrated system |

- Local adjusted daily Open/Close data: `data/raw/market/`. Macro observations: `data/raw/macro/`.
- **Weekly decision after Close; execution at the next available Open.** Positions and NAV are valued daily.
- **USD 100,000** initial capital; **2 bps commission + 1 bp slippage** on traded notional. Uninvested cash earns **0%**; B_v1 also includes simplified short-borrow and margin assumptions.
- Metrics use **252 trading sessions/year** and **zero diagnostic risk-free rate**.
- Historical macro release timing and data vintages are **not certified point-in-time**; adjusted prices and transaction costs are approximations.

## 3. Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -p 'test_*.py' -v

python3 main.py --level ALL                 # A1 + A2
python3 main.py --level B --b-version v2    # Selected B strategy
python3 main.py --level B --b-version v1    # Historical B baseline
```

`python3 main.py --level B` defaults to **v1** for backward compatibility. Normal execution is **training-only** and does not download new data by default. Keep the original raw CSVs for exact reproduction.

The previously performed B_v2 validation diagnostic is reproducible, provided ETF and macro data cover 2021:

```bash
python3 tests/holdout_diagnostic_bv2.py --through 2021-12-31
```

This manual diagnostic is excluded from `test_*.py` discovery and accepts **only** `2021-12-31`; it does not open the 2022+ final test. Its validation results have already been inspected, so it should be rerun only for explicit reproducibility checks. It requires the original frozen B artifacts and adequate historical data coverage.

## 4. Observed results

**Training (2005-2016)**; after modeled transaction costs:

| Strategy | CAGR | Annual volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| A1 - SPY buy-and-hold | 7.77% | 19.68% | 0.479 | -55.19% |
| A1 - weekly 60/40 | 7.13% | 10.88% | 0.688 | -32.00% |
| A1 - weekly equal-weight | 6.54% | 6.99% | 0.942 | -15.69% |
| A1 - equal-weight buy-and-hold | 6.16% | 8.75% | 0.727 | -17.63% |
| **A2 - volatility control** | **6.24%** | **6.41%** | **0.977** | **-14.28%** |
| B_v1 - Long/Short | 0.24% | 5.55% | 0.071 | -17.55% |
| **B_v2 - Long/Flat** | **2.64%** | **4.17%** | **0.646** | **-7.01%** |

**B_v2 validation (2017-2021), one subsequent diagnostic:** **CAGR 1.72%**, **annual volatility 3.37%**, **Sharpe 0.524**, **max drawdown -4.70%**, **cumulative return +8.92%**. Results are normalized to USD 100,000 at validation start for presentation, not a second initial-capital simulation. B_v2's rules were unchanged after inspection.

B_v2's drawdown is lower than A2's on training, but **its return and Sharpe are also lower**. Average target exposure is about **54%**, with cash earning zero; a fair incremental-value claim requires comparisons at comparable risk/exposure. Regime budgets improved B_v2 training Sharpe by only **0.037** against fixed average budgets. The validation result is not proof of alpha or of fully leakage-free historical inputs.

## 5. Architecture and next steps

- **Data and timing:** `src/data_acquisition.py`, `src/data_transformation.py` (including the data-loader API), `src/time_split.py`, and `src/macro_availability.py` (macro features and deterministic regime classification).
- **A1/A2:** `src/portfolio.py` (including the CVXPY A2 optimizer), `src/backtester.py`, and `src/evaluator.py`.
- **B:** `src/regime_allocator.py` (regime-label validation, explicit cluster-to-budget contract, signal construction and volatility caps), `src/regime_strategy.py` (B_v1/B_v2 composition), and `src/longshort_backtester.py` (the shared daily self-financing simulation, including Long/Flat as a special case).
- **Legacy agents:** `src/agents/` retains the earlier SHY/IEF/TLT Treasury research components. They are not yet integrated into the current SPY/IEF/GLD/SHY A/B pipeline. Previously generated `signals_statistical_{shy,ief,tlt}.csv` files are not part of the tracked A/B source; the agent code can generate statistical signals when suitable source features are available.
- **Diagnostics:** `tests/holdout_diagnostic_bv2.py` is a manual validation-only script, not an automated unit test. `tests/test_*.py` contains the regression/invariant tests.
- Generated outputs under `data/processed/`, `data/signals/` and `data/results/` are Git-ignored. The frozen B output directories are `data/results/B/regime_longshort/` and `data/results/B/regime_longflat_v2/`.
- Details and research decisions: [`docs/research_design.md`](docs/research_design.md) and [`docs/decision_log.md`](docs/decision_log.md); earlier Treasury-only research: [`docs/treasury_research_review.md`](docs/treasury_research_review.md).

**Next:** integrate the clustering regime provider, then assess forecasting/ML and agent contributions using the same causal execution conventions. Keep the final 2022+ test for the frozen integrated system, and resolve macro point-in-time availability before interpreting results as investable performance.

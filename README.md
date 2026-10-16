# LUISS Final Project Work

Systematic trading research project for the LUISS course **Data-Driven Models for Investment**.

## Current research direction

The current working hypothesis is a **Macro / Rates Regimes** framework:

> Can combinations of inflation, interest rates, yield-curve shape and rate volatility identify recurring macro/rates regimes associated with different fixed-income exposures?

The current implementation is an interpretable **economic baseline**, not the final strategy.

## Traded universe

The current chosen assets strategy uses three U.S. Treasury ETFs:

- **SHY**: short-duration Treasuries;
- **IEF**: intermediate-duration Treasuries;
- **TLT**: long-duration Treasuries.

The strategy operates at **weekly frequency**.

## Current pipeline

```text
        RAW DATA
            |
            v
 DATA TRANSFORMATION
            |
            v
  FEATURE ENGINEERING
            |
            v
      REGIME MODEL
            |
            v
   SIGNAL GENERATION
            |
            v
        PORTFOLIO
            |
            v
        BACKTEST
            |
            v
       EVALUATION
```

The entire pipeline can be executed from the repository root with:

```bash
python3 main.py
```

By default, the pipeline uses the raw CSV files already stored in the repository and does **not** download data again.

## Repository structure

```text
ProjectAgentTrading26/
|
|-- README.md
|-- requirements.txt
|-- config.yaml
|-- main.py
|
|-- docs/
|   |-- research_design.md
|   `-- decision_log.md
|
|-- data/
|   |-- raw/
|   |   |-- market/
|   |   `-- macro/
|   |-- processed/
|   |-- signals/
|   `-- results/
|
`-- src/
    |-- __init__.py
    |-- data_acquisition.py
    |-- data_loader.py
    |-- data_transformation.py
    |-- feature_engineering.py
    |-- regime_model.py
    |-- forecasting.py
    |-- signal_generation.py
    |-- portfolio.py
    |-- backtester.py
    |-- evaluator.py
    |-- utils.py
    `-- agents/
        `-- __init__.py
```

## Environment

The project has been tested with **Python 3.14.0**.
Exact library versions are pinned in `requirements.txt`.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Quick syntax check:

```bash
python3 -m py_compile main.py src/*.py
```

## Data

### Market data

Daily adjusted data are obtained from Yahoo Finance for SHY, IEF and TLT.

### Macro / rates data

Current FRED inputs include:

- Federal Funds Rate;
- 3M, 2Y, 5Y, 10Y and 30Y Treasury yields;
- CPI;
- Core CPI;
- 5Y and 10Y breakeven inflation.

Raw data are stored in `data/raw/` so the pipeline remains reproducible without requiring a new download.

## Data transformation

The pipeline:

- cleans and sorts raw observations;
- converts market and rates data to weekly frequency;
- aligns all series on a common calendar;
- applies conservative publication lags to low-frequency macro variables;
- preserves the last known macro observation until the next release.

Current provisional release-lag assumptions:

- CPI: 45 calendar days;
- Core CPI: 45 calendar days;
- FEDFUNDS: 35 calendar days.

These lags reduce publication-timing look-ahead bias but do **not** fully solve historical revision leakage.

A future improvement is to use **ALFRED / point-in-time vintage macro data** or exact historical release calendars.

## Feature engineering

The baseline currently includes:

- weekly ETF returns;
- weekly changes in Treasury yields;
- yield-curve spreads: 10Y-2Y, 10Y-3M, 30Y-10Y;
- weekly changes in those spreads;
- CPI YoY and Core CPI YoY computed from the original monthly series;
- weekly changes in 5Y and 10Y breakeven inflation;
- a 13-week realized rate-volatility proxy based on 10Y Treasury yield changes.

No future target variables are created in the feature-engineering stage.

## Deterministic regime baseline

The current regime model classifies four dimensions:

```text
Inflation   -> rising / falling
Rates       -> rising / falling
Curve       -> normal / inverted
Volatility  -> high / low
```

The implementation produces all **16 possible combined regimes**.

## Signal generation

The duration score currently uses:

- falling rates: positive contribution;
- rising rates: negative contribution;
- falling inflation: positive contribution;
- rising inflation: negative contribution;
- inverted curve: mild positive contribution;
- high volatility: reduced conviction.

Signal mapping:

```text
High duration score     -> TLT long
Intermediate score      -> IEF long
Low duration score      -> SHY long + TLT short
```

All final asset signals belong to:

```text
+1 = long
 0 = flat
-1 = short
```

## Portfolio construction

The current allocation method is `equal_active_signal`.

Examples:

```text
TLT signal = +1
-> TLT weight = +1.00

IEF signal = +1
-> IEF weight = +1.00

SHY signal = +1 and TLT signal = -1
-> SHY weight = +0.50
-> TLT weight = -0.50
```

The baseline uses gross exposure capped at 1.0 and no additional leverage.

## Backtest timing

Target weights produced at week `t` are applied only from the next period:

```text
Information available at t
            |
            v
Target weights at t
            |
            v
Applied weights at t+1
            |
            v
Realized return at t+1
```

This `shift(1)` rule prevents look-ahead bias.

Transaction costs are applied to portfolio turnover: 2 bps commission + 1 bps slippage per unit of turnover (`execution.commission_bps` and `execution.slippage_bps` in `config.yaml`).

## Evaluation framework

Current chronological split:

```text
TRAIN
2005-05-27 -> 2016-12-30

VALIDATION
2017-01-06 -> 2021-12-31

OOS
2022-01-01 -> 2026
LOCKED
```

The OOS period is currently disabled through:

```yaml
evaluation:
  evaluate_oos: false
```

It must remain locked until the model, features, thresholds and relevant hyperparameters are frozen.

While the OOS is locked, the backtest itself stops at `validation_end`: neither `backtest_weekly.csv` nor the final equity printed by `main.py` contain OOS returns.

Current evaluation metrics include cumulative return, annualized return, annualized volatility, Sharpe ratio, maximum drawdown and profit factor.

The 3M Treasury yield is currently used as a proxy for the risk-free rate.

The main benchmark is a true equal-weight buy-and-hold portfolio of SHY, IEF and TLT, together with individual ETF buy-and-hold benchmarks.

Note: `backtest_weekly.csv` also contains a `benchmark_equal_weight_rebalanced_*` series (weekly rebalanced to 1/3 each). It is kept only as a reference and is not the benchmark used in `evaluation_metrics.csv`.

## Current status

Completed:

- raw-data acquisition;
- data transformation;
- feature engineering;
- deterministic regime model;
- signal generation;
- portfolio construction;
- backtesting;
- train/validation evaluation;
- one-command end-to-end execution through `main.py`.

The deterministic baseline currently underperforms the fixed-income benchmark and should be treated as an **economic reference model**, not as the final strategy.

Next research stage:

1. define a forecasting / classification target;
2. implement a simple statistical model;
3. compare it against the deterministic baseline;
4. introduce nonlinear ML only if justified;
5. consider an agentic layer only after the quantitative core is stable.

## Agents

The agent layer is planned but currently disabled.

Potential future roles include Orchestrator, Macro Agent, Regime Agent, Strategy Agent, Risk Agent and Critic Agent.

Agents should complement the quantitative pipeline, not replace deterministic calculations.

## Documentation

See:

- `docs/research_design.md` for methodology and research architecture;
- `docs/decision_log.md` for important design decisions and unresolved methodological issues.

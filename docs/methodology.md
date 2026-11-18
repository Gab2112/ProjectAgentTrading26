# Methodology

This document describes the research design of ProjectAgentTrading26: the hypothesis, the data, the signal-generation logic, the portfolio and execution rules, the evaluation protocol and the treatment of the main backtesting pitfalls. All numerical settings quoted below are defined in `config.yaml`.

## 1. Hypothesis

Trend, mean-reversion, macroeconomic and machine-learning signals capture different and partly independent sources of predictability in liquid ETFs. An ensemble that weights these agents, and calibrates how much it trusts each of them, should deliver a better risk-adjusted return than any single rule and than passive exposure to the same asset classes, mainly by reducing drawdowns.

## 2. Data

| Item | Description |
|---|---|
| Universe | SPY (US equities), IEF (7-10Y Treasuries), GLD (gold), SHY (1-3Y Treasuries) |
| Market data | Yahoo Finance daily OHLCV, adjusted for splits and dividends (`auto_adjust=True`) |
| Macro data | FRED: Treasury yields 3M/2Y/5Y/10Y/30Y, Fed funds, CPI and core CPI, 5Y/10Y breakeven inflation, VIX |
| Sample | 18 Nov 2004 (first common session) to the last stored session; weekly decisions |
| Frequency | Signals are weekly; execution and accounting are daily |

Validation of the data (`src/data_transformation.py`, `src/data_loader.py`):

- the four ETFs are aligned on their common sessions; a session missing for one ETF is dropped for all of them, and the alignment fails if more than 1% of sessions are inconsistent;
- prices must be finite and positive, dates sorted and unique;
- `validate_adjusted_prices` rejects any daily close-to-close or close-to-open move larger than 25%: an unadjusted split would appear as such a jump. The largest move in the sample is reported in `data/processed/price_validation.csv`;
- monthly macro series are shifted by their conventional publication lag (45 days for CPI, 35 days for the Fed funds rate) before being aligned with backward `merge_asof`, so a value is used only after it would have been published.

## 3. Features

All features are backward-looking and computed on the decision date (`src/features.py`):

- per ETF: adjusted close, one-week return, 21- and 63-session momentum;
- macro: 10Y-2Y, 10Y-3M and 30Y-10Y slopes and their weekly changes, weekly changes of the 10Y, 2Y and 3M yields, CPI year-on-year inflation (lagged), VIX level and four-week change, breakeven inflation;
- SVM only: 1/4/13/26/52-week momentum, 13-week volatility, distance from the 52-week high, 13-week momentum of the other ETFs, and 13-week rate changes and volatility (lagged by one week).

## 4. Agents and signal logic

Every agent outputs, for every asset and week, a signal in {-1, 0, +1} and a confidence in [0, 1].

| Agent | Rule | Parameters and how they are chosen |
|---|---|---|
| SMA/EMA | +1 if EMA(fast) > SMA(slow) on weekly closes, -1 otherwise; 0 during the indicator warm-up | SMA in 20..100, EMA in 5..50 weeks; walk-forward (5-year train, 1-year test, rolled yearly); the final pair is the most frequent winner across windows |
| Z-score | z = (price - rolling mean) / rolling std; momentum mode: enter +1 when z > threshold and -1 when z < -threshold; mean-reversion mode: enter +1 when z < -threshold and -1 when z > threshold; a position is held until z crosses back through 0, then the signal is 0 | window, threshold and mode chosen by the same walk-forward |
| Macro yield curve | score = 0.50 x rate component + 0.35 x curve-regime component + 0.15 x inflation component + inversion bonus; +1 / -1 beyond +/-0.5 with an asset-specific mapping | fixed economic rules, no fitting |
| Random Forest | classifier of next-week direction on macro and price features; +1 if P(up) >= threshold, -1 if P(up) <= 1 - threshold, 0 otherwise | depth, minimum leaf size, threshold and number of trees by walk-forward; predictions used downstream are out-of-fold |
| SVM | classifier of the 4-week direction; standardised decision score z; +1 if z > 0.25, -1 if z < -0.25, 0 otherwise | kernel, C and gamma chosen each year by time-series cross-validation (gap = 4 weeks); out-of-fold predictions, refit yearly |

**Orchestrator.** For each asset the five agents are combined as

    Z_t = sum_i w_i s_i c_i^alpha_i / sum_i w_i c_i^alpha_i

where s_i and c_i are the signal and confidence of agent i, w_i its weight (non-negative, summing to one) and alpha_i a calibration exponent. The output signal is +1 if Z_t > theta_long, -1 if Z_t < -theta_short and 0 otherwise; the output confidence measures the distance of Z_t from the threshold that was crossed. The 2N + 2 parameters are fitted by a real-coded genetic algorithm (population 100, 40 generations, tournament selection, BLX-alpha crossover, Gaussian mutation with decaying variance, elitism, fixed seed) maximising Sharpe x (1 - MDD) of the weekly signal returns net of 10 bp per unit of turnover, minus a concentration penalty on the weights, with a minimum-activity constraint. The GA uses only the in-sample weeks on which both machine-learning agents have out-of-fold predictions (from 2010).

**Walk-forward of the Orchestrator.** For each test year Y from 2013 to 2020 the GA is refitted on all pool weeks before 1 January of Y (expanding window, at least `agents.orchestrator.walk_forward_min_train_years` = 3 years) and applied, unchanged, to the weeks of Y. The concatenated test-year outputs are the in-sample Orchestrator signals (`data/signals/signals_orchestrator_<asset>.csv`) used by the validation step and by the in-sample evaluation; they are out-of-sample with respect to the Orchestrator parameters. Per-fold parameters and train/test performance are saved in `data/results/training/orchestrator_walk_forward_<asset>.csv`. A final fit on the whole pool (2010-2020) is the model applied to the out-of-sample period.

## 5. Portfolio construction and execution

**Allocation rule.** At each weekly decision an asset is active long when its Orchestrator signal is +1 with confidence at least `min_confidence`, and active short when the signal is -1 with confidence at least `min_confidence` and shorting is allowed. SPY is held long by default unless a confirmed short signal is present (it is then short, if shorting is allowed, or flat). Each active asset receives min(1 / number of active assets, `max_weight_per_asset`), with the sign of its direction; the remainder is cash at 0%. A new target replaces the previous one only if a weight changes by at least 5 percentage points.

**Selection of the allocation parameters.** `min_confidence`, `max_weight_per_asset` and `allow_short` are selected by grid search on the validation window 2016-2020, the last five years of the in-sample period, by maximising the Sharpe ratio of the full daily backtest built on the walk-forward Orchestrator signals. The grid and the selected values are saved in `data/results/validation/`.

**Execution timing.** Decisions use only information available at the close of the last trading session of each week (Friday, or Thursday in a holiday week). Orders are executed at the open of the next trading session. The adjusted open is used as the execution price.

**Costs and sizing.** Transaction costs are optional in the project specification and are set to zero (`execution.commission_bps` and `execution.slippage_bps` in `config.yaml`; the engine charges them on traded notional when they are positive). With 2 bp commission plus 1 bp slippage the pipeline selects `min_confidence = 0.50` and the OOS Sharpe ratio is 1.29 instead of 1.41, so the conclusions do not depend on this assumption. Short positions pay 1% per year (ACT/365) and require collateral equal to the short value plus 50%. Weights are fractions of NAV, gross exposure is at most 100% (no leverage), and the initial capital is USD 100,000. The NAV is marked to market at every close.

**Per-asset signal test.** As required by the project specification, each Orchestrator signal is also backtested on its own asset with weight +1 (long), 0 (flat) or -1 (short), with the same execution rules.

## 6. Evaluation protocol

| Period | Dates | Use |
|---|---|---|
| Warm-up | Nov 2004 - May 2005 | indicators only |
| In-sample (IS) | 20 May 2005 - 31 Dec 2020 | training of every agent (walk-forward), Orchestrator walk-forward and final fit, validation |
| Validation | 1 Jan 2016 - 31 Dec 2020 (inside IS) | selection of the allocation parameters only |
| Out-of-sample (OOS) | 1 Jan 2021 - end of data | evaluated once, with every parameter fixed |

The IS period is evaluated twice. The *final fit* applies the final models to the whole IS period (May 2005 - December 2020), i.e. to the data they were trained on; it is an optimistic upper bound, reported only to measure overfitting. The *walk-forward* evaluation starts in January 2013, the first week of the Orchestrator walk-forward (the earlier IS years are consumed by the first training windows), and uses Orchestrator signals not seen by the GA; it remains somewhat optimistic, because the rule-based agents and the allocation parameters were selected on IS data.

**Benchmarks.** SPY buy-and-hold is the primary benchmark: it is the broad US equity market, the reference for market exposure, and one of the traded assets. Equal-weight buy-and-hold of the four ETFs represents a passive holder of the same universe. Weekly-rebalanced 60/40 and equal-weight portfolios, a volatility-controlled portfolio and two macro-regime rules are reported as simple active baselines.

**Metrics** (daily returns, 252 sessions per year, risk-free rate 0%): cumulative return, CAGR, annualised volatility, Sharpe and Sortino ratios, maximum drawdown, Calmar ratio, profit factor, hit rate, turnover, costs, beta and correlation to SPY, and the mean return on days when SPY falls. Equity curves start at 100 and are plotted with the benchmarks.

## 7. Pitfall analysis

| Pitfall | Treatment |
|---|---|
| Overfitting | Hyperparameters chosen by walk-forward (rule-based agents, Random Forest) or time-series cross-validation (SVM); the Orchestrator is estimated in a yearly expanding walk-forward; small grids; the GA objective penalises drawdown, weight concentration and inactivity and includes turnover costs; the allocation layer has only three parameters, chosen on a separate validation window; the OOS period is used once. The walk-forward logs in `data/results/training/` report the in-sample and test performance of every window, and windows with a positive in-sample but negative test Sharpe are flagged. The final-fit IS results are reported next to the walk-forward ones to measure the overfitting gap. |
| Look-ahead bias | Decisions at the weekly close, execution at the next open; every feature is backward-looking; monthly macro data are lagged by their publication delay; the SVM labels are purged by the forecast horizon; machine-learning predictions used to fit the Orchestrator are out-of-fold, and the Orchestrator signals used for validation are walk-forward; training code truncates its inputs at `train_end`; unit tests check that changing future prices does not change past features, weights or NAV. |
| Survivorship bias | The universe is fixed ex ante and consists of four broad, continuously listed ETFs (SPY since 1993, IEF and SHY since 2002, GLD since 2004) that represent asset classes rather than selected winners. The sample starts when all four exist, so no instrument is added or removed with hindsight. |
| Data snooping | One configuration file fixes every choice; the OOS period is not read by any selection step (`tune_allocation` uses the validation window only, and a unit test checks that altering OOS data leaves the selection unchanged). The number of configurations tried is recorded (grid-search logs and validation grid). The OOS period is evaluated once, against two pre-declared benchmarks. |
| Pretrained-model leakage | No pretrained model is used: every model is trained from scratch on the in-sample data, and the saved models record their training period (`training_provenance`), which is checked before inference. A sentiment agent based on a pretrained language model (FinBERT) is not part of the system, so no model whose pre-training data may overlap the test period enters the signals. |

## 8. Limitations

- FRED series are not point-in-time vintages; publication lags are conventional approximations.
- Execution at the adjusted open without transaction costs ignores commissions, bid-ask spreads, market impact and the difference between adjusted and executable prices.
- The Orchestrator applied to the OOS period is a single fit on 2010-2020; it is not re-estimated during the OOS period.
- The rule-based agents use parameters selected over the whole IS period, so their IS signals (and hence the validation score) are not fully out-of-sample; only the OOS period is fully independent.
- The Sharpe-ratio difference is not supported by a formal statistical test.
- With four assets and one OOS period of about six years, statistical power is limited.

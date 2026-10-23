# Treasury Regime Trading: Research Review and Phase Closure

**Project:** ProjectAgentTrading26 | LUISS, *Data-Driven Models for Investment*\
**Research phase:** 8-23 October 2026\
**Status:** Treasury research paused; change of market/universe **under consideration**, not yet agreed by the full team.\
**Version:** Research archive; not a final out-of-sample performance report.

## Executive summary

The team developed a reproducible systematic trading research pipeline for three U.S. Treasury ETFs: SHY (short duration), IEF (intermediate duration), and TLT (long duration). The research hypothesis was that macroeconomic/rates regimes, and later predictive models, could improve the allocation of duration exposure relative to passive benchmarks.

The implementation works end to end and its core temporal and portfolio invariants are covered by 31 automated tests. However, **the tested rules did not demonstrate robust economic outperformance of passive alternatives**. The original four-component deterministic duration score underperformed benchmarks. A simple Ridge model did not improve over an expanding historical-mean forecast on an internal train-only walk-forward test. A simple 30-year-yield trend rule performed well during the train diagnostic window but failed to sustain that advantage during 2017-2021 validation.

**Decision:** Pause further Treasury-specific parameter searches and modelling. Preserve the code, data pipeline, experiments, and results as a research baseline; evaluate alternative markets and investable universes jointly before changing the shared branch. No 2022+ OOS performance was consulted.

## 1. Research question and economic motivation

**Initial question:** Can changes in inflation, Treasury yields, the yield-curve shape, and interest-rate volatility identify market regimes that help rotate between short-, intermediate-, and long-duration Treasury ETF exposure?

Economic basis:

- Bond total returns depend on carry/income, changing yields, duration, convexity, and price dynamics. Higher yield sensitivity is not synonymous with higher expected risk-adjusted return.
- The Treasury yield curve can carry information about rates expectations and risk premia, but a macro indicator associated with long-horizon bond returns need not forecast **next week's** ETF relative return.
- Short-term returns may be dominated by news and shocks not captured by monthly macroeconomic data.

The project's first operational goal was to test this premise on three liquid ETF proxies, not to claim that U.S. bond markets are predictably inefficient.

## 2. Scope and data

| Item | Implementation |
| --- | --- |
| Universe | SHY, IEF, TLT |
| Trading/portfolio frequency | Weekly |
| Market data | Daily adjusted ETF OHLCV from Yahoo Finance, aggregated to weeks ending Friday |
| Macroeconomic data | FRED Treasury yields, CPI/Core CPI, FEDFUNDS, inflation breakevens |
| Feature families | ETF returns; Treasury yield levels and changes; yield-curve slopes and changes; inflation YoY; realized rates volatility |
| Baseline rate trend proxy | 2-year Treasury yield, backward 13-week change |
| Other regime components | CPI YoY trend; 10Y-2Y inversion; 13-week 10Y yield-change volatility relative to expanding historical median |
| Signal-to-position | Long TLT / long IEF / 50% long SHY + 50% short TLT |
| Original score thresholds | +2 and -2 |
| Execution approximation | Signal computed from week-t information; target weights shifted one weekly period; adjusted-close-to-adjusted-close returns |
| Turnover cost assumption | 2 bps commission + 1 bp slippage per unit of turnover |
| Temporal protection | `evaluation.evaluate_oos: false`; data and model development stop by 2021-12-31 |

The combined regime model has 16 theoretical binary combinations; 13 distinct states were observed in the available development sample. This is a **rule-based taxonomy**, not a statistically identified collection of latent economic states.

### Data partitions

- **Development dataset:** 940 weekly rows, 2004-01-02 to 2021-12-31, before excluding initial feature warm-up periods.
- **Active strategy train evaluation:** 606 weeks, 2005-05-27 to 2016-12-30.
- **Strategy validation evaluation:** 261 weeks, 2017-01-06 to 2021-12-31.
- **Final OOS period:** starts 2022-01-01, never activated in this phase.

These counts are different because the preprocessing and indicators have warm-up requirements. Some standalone diagnostics start in 2008; compare figures **only over matching sample windows**.

## 3. Architecture and technical verification

The pipeline comprises data acquisition/loading, transformation, feature engineering, regime model, signal generation, portfolio weights, backtest, and evaluation. It is runnable with `python3 main.py` when local prerequisites and raw datasets are present.

As reported after the 21 October stabilization:

- **31/31** automated unit tests passed (15 temporal tests; 16 pipeline/portfolio/execution/data tests).
- The pipeline completed its seven configured execution stages.
- Baseline end-of-development equity remained **$99,528.20** from $100,000 initial capital; total turnover **303** under the 3-bps convention.
- An audit of 13 raw CSVs found no invalid or duplicate dates; the normal loading policy is `strict=True`.
- A sample audit confirmed the configured monthly release lags (45 calendar days for CPI/Core CPI, 35 for FEDFUNDS) and expected year-over-year calculations.
- Checks confirmed `.shift(1)` in portfolio weights and internal consistency of close-to-close returns, gross/net returns, and stated costs.

**Boundaries of these checks:** Passing code tests does not prove the economic assumptions or eliminate every potential form of information leakage.

## 4. Experiments and results

### 4.1 Original deterministic baseline and passive comparators

All percentages in this table are **cumulative returns in their respective evaluation windows**, not CAGR.

| Strategy | Train 2005-2016 | Validation 2017-2021 |
| --- | ---: | ---: |
| Deterministic baseline: 2Y trend, original +/-2 thresholds | -3.11% | +2.72% |
| Exploratory threshold variant: +/-3 | +34.55% | +5.32% |
| SHY buy-and-hold | +27.94% | +7.58% |
| IEF buy-and-hold | +72.03% | +18.98% |
| TLT buy-and-hold | +93.50% | +38.18% |
| Equal-weight true buy-and-hold | +64.49% | +21.58% |

Changing thresholds from +/-2 to +/-3 reduced switching and improved observed backtest returns, but did **not** beat passive alternatives in validation. That modification was made after inspecting results and must not be treated as a clean independently tested selection.

### 4.2 Rate-state proxy: 2Y vs 10Y vs 30Y

With the rest of the original score and allocation unchanged, the **train-only** comparisons were:

| Yield used for the 13-week rate state | Cumulative return | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| 2Y | -3.11% | -0.102 | -36.68% |
| 10Y | +19.20% | 0.084 | -25.21% |
| 30Y | +46.96% | 0.267 | -19.89% |

Because TLT has greater sensitivity to long-term rates than short-term rates, the 30Y proxy has an intelligible economic rationale. However, it was also chosen after comparisons on train data: the gains do not establish independent forecasting power.

### 4.3 Defensive allocation alternatives under the 30Y rate state

**Train only**, all other deterministic rules preserved:

| Bearish duration configuration | Cumulative return | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| A: +50% SHY, -50% TLT | +46.96% | 0.267 | -19.89% |
| B: +100% SHY | +58.20% | 0.340 | -19.89% |
| C: -100% TLT | +35.10% | 0.184 | -22.22% |

These are **not equal-risk portfolios**. Gross nominal exposure does not equate duration exposure, and the short ETF leg adds lending, financing, collateral, and implementation assumptions that the backtester does not fully model. The long-only candidate B was retained as an experiment, never promoted to the official baseline.

### 4.4 Score diagnostic: next-week TLT minus SHY

Train-only diagnostic, using the original +/-2 score groups:

| Regime-score class | 2Y proxy: mean future TLT-SHY | 30Y proxy: mean future TLT-SHY |
| --- | ---: | ---: |
| Defensive | +14.27 bps | +1.19 bps |
| Intermediate | +18.68 bps | +13.83 bps |
| Long duration | -16.25 bps | +6.32 bps |

The 2Y score exhibited a **directional inconsistency** with the intended long-duration ranking over this train sample. The 30Y version was more plausible for the rate component but did not generate a clean ordering across all score classes. These are conditional means, not statistically validated causal effects or proof of an executable trading edge.

### 4.5 Ridge walk-forward forecasting: next-week TLT minus SHY

**Train only:** 619 complete feature/target rows, initial 156-row fit, 463 expanding walk-forward predictions (2008-02-15 to 2016-12-23). Predictors: 13-week 30Y yield change, 10Y-2Y spread, CPI YoY, 13-week 10Y rate volatility. Pipeline: within-fit standardization + Ridge regression (`alpha=10`, fixed ex ante). Comparator: expanding mean of previously observed target values.

| Metric | Ridge | Expanding historical mean |
| --- | ---: | ---: |
| RMSE, bps | 200.360 | 199.103 |
| MAE, bps | 154.316 | 151.931 |
| Directional accuracy | 50.54% | 54.43% |

**Relative predictive R2:** -0.01267, meaning Ridge's mean squared error was about 1.27% higher than the expanding-mean benchmark. No validation or 2022+ OOS observations were used in the Ridge experiment. This is a negative result for **this specification and horizon**, not a proof that bond returns cannot be predicted.

The experimental Ridge script `experiments/forecast_ridge_walkforward.py` was run locally; its presence in the remote `gianluca` branch was **not confirmed as of this archive draft**. Preserve its local source before any checkout, reset, clean, or branch changes.

### 4.6 Simple 30Y momentum: discrete long-only timing

Exploratory rule: if the 30Y yield has declined over the previous 13 weeks, allocate 100% to TLT; otherwise allocate 100% to SHY. Weekly signal decisions are applied to subsequent adjusted-close returns using the existing approximate timing convention; assumed costs are 3 bps per turnover unit.

| Period | Strategy cumulative return | TLT buy-and-hold | Strategy annualized volatility | Strategy max drawdown | Strategy turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train diagnostic, 2008-02-15 to 2016-12-23 (463 weeks) | +91.80% | +73.19% | 11.51% | -16.47% | 101 |
| Validation, 2017-01-06 to 2021-12-31 (261 weeks) | +7.15% | +38.18% | 10.95% | -13.76% | 94 |

Within the train diagnostic, the rule improved return, volatility, and drawdown relative to passive TLT over that **matched interval**. In validation, it underperformed passive SHY (+7.58%), IEF (+18.98%), TLT (+38.18%), and equal-weight buy-and-hold (+21.58%). Annualized volatility of equal-weight buy-and-hold in validation was 6.54%, versus 10.95% for the momentum strategy.

The chosen momentum rule was identified after other train explorations. The 2017-2021 validation period was also inspected in earlier development and is not an untouched independent test of all subsequent decisions. **No formal statistical-significance or full realistic execution analysis has been performed.**

## 5. Methodological limitations

1. **Predictive horizon mismatch.** Slow-moving macro signals may be poorly aligned with a one-week return target. This was hypothesized, not exhaustively tested.
2. **Feature construction and discretionary scores.** The score weights (+/-2 for yield direction, +/-1 for CPI trend and curve bonus) are economically motivated heuristics, not estimated out of sample. An inverted 10Y-2Y curve does not mechanically imply next-week gains in long-duration bonds.
3. **Risk asymmetry.** Equal gross dollar weights across SHY, IEF, and TLT imply materially different interest-rate/duration exposures.
4. **Trading implementation.** Friday adjusted-close-to-Friday adjusted-close returns with `shift(1)` approximate an execution price at the information cutoff. A trade placed after seeing Friday's close cannot be guaranteed at that same close. Monday-open execution has not been modelled.
5. **Short-position realism.** ETF shares can in principle be shorted subject to borrowing and broker conditions, but the model omits explicit borrow charges, financing/collateral treatment, and some practical rebalancing effects.
6. **Revised macro history.** Conservative CPI/FEDFUNDS release lags do not guarantee historical point-in-time/vintage correctness. The input is not a complete ALFRED vintage reconstruction.
7. **Selection and repeated testing.** Comparison of proxy maturities, position rules, thresholds, and post-hoc diagnostics creates model-selection risk; validation was consulted more than once. The final 2022+ holdout remains locked.
8. **Benchmark and sample alignment.** Train subwindows and validation spans differ in length; comparisons require matched dates, consistent costing, and risk-aware metrics. Strong cumulative returns alone are insufficient evidence of alpha.

## 6. Overall conclusions

- **Technical conclusion:** The pipeline and its main invariants are stable and largely reusable. Research execution and historical audit methods improved materially during the project.
- **Economic conclusion:** No evaluated Treasury strategy demonstrated a convincing and sufficiently stable net-of-cost advantage over simple passive portfolios across development periods. The apparent 30Y trend advantage was not sustained in validation.
- **Forecasting conclusion:** A small, interpretable, weekly Ridge specification failed to outperform a naive expanding-mean predictor on internal train walk-forward data. More complex algorithms are not warranted solely by this negative result.
- **Scope conclusion:** The findings **do not** imply that the bond market is untradeable or that all bond forecasting is impossible. They justify pausing this specific approach under the project's time and collaboration constraints.

## 7. Decision, preservation, and handoff

**Research status: PAUSED / ARCHIVED AS A CANDIDATE, NOT DELETED.** The team has not yet finalized a replacement market, asset universe, or project thesis. Discussion candidates include a 15-20-stock cross-sectional equity portfolio, U.S. sector ETFs, or a small liquid FX universe. Each would require its own historical universe construction, data-availability audit, trading/cost assumptions, and benchmarks. None is automatically easier to outperform.

**Preserve before pivoting:**

- `main` as the stable historical Treasury baseline; do not overwrite it with speculative changes.
- Current personal and teammate branches and existing stashes until their contents are accounted for.
- `docs/decision_log.md`, this archive, raw datasets, test modules, and code for experimental exposure comparisons.
- The **local** Ridge experiment script and its console results; the script was not found in the remote branch when this report was drafted.
- The 2022+ OOS lock. Do not run a final Treasury OOS merely to decide whether to abandon the market.

**Proposed next team meeting:** Choose the investment problem before developing signals. For each candidate market, establish (a) a clear economic hypothesis, (b) a point-in-time feasible asset universe and data sources, (c) realistic implementability, (d) simple passive and technical benchmarks, and (e) complementary roles for forecasting, clustering, and agents. Record a single agreed decision before reworking the pipeline.

## 8. Selected literature for interpretation (not replication claims)

- Campbell, J. Y., and R. J. Shiller (1991), *Yield Spreads and Interest Rate Movements: A Bird's Eye View*. Term-structure signals, expectations, and return dynamics.
- Cochrane, J. H., and M. Piazzesi (2005), *Bond Risk Premia*. Predictive information in bond forward rates, mainly on longer horizons.
- Moskowitz, T. J., Y. H. Ooi, and L. H. Pedersen (2012), *Time Series Momentum*. Cross-asset evidence for trend strategies, with caution about horizons and transfer to ETFs.
- Moreira, A., and T. Muir (2017), *Volatility-Managed Portfolios*. Motivation for distinguishing direction forecasts from risk exposure sizing.
- Bianchi, D., M. Büchner, and A. Tamoni (2021), *Bond Risk Premiums with Machine Learning*. Evidence and methodological limitations in more flexible bond-return prediction; see also the published correction.

These papers motivate testable hypotheses. They do not validate the project's specific feature weights, weekly trading rule, or observed backtest performance.

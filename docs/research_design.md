# Research Design and Validation Protocol

**Project:** ProjectAgentTrading26 · **Group:** 9
**Course:** Data-Driven Models for Investment, LUISS Guido Carli
**Scientific documentation review:** 10 November 2026 · **Code reference:** `integrazione-finale`; archived freeze and OOS manifests are controlling

## 1. Research question and analytical scope

**Question.** Can a systematic multi-asset investment strategy combining ETF prices, macroeconomic indicators, quantitative agents and constrained portfolio construction deliver superior **net risk-adjusted** outcomes versus simple investable benchmarks?

The project investigates both **forecast/decision quality** (do systematic signals help?) and **portfolio engineering** (do those signals lead to economically meaningful exposures after execution and costs?). A positive standalone return is not evidence of incremental alpha. Relative CAGR, volatility, Sharpe, drawdown, benchmark exposure, turnover and practical costs all matter.

The research maintains a distinction between:

- **Hypothesis and development:** ideas, model fitting, strategy comparisons and candidate selection using the pre-2022 period.
- **Official registered evaluation:** a frozen Core35–Satellite65 candidate assessed in 2022–2026, without changing the candidate afterward.
- **Subsequent explanatory research:** temporal subsamples and the Regime-Adaptive extension, reported as retrospective evidence and not substituted for the official result.

This file defines the research design; `decision_log.md` records key decisions and `core_satellite_final_2026.md` contains the longer frozen evaluation description.

## 2. Economic hypotheses

| Hypothesis | Mechanism | Observable prediction | Main falsifier |
|---|---|---|---|
| H1: cross-asset signals | Rates, inflation, equity risk and safe-haven demand can influence relative ETF returns | Allocations conditional on macro and price features improve out-of-sample risk-adjusted results | Passive SPY/IEF or SPY dominates after risk and costs |
| H2: agent consensus | Diversifying technical, macro and ML opinions may screen out some inconsistent trades | Momentum/agent agreement improves the net distribution of returns and/or drawdowns | No uplift relative to simpler momentum-only or passive rules |
| H3: Core–Satellite construction | A stable equity core raises participation while an active sleeve changes exposure | Better return/downside trade-off than standalone, low-exposure B/C systems | Satellite detracts from SPY participation or adds adverse short exposure |
| H4: conditional defensive filter *(later extension)* | Trend breaks may identify periods when negative or inconsistent signals should be restricted | A causal SMA200-triggered Long/Flat gate improves behavior across regimes | Effect limited to the inspected years; little benefit beyond simple SMA200 switching |

**H4 is exploratory, not part of the original frozen candidate specification.** The SMA200 rule uses only data available at each historical decision but was assessed on historical outcomes already known during subsequent research. These are separate methodological points: temporal signal causality does not establish independence of model selection.

## 3. Research lineage: what each level actually does

### 3.1 Treasury origin and diversification pivot

Early research tested duration timing among SHY, IEF and TLT, including historical signals based on changes in the 30-year Treasury yield. Weak later validation relative to a duration buy-and-hold alternative motivated a pivot away from concentrating exclusively on Treasury-duration forecasts.

The selected multi-asset universe is **SPY** (US large-cap equities), **IEF** (intermediate Treasuries), **GLD** (gold) and **SHY** (short Treasuries). Historical TLT input may remain in raw-data archives but does **not** become a fifth instrument in the final Core strategy.

### 3.2 A: transparent benchmarks and volatility control

- **A1:** static investable allocations such as SPY buy-and-hold, 60% SPY/40% IEF and equal-weight baselines.
- **A2:** risk-constrained allocation with a covariance lookback of **126 sessions** and ex-ante volatility target **8%** in the relevant development experiments.

A1/A2 establish whether complexity adds value against basic portfolio construction. The A2 risk limit is **not automatically inherited** by subsequent standalone or Core portfolios.

### 3.3 B: macro budgets and directional trend

- **B_v1:** a signed/long-short exploratory macro and trend framework.
- **B_v2:** macro-regime budgets with a trend-based **Long/Flat** interpretation; the underlying momentum uses returns over **21 and 63 daily sessions**.

The regime research used interest rates, curve slopes, inflation and market-stress data. Historical FRED release vintages are **not fully certified point-in-time**; calendar alignment and deliberate lags do not eliminate all vintage risks. Development-only configuration includes experimental clustering code, not a claim that clustering is active in the final strategy.

### 3.4 Agent ensemble and GA Orchestrator

The agent ensemble includes **EMA/SMA, Z-Score, macro yield-curve, Random Forest and Support Vector Machine** families. RF/SVM are fitted in a chronological research design; SVM research includes four-week forward classification and rolling out-of-fold procedures. A frozen **genetic-algorithm Orchestrator** aggregates agent information into an ETF-level signal **−1, 0, +1** and a **confidence** value on weekly decision dates.

Important distinction: chronological or out-of-fold model-level forecasts do **not** make the *final GA-selected ensemble* independent of its development-period optimization. **FinBERT was not included** in the frozen final ensemble; timestamped historical sentiment availability was not sufficiently established.

### 3.5 C: B + Orchestrator, not just another agent

The **historical standalone Level C** combines B's market structure with the agent outputs. Its relevant inputs are per-ETF agent direction/confidence, the B_v2 momentum panel, B_v2 macro budget panel and historical covariance data.

- **C_v1:** allow an ETF to be **Long** when the B trend is positive and the Orchestrator is Long with positive confidence. Otherwise **Flat**. Scale the B macro budget by those gates and apply the historical **126-session covariance / 8% ex-ante volatility cap**.
- **C_v2:** require sign agreement for **Long** or **Short** between B momentum and the Orchestrator; disagreement produces **Flat**. Retain the B budget basis and the historical signed risk cap.

The C development study is not the archived 2022–2026 official candidate. C results computed over development are not independent after Orchestrator selection.

### 3.6 Final selected candidate: Core35–Satellite65

The **Core35–Satellite65** finalist reuses the **frozen Orchestrator directly**, not the full C_v1/C_v2 budget-and-risk stack.

For each weekly decision date:

1. Begin with a nominal **+35% SPY Core**.
2. Read each eligible ETF's frozen signal and confidence.
3. Exclude a neutral signal or a confidence value below **0.35**.
4. Use direction × confidence divided by the volatility input; frozen signals do not contain volatility columns, so all ETFs use the common **0.10 fallback** in the archived candidate.
5. Apply the original SPY preference/short blocking based on a separate **30-week SMA**, and **1.5% asset-switch hysteresis**.
6. Place a nominal **±65% Satellite** in the chosen ETF; if no signal qualifies, leave the Satellite unallocated. The SPY Core is still present.

Target ETF weights are therefore:

\[
w_{i,t}=0.35\,\mathbf{1}\{i=\mathrm{SPY}\}+0.65\,s_{i,t},
\]

where \(s_{i,t}\in\{-1,0,+1\}\) has at most one nonzero ETF. A −65% SPY Satellite offsets the +35% Core to create **−30% net SPY**, not a perpetual minimum-long equity position.

The absence of ETF-specific realized-volatility inputs is a known selector limitation. Avoid describing this frozen ranking as cross-asset volatility parity.

## 4. Datasets, dates and information availability

| Window | Operational meaning |
|---|---|
| Historical ETF and macro collection from ~2004/2005 | Rolling-indicator history, supervised-model training inputs and regime features |
| Matched portfolio development: 2010–2021 | In-sample development comparison of selected portfolios |
| Fitting/selection cutoff: **31 December 2021** | Frozen model, GA and candidate policy boundary |
| Final evaluation: **7 January 2022–9 October 2026** | 1,193 daily observations; 249 weekly decision dates; 2026 ending before year-end |
| 3 January 2023–9 October 2026 | Descriptive slice of the same previously evaluated OOS NAVs, **not** new retraining or a fresh test |

Price data use adjusted ETF Open/Close to account for distributions and corporate actions. Source collection historically used Yahoo Finance and FRED, with local raw and processed CSV snapshots in the distribution. Decisions are generated using the observation set available by the weekly decision close; orders trade at the **next available trading Open**, not at the observed close.

**Unresolved information risk:** historical macro series may be revised, and their observation dates need not equal publication times. The archive explicitly records `macro_point_in_time_verified: false`. Do not claim definitive vintage-safe macro information or absence of every form of leakage.

## 5. Backtesting, benchmark and cost policy

All official comparisons share a daily position/cash/NAV accounting engine, with portfolios differing only in their target-weight and rebalancing rules.

| Element | Registered setting |
|---|---|
| Asset universe | SPY, IEF, GLD, SHY |
| Decision frequency | Weekly close |
| Execution | Next available trading Open |
| Portfolio accounting | Daily position, cash, borrowing and NAV updates |
| Initial capital | $100,000 |
| Commission | 2 bps of traded notional |
| Slippage | 1 bp of traded notional |
| Borrowing cost | 1% annualized on short notional |
| Cash yield | 0% assumed |
| Risk-free rate for Sharpe | 0% assumed |
| Annualization | 252 trading sessions |
| Benchmarks | SPY buy-and-hold; 60% SPY/40% IEF rebalanced weekly |

The simulator includes a short margin-buffer convention, but does not prove borrow availability, executable limit-order prices, taxable results or impact costs at scale. Target gross weight is at most 1.0; realized gross exposure can drift around rebalances.

## 6. Official results and registered interpretation

**Final, archived 2022–2026 comparison:**

| Portfolio | CAGR | Sharpe | Max drawdown |
|---|---:|---:|---:|
| **Core35–Satellite65** | **5.872%** | **0.551** | **−26.543%** |
| SPY buy-and-hold | 13.116% | 0.799 | −23.424% |
| 60/40 SPY–IEF | 7.388% | 0.694 | −19.840% |

The Core candidate's annualized volatility was **11.566%**, compared with **17.294%** for SPY and **11.174%** for 60/40. Its lower volatility versus SPY does not rescue its weaker risk-adjusted performance. The official system **did not beat either final benchmark** on CAGR, Sharpe or maximum drawdown. These returns measure a particular simulated period, not the expected future performance of any candidate.

Observed trading diagnostics include **151 nonzero executed-trading sessions**, cumulative turnover **130.27**, transaction fees of **$3,883.12** and short-borrow cost of **$296.52**. These distinguish the final portfolio from a nearly frictionless buy-and-hold comparison.

**Development comparison, 2010–2021:** Core CAGR **12.064%**, Sharpe **0.929**, maximum drawdown **−18.291%**. These results are development-period evidence and are not independently validated merely because their sample ends before the official OOS window.

## 7. Post-freeze temporal robustness

A pre-existing daily-return trajectory was split descriptively for **2022** and **2023–2026** without resetting holdings, changing weights or fitting any new models. For the archived Core candidate, the latter window returned **14.343% CAGR**, **1.234 Sharpe** and **−10.035% maximum drawdown**. Over the identical dates, SPY returned **22.381% CAGR** and 60/40 returned **14.061% CAGR**.

The Core's archived 2022 subperiod had **−20.787% cumulative return** (annualized subperiod CAGR **−21.160%**); the figures use different denominators and must not be confused. This supports the conclusion that **sample/regime composition matters**, especially the weak 2022 performance. It does **not** show what would happen if RF/SVM and the Orchestrator had been retrained using 2022 data. The official 2022–2026 metrics remain controlling.

## 8. Post-freeze Regime-Adaptive SMA200 extension

### Decision-time rule

At each weekly close \(t\), compute the adjusted SPY closing-price average over the **preceding 200 trading observations including \(t\)** and define:

\[
D_t=\mathbf{1}\left\{P_{\mathrm{SPY},t}<\frac{1}{200}\sum_{j=0}^{199}P_{\mathrm{SPY},t-j}\right\}.
\]

- **Normal regime, \(D_t=0\):** use the registered original Core35–Satellite65 target portfolio.
- **Defensive regime, \(D_t=1\):** for each candidate ETF, require a positive **21-day and 63-day momentum** signal **and** Orchestrator direction **+1**. Only eligible signals can enter the existing confidence-based satellite ranking. If no ETF passes, the Satellite is Flat and the +35% SPY Core remains.

The normal and defensive selectors maintain **independent hysteresis states**. The daily SMA200 regime is observed at the weekly decision close; execution stays at the next Open. The original 35/65 sizing, 0.35 confidence cutoff, 0.015 hysteresis and uniform 0.10 volatility fallback remain in effect.

**This is a C-inspired *signal gate*, not the full C risk allocator.** It omits C's B_v2 macro budget vector and its 8% ex-ante covariance cap. It also differs from the Core's original 30-week SPY SMA rule.

### Quantitative evidence and interpretation

| Candidate | Development 2010–2021 CAGR / Sharpe | 2022–2026 CAGR / Sharpe / MDD |
|---|---|---|
| Official Core35–Satellite65 | 12.064% / 0.929 | 5.872% / 0.551 / −26.543% |
| Regime-Adaptive SMA200 | **10.599% / 0.889** | **9.805% / 0.912 / −12.802%** |
| B/C-inspired Long/Flat always on | 8.928% / 0.909 | 11.280% / 1.166 / −8.506% |

The defensive condition occurred on **83 of 626 development decisions** and **58 of 249 retrospective 2022–2026 decisions**. The dynamic extension improves selected 2022–2026 historical metrics but lowers development CAGR and Sharpe relative to the original. A simple SPY/SHY SMA200 switching policy provides a necessary explanatory benchmark; the more complex agent strategy has not shown a statistically reliable incremental advantage against it.

The **supplemental** Newey–West five-lag tests fit the adaptive strategy's daily returns on an intercept and one comparator's daily returns, then annualize the intercept by 252. Against SPY, 60/40 and original Core the estimates are respectively **+2.55%, +3.66%, +5.01%**, with **p=0.344, p=0.200, p=0.055**. None reaches 5%; the unadjusted tests do not correct for the several explored candidate rules. The separately stored `Group9_alpha_significance_analysis.csv` supports these diagnostics but is **not** a frozen-project artifact or an output of `main.py --analysis`. **No statistically established alpha is claimed.**

### Implementation

- Research function: `src/level_c_strategy.py` (`build_regime_adaptive_core_targets`).
- Daily simulation and tabulation: `src/submission_runner.py`.
- Command interface: `main.py --analysis regime-adaptive`.
- Safety and reproducibility checks: `tests/test_exploratory_analysis.py`.

The research extension was introduced after the official candidate was frozen. Neither official model binaries, historical Core source nor archived OOS NAVs were modified.

## 9. Methodological threats and mitigations

| Threat | Current treatment | Residual caveat |
|---|---|---|
| Look-ahead in execution | Weekly close determines following Open order | Macro publication/vintage status not fully certified |
| Supervised-label leakage | Chronological RF/SVM fitting, embargo/gap where applicable | Verify each feature and label timestamp separately |
| Model selection / data snooping | Pre-2022 scientific freeze; development/OOS separation | Post-freeze variants are exploratory rather than independent holdouts |
| Survivorship / investability | Long-lived, historically tradable ETFs and adjusted prices | Universe represents a small surviving ETF set, not a full contemporaneous security universe |
| Unrealistic costs | Common fee/slippage/borrow settings | Market impact and borrow availability are not independently demonstrated |
| Non-stationarity | Development vs final holdout and regime slices | One fixed historical episode cannot establish stability |
| Apparent alpha | Risk-adjusted benchmark comparisons, HAC diagnostics | p-values do not imply selection-adjusted discovery; alpha not significant |
| Reproducibility | Scientific 47-file freeze, 13-file release index, exact NAV replay | Git source without untracked binaries/data is not a complete release |

## 10. Reproduction and source-of-truth hierarchy

The order of evidentiary priority is:

1. **Archived final NAV/metrics:** `reports/oos_final_2022_2026/*.csv` and `evaluation_provenance.json`.
2. **Frozen scientific provenance:** `reports/development_freeze_2021.json`, `reports/oos_input_audit.json` and `reports/oos_signals/inference_provenance.json`.
3. **Original frozen code and config:** `src/core_satellite.py`, `src/agent_backtest.py`, `src/execution_engine.py`, `config.yaml`, with `release_config.yaml` as a descriptive contract.
4. **Post-freeze research utilities:** `src/level_c_strategy.py`, `src/submission_runner.py`, `docs/regime_adaptive_exploratory_2026.md`.
5. **Narrative:** this design document, `decision_log.md`, `README.md`, report and presentation; narrative never overrides actual artifacts.

To inspect the registered experiment, use `python main.py` and `python main.py --release-action audit`. The full archive and binaries are required; the Git checkout alone is insufficient. To reproduce the official backtest without tuning, use `python main.py --release-action replay`. To run temporal or adaptive summaries, use `python main.py --analysis temporal-robustness` and `python main.py --analysis regime-adaptive`. Research-mode output is not permitted to overwrite protected datasets or reports.

**Conclusion.** The completed project demonstrates a traceable systematic trading workflow and a rigorously reported negative official OOS comparison. The follow-up Regime-Adaptive research is a plausible, implementable signal-construction extension with encouraging retrospective downside statistics, but not a validated source of independent statistically significant alpha.

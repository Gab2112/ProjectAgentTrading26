# Research Decision Log

**ProjectAgentTrading26 · Group 9**
**Course:** LUISS Data-Driven Models for Investment
**Consolidated:** 9 November 2026 · **Scientific review:** 10 November 2026 · **Repository branch:** `integrazione-finale`

## Purpose and evidence standard

This is a consolidated **research decision record**, not an assertion that every numbered decision was committed on the date of this document. Early decisions were reconstructed from historical experiments and the archived project narrative; dates should not be invented where contemporaneous timestamps are unavailable. The registered scientific freeze, OOS provenance, code and machine-generated outputs control any conflict with this narrative.

Use these labels consistently:

- **Historical development:** researched/selected with information available up to 31 December 2021; development results may reflect selection and are not independent validation.
- **Official frozen evaluation:** the registered Core35–Satellite65 comparison on 7 January 2022–9 October 2026.
- **Post-freeze diagnostic:** explains inspected data or evaluates a subsequently proposed rule; does not revise the registered OOS claim.
- **Release/communication:** ensures reproducibility and clear reporting, without changing trading decisions.

## 1. Decision register

### D01 · Begin with Treasury-duration timing

- **Context:** Research whether yield movements and macro information help allocate among SHY, IEF and TLT.
- **Decision:** Study simple duration-timing rules, including a 13-week change in the 30-year Treasury yield as an example.
- **Observation:** A representative historical validation was weaker than TLT buy-and-hold in 2017–2021, notwithstanding encouraging earlier fitting-period returns.
- **Outcome:** Do not adopt the Treasury-only timing strategy as the final investable candidate; retain the work as the foundation of the multi-asset pivot.
- **Evidence:** Historical Treasury research and project presentation history; not the final registered Core OOS.

### D02 · Move to a four-ETF multi-asset universe

- **Decision:** Trade **SPY, IEF, GLD and SHY** to represent equity, intermediate Treasury duration, gold and short-duration defensive exposure.
- **Reason:** Broaden the economic mechanisms beyond yield direction while maintaining a simple, explainable universe.
- **Boundary:** **TLT remains a historical raw-data input, not a final tradable asset.** Adding new ETF names requires data and model-availability checks, not a relabeling of existing agent forecasts.
- **Evidence:** `config.yaml`, `src/utils.py`, final target-weight CSVs.

### D03 · Establish explicit A1 baselines and A2 risk control

- **Decision:** Retain investable passive benchmarks before moving to complex strategies. Research a volatility-constrained A2 allocation with a 126-session historical covariance window and an 8% risk target.
- **Reason:** Portfolio improvement must be tested against simple alternatives, not only against a zero-return baseline.
- **Boundary:** The **A2 volatility target is not automatically a constraint on Core35–Satellite65**.
- **Evidence:** `config.yaml`; `src/portfolio.py`; historical A1/A2 development pipeline.

### D04 · Introduce macro regimes and momentum in B

- **Decision:** Explore macro-/rate-dependent budgets and 21-/63-session price-momentum signals, initially with signed exposure (**B_v1**) and subsequently a Long/Flat formulation (**B_v2**).
- **Reason:** Separate directional conviction from the budget assigned to each asset or economic regime.
- **Boundary:** Historical macro observations are not fully verified as the actual FRED vintages available at each trading date.
- **Evidence:** `src/regime_strategy.py`, `src/regime_allocator.py`, `release_config.yaml` caveats.

### D05 · Develop heterogeneous technical, macro and ML agents

- **Decision:** Use technical trend/mean-reversion agents, macro yield-curve analysis, Random Forest and SVM forecasts, coordinated by a GA Orchestrator.
- **Reason:** Different model families may carry partially complementary information.
- **Boundary:** Individual RF/SVM out-of-fold predictions do not make the **GA-selected combination** independently validated.
- **Evidence:** `src/agents/`, `src/agent_backtest.py`, `src/orchestrator_loader.py`, frozen signals.

### D06 · Exclude FinBERT from the selected candidate

- **Decision:** Retain the historical source file but **exclude FinBERT from final frozen inference**.
- **Reason:** Publication-timestamped sentiment inputs were insufficiently certified for this evaluation.
- **Outcome:** No FinBERT-derived performance or signal is claimed as part of the final 2022–2026 strategy.
- **Evidence:** `reports/oos_final_2022_2026/evaluation_provenance.json`: `finbert_used: false`.

### D07 · Link B and Orchestrator through the standalone Level C

- **Decision:** Build **C_v1 Long/Flat** with a positive B trend and a positive Orchestrator signal; build **C_v2** with signed momentum/agent agreement.
- **Portfolio logic:** Reuse B_v2 macro budgets and apply a 126-session covariance-based 8% ex-ante volatility overlay to the signed/long-only raw weights.
- **Reason:** Agent consensus acts as an explicit trade-admission rule rather than as an unexamined extra score.
- **Boundary:** C_v1/C_v2 are **development research paths**, not the final selected Core35–Satellite65 implementation.
- **Evidence:** `src/level_c_allocator.py`, `src/level_c_strategy.py` historical C functions.

### D08 · Select the Core35–Satellite65 candidate for final evaluation

- **Decision:** Combine a permanent nominal **+35% SPY Core** with a **±65% single-ETF Satellite** driven directly by the frozen Orchestrator.
- **Economic rationale:** Raise participation relative to some low-exposure standalone B/C portfolios while preserving a dynamic multi-asset decision sleeve.
- **Implementation:** Confidence minimum **0.35**, hysteresis **0.015**, common volatility fallback **0.10**, tie order SHY/IEF/GLD/SPY and separate **30-week SPY SMA** preference/short filter.
- **Important qualification:** The selection was made on development evidence, not because the 2022–2026 result was known. The archived Core portfolio does **not** call the full standalone C budget/cap allocator.
- **Evidence:** `src/core_satellite.py`, `src/agent_backtest.py`, `release_config.yaml`.

### D09 · Freeze parameters, models and scientific inputs at end-2021

- **Decision:** Lock source/configuration, pretrained model artifacts and signal-generation policy before the official final evaluation.
- **Audit trail:** **47 scientific freeze files**, with separate raw-input, signal-inference and release-artifact evidence.
- **Boundary:** A SHA-256 scientific freeze is distinct from the later full-distribution `PACKAGE_MANIFEST.json` and does not certify macro economic-data vintages.
- **Evidence:** `reports/development_freeze_2021.json`; `reports/oos_input_audit.json`; `reports/oos_signals/inference_provenance.json`.

### D10 · Use a common, cost-aware daily execution protocol

- **Decision:** Weekly end-of-session decisions, next available market Open execution and daily units/cash/NAV accounting.
- **Assumptions:** $100,000 initial capital, 2 bps commission, 1 bp slippage, 1% annualized short borrow, unremunerated cash, annualization 252 and risk-free rate zero for Sharpe.
- **Reason:** Eliminate a same-close fill after the signal is computed and compare the Core candidate against SPY and weekly 60/40 under the same engine.
- **Evidence:** `src/execution_engine.py`; `release_config.yaml`; final output CSVs.

### D11 · Preserve the official negative benchmark result

- **Decision:** Report all 2022–2026 archived metrics without after-the-fact parameter changes.
- **Results:** Core CAGR **5.872%**, Sharpe **0.551**, max drawdown **−26.543%**; SPY **13.116% / 0.799 / −23.424%**; 60/40 **7.388% / 0.694 / −19.840%**.
- **Interpretation:** The candidate **did not outperform either benchmark** on the controlling risk/return measures.
- **Evidence:** `reports/oos_final_2022_2026/comparison_metrics.csv` and `evaluation_provenance.json`.

### D12 · Document turnover, short exposure and economic friction

- **Decision:** Attribute results to observable execution and exposure components rather than blaming transaction costs alone.
- **Observation:** Core cumulative turnover **130.27**, modeled transaction fees **$3,883.12**, borrow fees **$296.52**, and **33 weekly SPY-short Satellite decisions**.
- **Implication:** Tests excluding all fees still did not make the official Core outperform the principal benchmarks. Direction, exposure and timing warrant separate diagnostics.
- **Boundary:** These are historical diagnostics; a particular allocation decision is not a causal explanation without a controlled counterfactual.
- **Evidence:** Archived NAV/trading reports; reproducible explanatory research outputs.

### D13 · Add the 2023–2026 temporal-robustness view

- **Decision:** Report the same frozen portfolios for the historical **3 January 2023–9 October 2026** subset.
- **Result:** Core **14.343% CAGR, 1.234 Sharpe, −10.035% drawdown**; SPY **22.381% CAGR**; 60/40 **14.061% CAGR**. In 2022 the Core's **cumulative** return was **−20.787%** (subperiod CAGR **−21.160%**).
- **Critical boundary:** This is **not new model training through 2022**; the NAV continues from the January 2022 starting portfolio, and the period was examined after the official OOS result was known. **No 2023-start replacement official OOS was adopted.**
- **Use:** Regime-sensitivity interpretation and appendix/backup evidence, never a replacement official sample.
- **Evidence:** `src/submission_runner.py`, `python main.py --analysis temporal-robustness`.

### D14 · Add a Regime-Adaptive SMA200 research mode

- **Decision:** Preserve the original 35/65 targets when SPY's weekly decision close is **at or above its trailing 200-daily-session SMA**. Below SMA200, filter candidate Satellite instruments to those with **positive momentum over both 21 and 63 daily sessions AND frozen Orchestrator signal +1**.
- **Implementation:** Reuse the original selector; retain independent normal/defensive hysteresis states, existing confidence threshold and costs. If no candidate passes, the Satellite is Flat while the SPY Core remains.
- **Reason:** Explore a parsimonious condition-dependent admission rule with an economic trend-following rationale, without inventing an indicator that directly encodes the year 2022.
- **Important boundary:** This is a **B/C-inspired consensus gate**, **not** full Level C with macro budgets and its 8% volatility cap. Its historical evaluation is **retrospective** and does not modify the official Core freeze.
- **Evidence:** `src/level_c_strategy.py::build_regime_adaptive_core_targets`; `docs/regime_adaptive_exploratory_2026.md`.

### D15 · Compare the dynamic extension across periods and baselines

- **Observed development 2010–2021:** Original Core **12.064% CAGR / 0.929 Sharpe**, Regime-Adaptive **10.599% / 0.889**.
- **Observed 2022–2026:** Original **5.872% / 0.551 / −26.543% MDD**; Regime-Adaptive **9.805% / 0.912 / −12.802% MDD**.
- **Decision:** Retain the extension as a serious, inspectable alternative with observed defensive benefits, **not** as a proved superior selected OOS strategy.
- **Causal caution:** Being computable from past prices avoids mechanically using future prices in the signal. It does not supply an independent sample for an additional rule evaluated after inspecting the same period.
- **Evidence:** `python main.py --analysis regime-adaptive`, exploratory test suite.

### D16 · Test whether adaptive performance demonstrates alpha

- **Method:** Supplemental one-factor daily-return regressions: adaptive return = intercept + beta × comparator return + residual. The intercept is multiplied by 252; standard errors use **Newey–West/HAC with 5 lags**, zero assumed risk-free rate.
- **Observed alpha estimates and p-values:** Versus SPY **+2.55% annualized, p=0.344**; 60/40 **+3.66%, p=0.200**; original Core **+5.01%, p=0.055**.
- **Decision:** **Do not claim statistically significant alpha at the 5% level.** These are unadjusted diagnostic comparisons and do not control selection over multiple researched variants.
- **Boundary:** Economic logic, historical improvement and statistical significance are different claims. A simple SPY/SHY SMA200 comparator is also relevant before attributing any effect specifically to the agents.
- **Evidence:** Supplemental exploratory `Group9_alpha_significance_analysis.csv`; not part of the 47-file scientific freeze.

### D17 · Keep official, diagnostic and optional presentation artifacts separate

- **Decision:** Keep the original registered 2022–2026 comparison controlling; add the temporal breakdown and SMA200 extension as explicitly labeled supporting research. Update the repository documentation first, then perform a non-destructive repository inventory and cleanup, and update the presentation **last**, followed by the optional academic report.
- **Reason:** Preserve scientific provenance and prevent competing definitions of 'final result'.
- **Release rule:** A `group_9.zip` submission must contain a complete `project/` and a presentation at the ZIP root; regenerate the `project/` distribution manifest for its own contents, then independently verify the presentation at the outer ZIP root. Do not mutate or rewrite the original 47-file freeze to reflect new documentation.
- **Operational status:** Submission documentation, package generation and team approval/merge remain separate operations; do not infer their completion from this research log.

## 2. Artifact ownership and non-destructive editing policy

| Class | Examples | Allowed during documentation revision? |
|---|---|---|
| Scientific freeze | 47-file pre-OOS code/config/model/input hash set | **No mutation** |
| Archived evaluation | Official 2022–2026 NAVs, target CSVs, provenance | **No mutation** |
| Explanatory research | SMA200 code, temporal-analysis commands and tests | Only through reviewed, tested changes; do not overwrite archived outputs |
| Narrative | `README.md`, `docs/research_design.md`, `docs/decision_log.md` | **Yes**, in the present documentation phase |
| Submission/distribution | `PACKAGE_MANIFEST.json`, `group_9.zip`, presentation | Regenerate **only at final packaging**, then audit |

The presence of an untracked local `reports/` folder in a Git checkout is not evidence that the reports are disposable. Inspect and hash the content before deciding what to track, relocate or package.

## 3. What would change the conclusions?

- A **truly new, independent future evaluation** with pre-registered adaptive rules would add evidence about economic generalization.
- A verified point-in-time archive of historical macro releases would reduce macro leakage uncertainty.
- Comparable risk/exposure ablations against simple SMA200 and momentum alternatives would help identify the incremental role of agents rather than beta reduction.
- Updating documentation, cherry-picking promising years or selecting parameters using already-observed outcomes **does not retroactively upgrade** an existing test.

**Current disposition:** Scientific research is complete for the course submission. Preserve the final frozen result, include the diagnostic extension with its caveats, validate repository integrity, and finish the release artifacts in a controlled order.

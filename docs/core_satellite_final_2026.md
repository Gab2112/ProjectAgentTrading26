# Frozen Core-Satellite Research: Final Evaluation

**Project:** ProjectAgentTrading26
**Course:** Data-Driven Models for Investment, LUISS Guido Carli
**Research snapshot:** 7 November 2026
**Status:** Original final OOS evaluation frozen and complete; this is the archived candidate report, not the separate post-freeze SMA200 report
**Study design:** Systematic multi-asset allocation with frozen agent inference and daily execution accounting

---

## 1. Executive Summary

ProjectAgentTrading26 investigates whether a heterogeneous ensemble of systematic forecasting agents can improve the net risk-adjusted performance of an ETF portfolio relative to simple passive benchmarks. The final selected strategy, **Core35_Satellite65**, combines a nominal 35% long SPY core with a 65% dynamically selected long or short satellite. An ensemble of technical, macroeconomic and machine-learning agents informs a frozen Genetic Algorithm (GA) Orchestrator, which supplies weekly directional signals and confidence measures.

The final strategy and model artifacts were frozen using information through **31 December 2021**. The protected out-of-sample (OOS) window began in January 2022 and runs through **9 October 2026** in the archived data. No post-OOS parameter optimization or model refitting was used to improve the reported final result.

**Principal finding:** The final Core-Satellite strategy **did not outperform** either SPY Buy & Hold or a weekly rebalanced 60/40 SPY-IEF portfolio. It achieved a 5.872% CAGR and a 0.551 Sharpe ratio, compared with 13.116% and 0.799 for SPY, and 7.388% and 0.694 for the 60/40 benchmark. Its OOS maximum drawdown was also worse than both comparators.

The research therefore offers an inspectable systematic-trading engineering workflow and an informative **negative empirical result**, not evidence of a deployable alpha strategy. An unresolved limitation is that historic FRED macroeconomic vintages and precise information-release timing have not been fully verified point-in-time.

**Historical scope note (10 November 2026).** This document preserves the original frozen candidate and its official evaluation. The later 2023–2026 slice and regime-adaptive SMA200 results are documented separately in [`regime_adaptive_exploratory_2026.md`](regime_adaptive_exploratory_2026.md); they do not revise the performance in this report.

## 2. Research Question and Testable Claims

> Can a frozen multi-agent signal ensemble, implemented in a disciplined Core-Satellite portfolio, improve net risk-adjusted investment performance and drawdown characteristics over investable passive benchmarks?

Three hypotheses guided the research:

1. Combining diverse information channels could improve robustness relative to individual directional signals.
2. A dynamic satellite could mitigate risk during unfavorable market conditions.
3. Any improvement must survive transaction costs, borrowing fees, chronological separation, and comparison with simpler alternatives.

Positive absolute performance is not sufficient evidence of success. Performance must be interpreted relative to the benchmark portfolios, the risk taken, and the costs required to implement the strategy.

## 3. Asset Universe

| Ticker | Instrument | Research role |
|---|---|---|
| **SPY** | US large-cap equity ETF | Core US equity exposure and directional satellite candidate |
| **IEF** | Intermediate US Treasury ETF | Duration exposure and potential diversification |
| **GLD** | Gold ETF | Alternative asset diversification |
| **SHY** | Short US Treasury ETF | Lower-duration defensive exposure |

The four-ETF universe keeps the strategy interpretable but limits the opportunity set. Diversification cannot be assumed merely because four assets are eligible: realized exposures depend on the selected satellite and may concentrate in a single asset.

## 4. Research Architecture

The historical architecture includes these agent families:

- **SMA/EMA:** moving-average trend relationships.
- **Z-Score:** departures from a rolling historical reference, used in a systematic signal rule.
- **Macro Yield Curve:** macroeconomic conditions represented through lagged weekly features.
- **Random Forest (RF):** nonlinear tree-ensemble forecasts.
- **Support Vector Machine (SVM):** machine-learning classification using chronological training procedures.
- **Frozen GA Orchestrator:** combines individual agent information into directional and confidence signals; its parameters were selected before the final OOS evaluation.

The macro pipeline applies a weekly lag to specified macro features to reduce forward-information leakage. This is distinct from validating the publication timestamps and historic vintages of external economic series.

**FinBERT was excluded** from the final candidate because historically timestamped sentiment inputs could not be adequately certified for the experiment.

### System workflow

```text
Historic ETF prices + macro observations
                  |
                  v
        Decision-date features
                  |
        +---------+---------+
        |         |         |
        v         v         v
   Technical    Macro     RF / SVM
     agents     agent      models
        |         |         |
        +---------+---------+
                  |
                  v
        Frozen GA Orchestrator
                  |
                  v
      Weekly direction/confidence
                  |
                  v
         Satellite selection
                  |
        +---------+---------+
        |                   |
        v                   v
  SPY Core +35%      Satellite +/-65%
        |                   |
        +---------+---------+
                  |
                  v
           ETF target weights
                  |
                  v
      Next available trading Open
                  |
                  v
     Daily positions, cash and NAV
                  |
                  v
        Returns, risk and costs
```

This final Core-Satellite system is not the same as the project's earlier experimental **C_v1** and **C_v2** allocations. Those versions remain part of the development record and should not be confused with the selected final portfolio.

## 5. Frozen Portfolio Construction

### 5.1 Core and satellite

The strategy starts with a **nominal +35% SPY core**. A **65% satellite** is dynamically assigned based on the frozen selection procedure, which uses agent-derived direction and confidence. The saved signal CSVs lack a volatility column, so the selector uses the fixed 0.10 volatility fallback for every asset, **not an ETF-specific historical volatility estimate**.

| Setting | Frozen value |
|---|---:|
| SPY core allocation | +35% |
| Satellite allocation | 65% nominal magnitude |
| Minimum confidence | 0.35 |
| Hysteresis parameter | 0.015 |
| Default volatility input | 0.10 |
| Decision frequency | Weekly |
| Execution rule | Next available trading Open |

Additional frozen selector rules:

- **SPY SMA filter:** block a short SPY signal whenever SPY is above its 30-observation weekly moving average.
- **SPY selection boost:** under that condition multiply the SPY score by `min(2.5, max(1.2, close/SMA30))`, giving at least a 20% preference even just above the SMA.
- **Volatility fallback:** all eight archived Orchestrator signal CSVs include direction and confidence only; `volatility = 0.10` for every instrument when calculating the selection ranking. This is **not** realized-volatility-based risk parity.
- **Tie preference:** `SHY, IEF, GLD, SPY` is the historical ordering, with asset-switch hysteresis; a sign reversal within one asset does not necessarily count as switching assets.
- **Exposure:** target gross weight is constrained to 100%, but marked-to-market realized gross exposure can exceed it through price drift. A realized maximum around 101.51% was observed.

Frozen source and `config.yaml` remain authoritative for historical model behavior. `release_config.yaml` records the final settings as an assertion rather than as editable training knobs.

### 5.2 Net equity exposure can be negative

The core does **not** impose a minimum positive net equity exposure. If the satellite chooses **-65% SPY**, then:

```text
SPY core:       +35%
SPY satellite:  -65%
                 ---
Net SPY:        -30%
```

This is not a conventional permanent-equity, long-only Core-Satellite portfolio. The distinction matters to any interpretation of leverage, defensive behavior and short-borrow exposure.

### 5.3 Hysteresis

Hysteresis is intended to reduce unnecessary changes of satellite selection when alternatives have similar scores. Its presence does not guarantee low turnover, as the OOS simulation shows.

## 6. Trading Execution and Portfolio Accounting

### Decision and execution separation

Weekly signals are interpreted as decisions after the relevant weekly information set. Orders are scheduled for the **next available trading Open**; the simulator does not award a same-close fill after a closing price is known.

### Research cost assumptions

| Assumption | Setting |
|---|---:|
| Initial NAV | USD 100,000 |
| Commission | 2 basis points on traded notional |
| Slippage | 1 basis point on traded notional |
| Short-borrow fee | 1% annualized, when applicable |
| Cash remuneration | Zero |
| Return sampling | Daily NAV |

These are model assumptions, not observed broker quotes. The execution engine tracks units, cash, NAV, trades, turnover, commissions, slippage, borrowing expenses and gross/net/short exposures. Neither dynamic market impact nor assured borrow availability is established by this backtest.

### Comparable benchmarks

- **SPY_buy_hold:** one initial SPY purchase, without recurring target-weight rebalancing.
- **Sixty_Forty:** 60% SPY and 40% IEF, with the configured weekly rebalancing convention.

All three use the same daily simulation accounting framework for comparable evaluation, while retaining their deliberately different allocation and rebalancing policies.

## 7. Chronological Research Protocol

| Stage | Time range | Purpose |
|---|---|---|
| Feature warm-up | Earlier history, before evaluated development trades | Rolling indicators and feature history |
| Development | 2010-2021, cutoff 31 Dec 2021 | Model estimation, GA optimization and selection |
| Protected OOS | Jan 2022 to 9 Oct 2026 | Final evaluation of unchanged frozen candidate |

The earliest archived weekly OOS signal is **7 January 2022**; there are **249 weekly signals through 9 October 2026**. An order requires a subsequent available trading Open to execute. The last decision date should not be mistaken for a guaranteed executed order.

### 7.1 In-sample interpretation of development

The chosen final system was selected using development-period information; the Orchestrator was also optimized in that research window. Accordingly, 2010-2021 performance is **in-sample for the final selected system**, even though individual RF/SVM training procedures may include chronological or out-of-fold safeguards.

There is **no separately demonstrated independent validation result for the final selected Core-Satellite model** before the protected OOS evaluation.

### 7.2 Frozen inference and causal checks

The development freeze contains SHA-256 digests of **47 files**, covering the relevant frozen research inputs, code, settings and model artifacts. The OOS inference used pre-existing frozen parameters and binaries; no additional fitting was performed.

Eight prefix-based signal reconstructions were reported as passing causal-consistency checks, supporting the absence of obvious future-price dependence in those tests. They are not a proof that every historical data source was available at its assigned timestamp.

### 7.3 Macroeconomic vintage caveat

Historical **FRED series vintages and exact publication-time availability were not fully certified point-in-time**. A revised observation visible in a modern download might differ from what was known at the original trading date. This limits the strength of any real-time implementability claim and should accompany the reported OOS results.

## 8. Development Results: 2010-2021

The development comparison was reproduced using the frozen development CLI.

| Strategy | CAGR | Annual volatility | Sharpe | Maximum drawdown | Executed trading sessions |
|---|---:|---:|---:|---:|---:|
| **Core35_Satellite65** | **12.064%** | **13.210%** | **0.929** | **-18.291%** | **205** |
| SPY_buy_hold | 14.767% | 16.948% | 0.898 | -33.717% | 1 |
| Sixty_Forty | 10.859% | 9.356% | 1.149 | -19.490% | 625 |

The Core-Satellite candidate had a shallower development drawdown than SPY and a slightly higher Sharpe ratio, but a lower CAGR. The 60/40 portfolio achieved the strongest development Sharpe. These observations motivated a final holdout evaluation; they do **not** establish generalizable alpha.

The trade counts represent nonzero executed trading sessions, not individual orders or the same quantity as cumulative turnover.

## 9. Final OOS Performance: 2022-2026

### 9.1 Main comparison

All final NAVs assume **USD 100,000 starting capital**, before the strategies diverge.

| Strategy | CAGR | Annual volatility | Sharpe | Maximum drawdown | Ending NAV |
|---|---:|---:|---:|---:|---:|
| **Core35_Satellite65** | **5.872%** | **11.566%** | **0.551** | **-26.543%** | **$131,013.46** |
| SPY_buy_hold | 13.116% | 17.294% | 0.799 | -23.424% | $179,222.60 |
| Sixty_Forty | 7.388% | 11.174% | 0.694 | -19.840% | $140,133.46 |

**The selected strategy underperformed both benchmarks in CAGR, Sharpe and maximum drawdown.** It did exhibit lower annual volatility than SPY, but had slightly higher volatility than 60/40.

It would be incorrect to claim that the final system consistently reduced drawdown risk or delivered superior risk-adjusted returns.

### 9.2 Calendar-year returns

| Year | Core-Satellite | SPY | 60/40 |
|---|---:|---:|---:|
| 2022 | -20.79% | -16.03% | -14.48% |
| 2023 | +10.60% | +26.18% | +16.90% |
| 2024 | +22.17% | +24.89% | +14.35% |
| 2025 | +12.78% | +17.72% | +14.52% |
| 2026 YTD, through 9 Oct | +8.53% | +15.06% | +7.05% |

Annual percentages are rounded. **2022 is partial because the first recorded OOS decision was 7 January; 2026 is partial through 9 October.**

The substantial 2022 loss and the lower subsequent capture of equity gains are consistent with the weaker overall OOS result, but the table alone cannot prove which model decisions caused the difference.

## 10. Trading and Exposure Diagnostics

### 10.1 Core-Satellite summary

| Diagnostic | Final OOS observation |
|---|---:|
| Weekly satellite decisions | 249 |
| Nonzero executed trading sessions | 151 |
| Cumulative turnover | 130.27 |
| Transaction costs | $3,883.12 |
| Short-borrow fees | $296.52 |
| Mean gross exposure | ~90.6% |
| Mean short exposure | ~7.4% |

**Cumulative turnover of 130.27** is the sum of the simulation's turnover ratios; it is **not** a single-period 130.27% return or an annual turnover rate.

### 10.2 Benchmark trading costs

| Strategy | Nonzero trading sessions | Cumulative turnover | Transaction costs |
|---|---:|---:|---:|
| Core35_Satellite65 | 151 | 130.27 | $3,883.12 |
| SPY_buy_hold | 1 | 1.00 | $29.99 |
| Sixty_Forty | 248 | 3.06 | $95.28 |

A frequently rebalanced strategy may still incur low aggregate turnover if individual rebalances are small. By contrast, directional changes in a 65% satellite can cause much larger changes in traded notional.

### 10.3 Satellite decision distribution

| Decision | Weeks |
|---|---:|
| SPY Long | 139 |
| SHY Long | 43 |
| SPY Short | 33 |
| IEF Long | 13 |
| SHY Short | 11 |
| GLD Long | 8 |
| IEF Short | 2 |
| **Total** | **249** |

The system selected SPY Long or SPY Short in **172 of 249 weeks**, indicating material dependence on equity timing despite the four-asset universe. In **33 weeks**, a -65% SPY satellite offset the +35% core, producing a net target of -30% SPY.

Trading costs, switching frequency, confidence thresholds and exposure timing are plausible contributors to OOS underperformance. This is an **interpretation**, not an independently estimated attribution of excess returns.

## 11. Scientific Provenance and Protected Artifacts

The project records research provenance using exact file hashes and separate development/OOS artifacts.

| Artifact | Purpose |
|---|---|
| `reports/development_freeze_2021.json` | Original 47-file pre-OOS freeze |
| `reports/oos_input_audit.json` | Input-file and historical data integrity evidence |
| `reports/oos_signals/inference_provenance.json` | Frozen inference provenance and checks |
| `reports/oos_final_2022_2026/evaluation_provenance.json` | Final one-shot evaluation provenance |
| `reports/release_artifact_index.json` | Hashes of 13 release artifacts |
| `reports/oos_final_2022_2026/comparison_metrics.csv` | Archived performance comparison |
| `reports/oos_final_2022_2026/*_daily.csv` | Archived daily portfolio trajectories |
| `reports/oos_signals/signals_orchestrator_*.csv` | Four frozen ETF signal files |

The original scientific integration commit is **`14ac63f`**. Subsequent commits add the read-only release CLI (`5ec9431`) and the development CLI (`ec8af5b`). The release index points to the original integration commit deliberately; later tooling commits do not change which scientific code was frozen.

An external archive, **`ProjectAgentTrading26_frozen_20261107.zip`**, was created and verified before the later release-CLI work. Its SHA-256 was recorded in the local release index. This preserves the original scientific experiment but should not be described as a full snapshot of all later documentation/software-release changes.

Some `.joblib` binaries, processed data and generated signals are not tracked by Git. A new clone will **not** reproduce the complete frozen evaluation solely from the tracked source tree unless the corresponding verified artifacts are restored.

## 12. Supported Local Commands (Group 9 release)

Use **Python 3.11**, the included frozen artifacts and the declared dependencies.

```bash
python main.py                                      # Read frozen OOS results
python main.py --release-action audit                # 47 + 13 + complete ZIP manifest + calendars
python main.py --release-action verify-models        # Load all four GA parameter snapshots safely
python main.py --level CORE                         # Frozen 2010-2021 development backtest
python main.py --level C                            # Historic C_v1/C_v2, development only
python main.py --release-action replay               # Independent temp-copy OOS backtest replay
python main.py --release-action charts               # Generate figures from archived NAV
python -m unittest discover -s tests -p 'test_*.py' -q
```

`--level ALL` executes historical A1, A2, B_v1, B_v2 and C development
comparisons in an isolated temporary folder before the final Core development
comparison. `--level C` and `--level CORE` are distinct: C belongs to the
historical B/C consensus research, while CORE is the selected 35/65 candidate.

**Reproduction versus retuning:** The OOS replay reproduces the original
frozen execution on the same input data in a disposable workspace and
compares all five core outputs against the originals. It does **not** train,
optimize or select new parameters. This is permitted as a reproducibility
check, but new hyperparameter tuning after viewing OOS is not valid.

**Audit scope:** `PACKAGE_MANIFEST.json` verifies every file distributed in
the submission including `data/processed/market_daily.csv`, which the
scientific 47-file freeze did not originally cover. The historical 47-file
manifest is preserved unaltered. Both OOS and development signal date
calendars are required to be complete across the observed window.

**Orchestrator loading:** The legacy `OrchestratorAgent.load_params()` does not
understand nested frozen `best_params` with asymmetric thresholds. Use
`src.orchestrator_loader.load_frozen_orchestrator(asset)`; this adapter
validates and reconstructs the exact effective weights and directional
thresholds without modifying the frozen agent implementation.

**Testing scope:** The local Group 9 checkout was reported to pass **275 tests, 5 skipped** on 10 November 2026. This historical report was originally drafted alongside a smaller standalone release ZIP; its test counts and file-map examples should not be confused with the current full Git branch. GitHub Actions tests the tracked source tree, but does not restore all ignored frozen model/report artifacts for a complete release audit.

## 13. Research Interpretation

The final experiment separates the **engineering achievement** of constructing a systematic, testable pipeline from the **economic question** of whether it beats a passive alternative. The former includes agent integration, trading-rule specification, a shared execution engine, borrowing cost accounting, chronological evaluation, reproducibility controls and archival provenance. The latter was **not demonstrated** in the final OOS results.

The contrast between encouraging development drawdowns and weaker OOS performance underscores several risks: model-selection bias, regime sensitivity, directional concentration, turnover, trading costs and incomplete information-vintage reconstruction. Those mechanisms are reasonable hypotheses; a causal explanation would require additional prespecified analysis, not opportunistic tuning of the existing final holdout.

## 14. Limitations

1. **Historic macro vintages:** Complete point-in-time verification is unavailable.
2. **Development selection:** Development-period scores are not independent validation of the chosen final model.
3. **Execution realism:** Fixed costs do not capture every liquidity, borrow, margin, tax or operational constraint.
4. **Concentration:** A multi-asset eligibility set does not guarantee diversified realized risk.
5. **Finite OOS sample:** One 2022-2026 historical period is not a guarantee of future outcomes.
6. **Artifact availability:** The Git tree alone is not a complete frozen scientific archive; binaries and generated artifacts need controlled restoration.
7. **No post-test retuning:** Any later rule modification constitutes a new experiment and must not retroactively rewrite the frozen OOS result.

## 15. Release and Submission Checklist (historical snapshot; recheck at final packaging)

- [x] Define and freeze the 2010-2021 selected candidate.
- [x] Preserve model and training provenance.
- [x] Verify historical input hashes.
- [x] Generate and audit frozen OOS signals.
- [x] Complete the single final OOS evaluation.
- [x] Preserve the negative benchmark comparison.
- [x] Create a scientific backup ZIP and release artifact index.
- [x] Add release integrity and archived-result CLI commands.
- [x] Add the frozen development comparison CLI.
- [x] Local test run on 10 November 2026: **275 tests, 5 skipped, zero failures** (user-provided test output).
- [x] Add a final-status entry point to `README.md`.
- [x] Clearly distinguish historical A/B/C from the final strategy.
- [x] Include model and data artifacts needed by the submission.
- [ ] Verify this revised submission on a fresh Python 3.11 environment (current container: Python 3.13).
- [x] Preserve historical CI evidence separately from a full package audit; no claim of a current 275-test GitHub Actions run.
- [ ] Team review and merge of the original research PR.
- [ ] Add the finalized presentation to the project root and the required outer ZIP root.

## 16. Final Conclusion

**Core35_Satellite65 did not outperform SPY Buy & Hold or the 60/40 SPY-IEF benchmark in the final protected OOS evaluation.** It delivered lower CAGR and Sharpe and a more severe maximum drawdown than either benchmark. The favorable development-period drawdown comparison was not replicated.

ProjectAgentTrading26 should therefore be presented as a transparent, reproducibility-oriented systematic trading research exercise, including the negative results and the outstanding point-in-time macro-data limitation, **not as a validated source of investable alpha**.

---

## Appendix A. Main File Map

```text
ProjectAgentTrading26-final/
|-- main.py
|-- config.yaml
|-- src/
|   |-- core_satellite.py
|   |-- frozen_oos_inference.py
|   |-- final_oos_backtest.py
|   |-- release_pipeline.py
|   |-- execution_engine.py
|   `-- agents/
|-- reports/
|   |-- development_freeze_2021.json
|   |-- oos_input_audit.json
|   |-- release_artifact_index.json
|   |-- oos_signals/
|   |   |-- inference_provenance.json
|   |   `-- signals_orchestrator_*.csv
|   `-- oos_final_2022_2026/
|       |-- comparison_metrics.csv
|       |-- evaluation_provenance.json
|       `-- *_daily.csv
|-- tests/
|   |-- test_release_cli.py
|   `-- test_core_development_cli.py
`-- docs/
    `-- core_satellite_final_2026.md
```

## Appendix B. Research Integrity Statement

The published final metrics refer to the archived one-shot evaluation. Subsequent release tooling does not modify frozen scientific inputs, models, strategy parameters or OOS performance outputs. SHA-256 records provide evidence of file consistency, not independent proof of every historical data availability assumption or future trading profitability.

## Appendix C. Exact evaluation vs optimized GA surrogate

The training Genetic Algorithm evaluates candidate agent combinations using
a weekly, per-ETF fitness function with its own exposure and trading-cost
conventions. The selected Core-Satellite portfolio then uses one asset at a
time in the satellite, a permanent 35% nominal SPY core, and a daily next-open
accounting engine. Thus the GA did **not** directly optimize the precise daily
net Core-Satellite portfolio objective. This mismatch may contribute to
underperformance, but no independent attribution experiment was conducted.

The **2005-2021** period contains model-fitting and warm-up history; matched
portfolio development metrics are calculated for **2010-2021**. Chronological
OOF forecasting for some ML agents does not independently validate the
GA-selected final combination. The macro one-week lag noted for the SVM
features must not be generalized automatically to all agents or all FRED
observation vintages.

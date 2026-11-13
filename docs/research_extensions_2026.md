# Equal Active and SMA200: Retrospective Research Extensions

**Group 9 | 13 November 2026 | Supplemental research, not an amended official OOS**

## Scientific boundary

The official, pre-2022-frozen selected portfolio remains **Core35-Satellite65**. Its protected final evaluation covers **7 January 2022 to 9 October 2026**, with 249 weekly decisions, 1,193 daily NAV rows, and original net annualized metrics **5.872% CAGR**, **0.551 Sharpe**, **-26.543% maximum drawdown**. The research below was designed *after the group saw* this result, and is not a newly independent holdout. Its results cannot replace the registered final OOS test or establish significant outperformance.

## Common, fully reconciled retrospective calculation

All strategies below use **the same saved 2022-2026 Orchestrator signals** from `reports/oos_signals/`, **the same daily adjusted open/close market history**, the same next-session-open execution and mark-to-market accounting, **2 bps commissions + 1 bp slippage**, and **1% annual short borrowing cost**. The assumed risk-free rate is 0; annualization is 252 trading sessions. Cash earns 0. This enables a *controlled allocation-policy comparison*, not a reconstruction of the original 2021-start weekly script by Gabriele.

| Strategy | Scientific status | CAGR | Sharpe | Maximum drawdown | Mean gross exposure |
| --- | --- | ---: | ---: | ---: | ---: |
| Core35-Satellite65 | Frozen official selected policy | 5.872% | 0.551 | -26.543% | 90.60% |
| Equal Active, SPY/IEF/GLD | Post-selection policy variant | 5.391% | 0.678 | -13.432% | 69.16% |
| Equal Active, SPY/IEF/GLD/SHY | Post-selection policy variant | 4.587% | 0.692 | -13.432% | 79.84% |
| Regime-Adaptive SMA200 | Post-selection regime extension | 9.805% | 0.912 | -12.802% | 86.81% |
| SPY Buy & Hold | Official benchmark | 13.116% | 0.799 | -23.424% | 99.92% |
| Weekly 60/40 SPY/IEF | Official benchmark | 7.388% | 0.694 | -19.840% | 99.92% |

Figures are reported to three decimals. Complete unrounded numbers and daily paths are in `research_outputs/strategy_comparison_metrics.csv` and `research_outputs/strategy_comparison_nav.csv`. Both the official Core and the benchmarks are independently replayed from the same simulator and **their daily NAVs match the archived results**. The official archive's `reports/` is left unchanged.

### Equal Active allocation

The rule is adapted from Gabriele Di Raimondo's post-freeze research implementation:

- SPY is active **unless** the saved GA Orchestrator indicates short with confidence at least 0.20.
- IEF, GLD, and optionally SHY are active only on a saved long (+1) with confidence at least 0.20.
- Equal-weight active ETFs, capped at 50% each. When multiple assets are active, capped weights may be re-normalized to invest fully; with only one active asset, the other 50% stays in non-interest-bearing cash.
- Keep the previous weekly target unless some absolute target-weight change reaches 5 percentage points.
- No leverage, no shorting under this research policy. The 3-ETF and 4-ETF variants are separately displayed.

**Do not conflate this replay with Gabriele's original backtest.** His exploratory `src/test_oos_agent.py` started in **2021**, used weekly close-to-close returns and zero transaction costs, and used a different Sharpe convention. Since the model development cutoff is **31 December 2021**, the 2021 rows are not out-of-sample. The original reported 3-ETF figures therefore are **not** transplanted into the common-table results. His code and plots are retained through Git history / `research_experiments` instead of being silently rewritten.

### Regime-Adaptive SMA200 allocation

- SPY daily adjusted close **at or above** its prior-200-trading-session moving average: keep the **unchanged** 35% SPY Core + 65% frozen dynamic satellite target.
- SPY close **below** SMA200: require a positive 21-session and 63-session price momentum **and** a frozen Orchestrator +1 signal before a satellite can be selected. Preserve the original confidence threshold 0.35, volatility fallback 0.10, hysteresis 0.015 and frozen tie-breaking rule. Remaining weight can be left flat; keep the +35% SPY core.
- Above- and below-regime selectors retain **independent hysteresis state**. SMA200 is formed from prices available by each weekly decision close; the next trade is executed at the subsequent open. No RF, SVM, or GA refit occurs.
- The SPY SMA200 stress condition is active on **58 of 249 weekly decision dates** on the retrospective 2022-2026 window.

The SMA200 strategy was designed/assessed **after inspecting OOS results**. A simple SPY/SHY SMA200 strategy had similar retrospective performance in the prior research, and reported HAC intercept tests did **not** establish significant alpha at 5% after conventional testing. We therefore make **no causal attribution to the multi-agent layer** and no predictive-performance generalization.

## Research code and verification

```bash
python scripts/run_strategy_comparison.py
python -m unittest discover -s tests -p 'test_*.py' -q
python main.py --release-action audit
python main.py --release-action verify-models
python main.py --release-action replay
```

The comparison script is **read-only by default**. Optional CSV/plot export requires `--output-dir` pointing **outside `project/`** (e.g. `../group9_research_replay`), so the packaged files under `research_outputs/`, the package manifest and frozen records stay unmodified. It verifies the saved original Core targets and all three official daily NAV series before presenting any additional results. The repo's `PACKAGE_MANIFEST.json` covers these new research files but does not alter the historical `development_freeze_2021.json` or `release_artifact_index.json`.

## Interpretation for the oral presentation

Equal Active has much smaller drawdowns than frozen Core, in part with lower average market exposure; **it does not have higher CAGR** on this matched replay. The adaptive SMA200 policy has stronger descriptive CAGR and risk metrics on the same observed window; however that retrospective gain might reflect trend-timing and policy selection after seeing 2022, not a new statistically proven alpha. The official selection and its negative benchmark result remain the central experimental finding.

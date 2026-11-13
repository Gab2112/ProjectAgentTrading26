# Final Policy Study: Equal Active 4 ETFs and Regime-Adaptive SMA200

**Group 9 | 13 November 2026 | Completed allocation-policy comparison**

## Scientific boundary and the original 2021 freeze

The completed project compares two final policy alternatives and uses the
original Core35-Satellite65 candidate as the frozen research baseline.
The model **training cutoff** (31 December 2021), development-only policy
scores, and **historical policy-selection timestamp** are distinct facts:
computing a development ranking today does not retroactively register it in
2021. The archived 2021 manifest continues to certify Core35-Satellite65.

### Development-period policy comparison: 2010-2021

The table below is computed exclusively from historical market data truncated
at 31 December 2021 and saved development signals. It uses the same next-open
simulator and trading-cost conventions as the 2022-2026 comparison.

| Policy | CAGR | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Core35-Satellite65 | 12.064% | 0.929 | -18.291% |
| **Equal Active, four ETFs** | 8.187% | **1.353** | **-9.251%** |
| **Regime-Adaptive SMA200** | 10.599% | 0.889 | -16.484% |

Equal Active (four ETFs) has the strongest development Sharpe and smallest
maximum drawdown among the three displayed policies. Its better risk-adjusted
development result does not establish that it was historically registered as
the selected 2021 policy; the published historical selection record remains
unchanged.

The original, pre-2022-frozen selected portfolio was **Core35-Satellite65**. Its protected final evaluation covers **7 January 2022 to 9 October 2026**, with 249 weekly decisions, 1,193 daily NAV rows, and original net annualized metrics **5.872% CAGR**, **0.551 Sharpe**, **-26.543% maximum drawdown**. The research below was designed *after the group saw* this result, and is not a newly independent holdout. Its results cannot replace the registered final OOS test or establish significant outperformance.

## Common, fully reconciled 2022-2026 historical evaluation

All strategies below use **the same saved 2022-2026 Orchestrator signals** from `reports/oos_signals/`, **the same daily adjusted open/close market history**, the same next-session-open execution and mark-to-market accounting, **2 bps commissions + 1 bp slippage**, and **1% annual short borrowing cost**. The assumed risk-free rate is 0; annualization is 252 trading sessions. Cash earns 0. This enables a *controlled allocation-policy comparison*, not a reconstruction of the original 2021-start weekly script by Gabriele.

| Strategy | Scientific status | CAGR | Sharpe | Maximum drawdown | Mean gross exposure |
| --- | --- | ---: | ---: | ---: | ---: |
| Core35-Satellite65 | Original frozen reference policy | 5.872% | 0.551 | -26.543% | 90.60% |
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

Equal Active has much smaller drawdowns than frozen Core, in part with lower average market exposure; **it does not have higher CAGR** on this matched replay. The adaptive SMA200 policy has stronger descriptive CAGR and risk metrics on the same observed window; however that retrospective gain might reflect trend-timing and policy selection after seeing 2022, not a new statistically proven alpha. The original frozen evaluation and the two concluding policy studies are all reported, with distinct provenance.

The two concluding policies are declared independently of the original frozen
Core configuration in `final_policy_config.yaml`, with a separate research
provenance and historical date boundaries. The original training manifest
and archived selected strategy are never modified.

## Repeatable two-period final policy report

```bash
python scripts/evaluate_final_policies.py
python main.py
```

These entry points display the complete development-only and later historical
comparison without rewriting trained models, frozen manifests or stored NAVs.
The 2021-only calculation does not use 2022-2026 market prices or signals to
produce the development scores. A separate out-of-time prospective assessment
would be needed for a newly independent policy-validation claim.

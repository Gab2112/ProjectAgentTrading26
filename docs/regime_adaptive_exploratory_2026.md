# Temporal Robustness and Regime-Adaptive Extension

This is an **exploratory, post-freeze** extension to ProjectAgentTrading26.
The official Core35--Satellite65 OOS candidate, original metrics and archive
checksums are preserved. The 2022--2026 results were inspected before these
variants were designed. Their performance is retrospective, not new independent OOS.

## Commands (from repository root)

```bash
python main.py --analysis temporal-robustness
python main.py --analysis regime-adaptive
python main.py --analysis regime-adaptive --output-dir exploratory_outputs
python -m unittest discover -s tests -p 'test_exploratory_analysis.py' -q
```

No output CSV is created unless `--output-dir` is passed. Output paths inside
`reports/`, `data/`, `src/`, `tests/`, `docs/`, or `figures/` are prohibited.
If an output directory is desired in the repository, use the untracked
`exploratory_outputs/` folder; avoid broad `git add .` commands.

## Temporal robustness

The 2023--2026 figures are calculated from the **existing** daily portfolio
return series beginning in January 2022. Holdings are not reset at January
2023, no supervised model is refit, and no new backtest period is designated.
All sample windows are descriptive slices of the same archived evaluation.

## Regime-adaptive extension

- For SPY close **at or above** its past-200-trading-session simple average,
  preserve the frozen Core35--Satellite65 target weights.
- For SPY close **below** that SMA200, apply the B_v2 positive 21/63-session
  momentum gate and require a +1 direction from the frozen agent Orchestrator
  for ETF eligibility, then run the existing frozen satellite selector.
- Preserve 35% SPY Core and nominal 65% single-asset Satellite, with the
  original volatility fallback, confidence threshold and hysteresis.
- Evaluate development (2010--2021) and retrospective previously seen
  2022--2026 on the same daily next-open execution engine and original costs.
- Both potential satellite selectors maintain independent hysteresis state.
  This is an explicit modeling assumption of the experiment, not a claimed
  property of the original frozen candidate.

This mode reuses the original `src/level_c_strategy.py` as the extension site,
`src/submission_runner.py` for read-only calculation and `main.py` as a CLI.
The scientific freeze lists 47 code/model/input files. None are edited here.

## Interpretation

The original candidate's full 2022--2026 CAGR remains 5.872%, Sharpe 0.551,
MDD -26.543%; neither this extension nor the temporal slice changes that
registered finding. Include experimental outcomes only as retrospective
analysis or supporting appendix, not a revised official OOS test.

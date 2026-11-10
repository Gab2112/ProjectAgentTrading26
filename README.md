# ProjectAgentTrading26 | Group 9

**LUISS Guido Carli — Data-Driven Models for Investment**
**Research system:** multi-asset systematic trading with quantitative agents
**Document revision:** 9 November 2026 · **Code reference:** `integrazione-finale`, commit `5e2f2b0`

> **Reading guide.** The archived **Core35–Satellite65** strategy is the official, pre-2022-frozen candidate. Temporal-slice statistics and the **Regime-Adaptive SMA200** extension are subsequent research diagnostics. They are not replacements for, or independent new out-of-sample validations of, the official candidate.

## 1. At a glance

This project studies whether quantitative, macroeconomic and machine-learning signals can improve portfolio performance after costs. The research progressed from Treasury timing to an investable four-ETF universe (**SPY, IEF, GLD, SHY**), through static and risk-controlled baselines (**A**), macro/momentum allocations (**B**), an agent-consensus layer (**C**), and a selected **Core35–Satellite65** implementation.

The final candidate allocates a nominal **+35% to SPY** and **±65% to one dynamically selected ETF**. An ensemble of technical, macro, Random Forest and SVM agents supplies signals and confidence through a **frozen genetic-algorithm Orchestrator**. Decisions occur at weekly closes; orders execute at the next available trading open. Cash, positions, transaction costs and short-borrowing expenses are accounted for daily.

The official 2022–2026 evaluation is an informative **negative benchmark result**. Later diagnostics identify temporal sensitivity and possible defensive improvements; they do **not** retroactively change the original evaluation.

## 2. Registered final comparison

**Official final evaluation: 7 January 2022–9 October 2026; 1,193 daily observations, 249 weekly decisions.**

| Portfolio | CAGR | Annualized volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| **Core35–Satellite65, official** | **5.872%** | 11.566% | **0.551** | **−26.543%** |
| SPY buy and hold | 13.116% | 17.294% | 0.799 | −23.424% |
| 60% SPY / 40% IEF, weekly rebalanced | 7.388% | 11.174% | 0.694 | −19.840% |

These results use **$100,000 initial capital**, **2 bps commissions + 1 bp slippage on traded notional**, **1% annualized short borrow**, unremunerated cash, 252 trading days/year and a zero assumed risk-free rate for Sharpe. The official candidate underperformed both benchmarks on CAGR, Sharpe and drawdown. Its 2022–2026 ending NAV is **$131,013.46**.

Source of truth: `reports/oos_final_2022_2026/comparison_metrics.csv` and `reports/oos_final_2022_2026/evaluation_provenance.json` in the complete release archive. The Git checkout alone may not contain these binary/data/report artifacts.

## 3. Further analyses (not a new independent holdout)

### Temporal robustness

The archived NAVs were also summarized separately for 2022 and **3 January 2023–9 October 2026**. The latter is a descriptive subset of the **same continuing simulation**: positions are **not reset**, signals are **not retrained**, and no new test sample is created.

| Portfolio | 2023–2026 CAGR | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Core35–Satellite65 | **14.343%** | 1.234 | −10.035% |
| SPY buy and hold | 22.381% | 1.431 | −18.755% |
| 60/40 SPY–IEF | 14.061% | 1.428 | −10.505% |

This helps quantify the effect of the exceptionally difficult 2022 sample; it does **not** justify dropping 2022 from the official test.

### Regime-Adaptive SMA200 research extension

The extension combines the **same 35% SPY core** with one of two satellite-selection rules:

1. **Normal regime:** SPY close at or above its trailing **200-trading-session SMA** → original Core-Satellite targets.
2. **Defensive regime:** SPY close below SMA200 → only ETFs with **positive 21- and 63-session momentum** and **Orchestrator signal +1** are eligible for the satellite. If none qualifies, the satellite is flat. The original selector then ranks eligible ETFs using its frozen confidence, fallback-volatility and hysteresis conventions.

The SMA200 uses information observable at the decision close and the result is executed at the next trading open. The extension is economically motivated, but was evaluated on already-inspected historical samples; treat the outcomes as **retrospective**.

| Policy | Development 2010–2021 CAGR | 2022–2026 CAGR | 2022–2026 Sharpe | 2022–2026 Max DD |
|---|---:|---:|---:|---:|
| Original Core35–Satellite65 | 12.064% | 5.872% | 0.551 | −26.543% |
| B/C-inspired Long/Flat filter, always enabled | 8.928% | 11.280% | 1.166 | −8.506% |
| **Regime-Adaptive SMA200** | **10.599%** | **9.805%** | **0.912** | **−12.802%** |

The adaptive strategy underperformed the original in development. Exploratory HAC/Newey–West regressions do **not** establish positive alpha significant at the 5% level versus SPY (p≈0.344), 60/40 (p≈0.200), or the frozen Core strategy (p≈0.055), using a 5-lag daily-return specification. These p-values are diagnostic and do not correct for multiple strategy searches. The alternative is **not presented as a statistically validated alpha generator**.

**Important architecture distinction:** the historical `C_v1`/`C_v2` modules include **B_v2 macro budgets** and a **126-session covariance-based 8% ex-ante volatility cap**. The Regime-Adaptive *Core* strategy reuses the **B/C idea of momentum-plus-agent agreement**, **not** those macro budgets or the 8% cap. It is not identical to standalone Level C.

Detailed policy: [`docs/regime_adaptive_exploratory_2026.md`](docs/regime_adaptive_exploratory_2026.md).

## 4. Reproducible environment

Use **Python 3.11** and the complete submission `project/` tree, including verified input data and model artifacts. The source-only GitHub tree may be insufficient for a full replay.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Read and verify the official release

```bash
python main.py                              # display frozen final metrics
python main.py --release-action audit       # scientific freeze, data and release integrity
python main.py --release-action verify-models
python main.py --release-action replay      # exact locked OOS replay in an isolated temporary copy
python main.py --release-action charts      # archived equity/drawdown visuals
```

The original OOS replay reconstructs **the same frozen candidate** and compares the archived results. It performs **no fitting or post-hoc selection**.

### Historical development experiments

```bash
python main.py --level A1
python main.py --level A2
python main.py --level B --b-version v1
python main.py --level B --b-version v2
python main.py --level C
python main.py --level CORE
python main.py --level ALL
```

These modes are **development-only** and run in disposable working copies when they can generate files. The historical 2005–2021 model/indicator history and the **2010–2021 matched portfolio development** window have different meanings; never label development comparison results as independent OOS.

### Additional diagnostics

```bash
python main.py --analysis temporal-robustness
python main.py --analysis regime-adaptive
python main.py --analysis regime-adaptive --output-dir exploratory_outputs
python -m unittest discover -s tests -p 'test_*.py' -v
```

The research commands do **not** overwrite frozen output. Optional research CSV export must not target `data/`, `reports/`, `src/`, `docs/`, `tests/` or `figures/`. Avoid committing `exploratory_outputs/` unless deliberately curating an appendix.

## 5. Architecture: how B, C and the Core relate

```text
Adjusted ETF prices + macro series
           |
           +--> A1 static portfolios / A2 risk control
           |
           +--> B: macro regimes, budgets and 21/63 momentum
           |                   |
           +--> Technical / macro / RF / SVM agents
                               |
                     GA Orchestrator
                               |
                     direction + confidence
                               |
                     +---------+---------+
                     |                   |
             C_v1 / C_v2           Official Core35/65
          B budgets + momentum     Core + direct agent
          + agent consensus        satellite selection
          + portfolio vol cap               |
                                           +--> SMA200 research extension
                                                conditional Long/Flat gate
```

The **official Core35/65 does not run standalone C**. The newer regime extension reintroduces the *consensus principle* conditionally, without changing the registered historical Core candidate.

## 6. Frozen strategy specifics

- Weekly signal, next-open execution; no same-close execution after observing a close.
- Confidence threshold **0.35**; asset-switch hysteresis **0.015**.
- Saved Orchestrator signals contain **no ETF volatility column**. The selector uses the common fallback **0.10** for all ETFs; do not claim ETF-specific volatility-based ranking.
- SPY-short filter and preference use an additional **30-week SMA**. This is distinct from the extension's **200-daily-session SMA**.
- The Core can be outweighed by a short satellite: **+35% SPY −65% SPY = −30% net SPY**.
- The target gross exposure is at most 100%, though realized exposure can drift between rebalances.
- FinBERT code exists historically but was **not used** in final frozen inference; TLT is retained for earlier Treasury data audit, **not traded** in the four-ETF final candidate.

## 7. Files and provenance

| Path | Role |
|---|---|
| `main.py` | Safe command-line interface |
| `src/agents/`, `src/agent_backtest.py` | Agent research and frozen-compatible selector |
| `src/level_c_allocator.py`, `src/level_c_strategy.py` | Historical C and additional regime extension |
| `src/core_satellite.py` | Frozen Core target construction |
| `src/submission_runner.py` | Isolated development and research summaries |
| `src/execution_engine.py` | Shared next-open execution, cash and NAV |
| `config.yaml` | Original historical research settings; **not** final strategy override |
| `release_config.yaml` | Read-only assertions of the frozen candidate |
| `data/` | Source, processed data and agent signals (may require release archive) |
| `reports/` | Freeze/verification evidence, official NAVs and provenance (may be untracked in Git) |
| `docs/research_design.md` | Hypotheses, design, methods, validation and limitations |
| `docs/decision_log.md` | Chronological decisions, rationale and reproducibility boundaries |
| `docs/core_satellite_final_2026.md` | Detailed archived Core strategy report |
| `docs/regime_adaptive_exploratory_2026.md` | Retrospective SMA200 extension contract |
| `tests/` | Repository and exploratory regression checks |
| `PACKAGE_MANIFEST.json` | Full **submission ZIP** file integrity index (rebuild when packaging) |

The historical **47-file scientific freeze** and the **13-file release artifact index** must remain unchanged when editing ordinary documentation. `PACKAGE_MANIFEST.json` is a separate manifest of files **inside `project/`**: rebuild it after those project files are finalized. Then add and verify `presentation.pdf` or `.pptx` separately at the **outer ZIP root**. Do not conflate these integrity layers.

## 8. Integrity, limitations and submission

- The FRED macro input **publication vintages are not completely point-in-time verified**, even where the code uses deliberate lags.
- Chronological RF/SVM out-of-fold signals do not make a GA-selected ensemble independent of the development-period selection process.
- Fixed commission/slippage and annual borrow are assumptions, not executable broker guarantees.
- Empirical superiority on an inspected sample is not proof of future alpha; the exploratory SMA200 result has **no demonstrated statistically significant alpha** at 5% using the cited diagnostics.
- A clean Git checkout may need separate binary/model/report artifacts; `python main.py --release-action audit` is the final authority for restored release inputs.

**Submission requirement:** package a `group_9.zip` containing `project/` with reproducible scripts, data, models and outputs, and **`presentation.pdf` or `presentation.pptx` at the ZIP root**. Presentation is prepared last; the optional academic report can be packaged separately if the group chooses.

See [`docs/research_design.md`](docs/research_design.md) for methods and [`docs/decision_log.md`](docs/decision_log.md) for the full decision trail.

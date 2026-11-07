# ProjectAgentTrading26 | Final Research Release

**LUISS Guido Carli, Data-Driven Models for Investment**
**Research status: 7 November 2026**
**Python: 3.11**

This repository contains a systematic multi-asset research project combining
technical, macroeconomic and machine-learning agents. Its selected final
candidate is **Core35_Satellite65**: a nominal 35% SPY Core and a 65%
dynamic long/short Satellite selected through a frozen agent Orchestrator.

**Finding: the final strategy did not outperform either benchmark in the
protected 2022-2026 OOS evaluation.** The project presents a reproducible
research workflow and its negative empirical result, not a validated alpha
strategy or an investment recommendation.

## Final results

| Strategy | OOS CAGR | Volatility | Sharpe | Max drawdown | Final NAV (USD 100k start) |
|---|---:|---:|---:|---:|---:|
| Core35_Satellite65 | 5.872% | 11.566% | 0.551 | -26.543% | $131,013.46 |
| SPY Buy & Hold | 13.116% | 17.294% | 0.799 | -23.424% | $179,222.60 |
| 60/40 SPY-IEF | 7.388% | 11.174% | 0.694 | -19.840% | $140,133.46 |

Development window: 2010-2021, with model selection and GA optimization.
Protected OOS: January 2022 through 9 October 2026. The final OOS was
evaluated once with frozen models and portfolio rules; no post-test retuning.
The final candidate's development scores are **in-sample**, not an
independent validation set. Exact historic FRED macro vintages and
publication availability were **not fully point-in-time certified**.

## Read first

- [Final research design, evidence and limitations](docs/core_satellite_final_2026.md)
- [Reproducibility and artifact-restoration instructions](docs/reproducibility_release_2026.md)
- [Release summary for pull-request review](docs/release_summary_2026.md)
- [Archived 4 November README and historical A/B/C descriptions](docs/readme_archived_2026-11-04.md)
- [Historical Level C research](docs/level_c_research.md)
- [Research decision log](docs/decision_log.md)

## System components

`SPY`, `IEF`, `GLD` and `SHY` comprise the ETF universe. The analysis
combines SMA/EMA, Z-Score, macro yield curve, Random Forest and SVM agents
through the frozen GA Orchestrator. FinBERT is excluded from the selected
final strategy. Weekly decisions are executed at the next available Open
by a daily NAV and cost-accounting engine. The simulation includes 2 bps
commission, 1 bp slippage and 1% annual short borrow where applicable.

Key modules are `src/core_satellite.py`,
`src/frozen_oos_inference.py`, `src/final_oos_backtest.py`,
`src/execution_engine.py`, `src/agents/` and `src/release_pipeline.py`.

## Installation and local commands

Use Python 3.11, preferably in a fresh virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**The read-only release commands require archived `reports/` data and the
frozen model assets, which are not all tracked by Git.** Obtain the
separately distributed, verified release ZIP and follow the restoration
procedure in the [reproducibility guide](docs/reproducibility_release_2026.md).

```bash
python main.py --release-action audit
python main.py --release-action frozen-results
python main.py --core-development
python -m unittest discover -s tests -p 'test_*.py' -q
```

The first command verifies file hashes and provenance. The second reads
archived OOS results; neither performs model fitting or backtesting.
`--core-development` reproduces the **2010-2021** development backtest
from frozen inputs, but does not refit models. Do not run the one-shot
OOS evaluator as a means of iterating on the strategy.

**Legacy research interface:** `python main.py --level ALL` still means
A1+A2. The original `--level C` dispatcher is not implemented; it does
not represent the final Core-Satellite workflow. See the archived README
for older research experiments and commands.

## Distribution and verification

The development freeze contains **47 hashed files**. The local post-OOS
release index verifies **13 published result artifacts**. Historical
inputs, model binaries (`*.joblib`), development signals, processed data
and final OOS reports must accompany a clean-clone reproduction.

The release package `ProjectAgentTrading26_RELEASE_20261107.zip` provides
tracked source plus the required ignored scientific artifacts and an
internal SHA-256 manifest. It is distributed separately and is **not**
committed to this Git repository. GitHub Actions runs the unit tests
on pull requests; it does not replace the complete artifact audit.

**Verified local checkpoint (7 November 2026):** 272 Python unit tests
passed, 47 frozen files checked, and 13 OOS archive artifacts checked.
The clean-release restoration check and remote CI are separately reported
when executed; neither is inferred solely from these numbers.

## Academic integrity and limitations

The final Core-Satellite strategy trailed SPY and 60/40 on OOS CAGR and
Sharpe, and suffered a deeper maximum drawdown. Reporting this result
faithfully is part of the research protocol. The frozen data and code
hashes verify consistency with the archived experiment, but cannot
independently establish historic macro availability, model profitability
or live execution feasibility.

# Group 9 | ProjectAgentTrading26

**LUISS Guido Carli · Data-Driven Models for Investment**
**Python 3.11 · Scientific freeze 7 November 2026 · Final submission edition 13 November 2026**

This completed project evaluates a **multi-agent investment system with three
portfolio-allocation policies**. The final comparative focus is on
**Equal Active (four ETFs: SPY, IEF, GLD and SHY)** and
**Regime-Adaptive SMA200**. The originally selected and frozen
**Core35-Satellite65** strategy remains an immutable historical baseline.
All three reuse the same pre-2022 fitted agent ensemble.

**Historical provenance.** The archived 2021 development-freeze record identifies
Core35-Satellite65 as the original candidate selected for its protected
2022-2026 evaluation. The additional Equal Active and SMA200 policy comparison
was reconstructed after that historical evaluation had been examined. A
reconstructed development-only ranking is useful evidence about the policies,
but it is **not proof of a contemporaneous 2021 registration** for those new
allocation rules. The official Core result and its negative benchmark finding
remain intact.

## Final policy study: development and subsequent historical evaluation

All policies use the same previously trained four-ETF signals, daily next-open
execution and the same cost assumptions. The development calculation consumes
**only prices and agent signals dated on or before 31 December 2021**.

| Policy | 2010-2021 CAGR | 2010-2021 Sharpe | 2010-2021 MDD | 2022-2026 CAGR | 2022-2026 Sharpe | 2022-2026 MDD |
|:--|--:|--:|--:|--:|--:|--:|
| Core35-Satellite65 (archived reference) | 12.064% | 0.929 | -18.291% | 5.872% | 0.551 | -26.543% |
| **Equal Active, 4 ETFs** | 8.187% | **1.353** | **-9.251%** | 4.587% | 0.692 | -13.432% |
| **Regime-Adaptive SMA200** | 10.599% | 0.889 | -16.484% | **9.805%** | **0.912** | **-12.802%** |

Equal Active has the strongest development Sharpe and shallowest development
drawdown among these three policies; SMA200 has the strongest 2022-2026 Sharpe
and return among the three. Neither claim implies statistical evidence of
future alpha. The policy set and the later period had been examined during
research, so the 2022-2026 comparison of **new policies** is historical rather
than a second independent, pre-registered test. The archived Core result is the
original frozen OOS evaluation. Its SPY and weekly 60/40 comparators remain
available in `reports/oos_final_2022_2026/`.

Reproduce this two-period study directly:

```bash
python main.py
python scripts/evaluate_final_policies.py
python main.py --release-action frozen-results   # original Core OOS archive
python main.py --release-action audit            # unchanged scientific freeze
python main.py --release-action replay           # unchanged frozen Core replay
```

The completed policy set is explicitly declared in `final_policy_config.yaml`;
this separate research configuration does **not** replace the registered
`release_config.yaml` for the original historical Core policy.

**No new RF/SVM/GA training was needed:** the new policies change allocation
rules, not the previously trained predictive agents. Full provenance is in
`docs/research_extensions_2026.md` and `docs/release_architecture_2026.md`.

## 1. Installation (no remote data downloads)

Extract this ZIP, enter its **`project/`** directory, and create an environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows activate `.venv` with the corresponding shell command. Raw data,
processed market data, eight frozen RF/SVM binaries, weekly signals and final
OOS outputs are included. No API keys or data downloads are required.

## 2. Main interface and execution order

```bash
python main.py                              # Final multi-policy project results
python main.py --release-action audit       # Scientific + ZIP integrity audit
python main.py --release-action verify-models
python main.py --level CORE                # Re-run the development-period comparison
python main.py --release-action replay      # Full locked OOS backtest replay in isolated temp copy
python main.py --release-action charts      # Generate equity and drawdown charts
```

The **`replay`** command reruns the originally selected and frozen Core strategy on the
**same** OOS inputs in a disposable temporary project, compares the original
metrics, target weights and three daily NAV files, then discards the copy. It
does **not** refit models, re-run GA optimization or alter archived results.
This archival command never retunes the original GA/Core parameters from the OOS; the two additional policies have their separate, explicitly documented research history.

Historical A/B/C development experiments remain available separately:

```bash
python main.py --level A1
python main.py --level A2
python main.py --level B --b-version v1
python main.py --level B --b-version v2
python main.py --level C         # Matched B_v1/B_v2/C_v1/C_v2 development comparison
python main.py --level ALL       # All historical development experiments + CORE
python -m unittest discover -s tests -p 'test_*.py' -q
```

A/B/C development commands generate derived inputs **only inside temporary
copies** to prevent overwriting the frozen submission. The C experiments are
**not** aliases for the final Core-Satellite strategy and are **in-sample**
for the GA Orchestrator. All development tests remain restricted to information
through 2021. The submission includes a focused smoke-test suite; the original
repository's 275-test checkpoint is documented separately and is **not**
reproduced by the reduced test suite.


### Optional reproduction of the presentation

The final editable PowerPoint is included at the ZIP root. To rebuild it from
its **included source code**, use Node.js 20 or newer and the pinned
`pptxgenjs` version from `package.json`:

```bash
npm install --ignore-scripts
GROUP9_SLIDES_OUTPUT=../group9_presentation_rebuilt.pptx npm run build:slides
```

Specify an output outside `project/` as above: the signed submission manifest
covers the published `project/presentation.pptx`, and an ordinary visual
rebuild should not overwrite those audited bytes. Regeneration is optional
for grading; no Node.js dependency is needed for Python research execution.

## 3. Archived Core-Satellite methodology (historical baseline)

- Weekly close signal, **next trading-day Open** execution, daily NAV/cash and
  short-borrow accounting; USD 100k initial capital, 2 bps commission, 1 bp
  slippage, 1% annual borrow on shorts, unremunerated cash.
- The 35% SPY core is combined with one 65% satellite; the satellite can
  offset the core, generating a **-30% net SPY target** when short SPY is
  selected. The target gross exposure is <=100%, but actual exposure can drift.
- Satellite confidence cutoff **0.35**, asset-switch hysteresis **0.015**.
  The saved Orchestrator signals contain **no per-asset volatility column**:
  the selector uses the constant fallback **0.10** for all assets, so the
  ordering normally follows absolute confidence, not estimated volatilities.
- SPY shorts are blocked above its 30-week SMA. Above that SMA, SPY's
  selection score receives a multiplier from **1.2 to 2.5**. The hysteresis
  keys on asset identity, not necessarily a sign reversal in that asset.
- The chosen weights, thresholds and GA parameters are **pre-OOS snapshots**.
  `config.yaml` and frozen source modules remain byte-identical. The read-only
  `release_config.yaml` states the effective rules and checks consistency;
  changing it does not retune historical models.

## 4. Directory map

- `main.py`: unified entry point, safe default and explicit modes.
- `config.yaml`: original frozen legacy training configuration.
- `release_config.yaml`: effective **historical Core-Satellite** frozen contract.
- `src/`: reusable data, agent, strategy, simulation, integrity and plotting code.
- `src/orchestrator_loader.py`: official compatible loader for nested GA
  `best_params`, including separate long/short thresholds.
- `data/raw/`: original market and macro snapshots, no network calls.
- `data/processed/`: development market input needed for exact replay.
- `data/signals/`: frozen development Orchestrator signals.
- `reports/`: 47-file scientific freeze, raw-input hashes, OOS provenance,
  249 decision signals and benchmark results.
- `figures/`: pre-generated OOS equity and drawdown images.
- `docs/core_satellite_final_2026.md`: detailed design, risks and findings.
- `tests/test_submission.py`: targeted submission-interface tests.
- `PACKAGE_MANIFEST.json`: integrity index for *every distributed file*.

**Why are TLT and FinBERT source present?** TLT.csv is retained solely to
satisfy the original 16-input historical audit: TLT is **not invested**.
The FinBERT agent source remains as a non-operative transitive import in the
frozen modules. FinBERT is **not part of the final ensemble**.

## 5. Scientific cautions and scope

The 2005-2021 fitting window and 2010-2021 matched development comparison are
**not equivalent**. Development results involve model selection and must not
be described as independent final validation. The GA optimizer uses a
single-asset weekly surrogate fitness, not the exact net portfolio objective
of the final daily Core-Satellite backtest. FRED macro observation vintages
and historical release timestamps have **not** been completely verified as
point-in-time. Fixed execution costs are research assumptions.

The original Git collaboration history documents research decisions beyond
the fully reproducible scientific snapshot distributed in this submission. **The completed `group_9.zip` contains `presentation.pptx` at its outer ZIP root as required by LUISS**, while the protected historical results remain unchanged.

## 6. Final policy comparison and historical evaluation

The final research comparison presents
Gabriele Di Raimondo's **Equal Active** policy (3 and 4 ETFs) and Gianluca
Addorisio's **Regime-Adaptive SMA200** extension on the **same 2022-2026 daily
execution engine**, archived frozen Orchestrator signals and 2 bps + 1 bp costs.
**These policies were considered after observing the original OOS**. Their
the new-policy metrics are historical and must **not** be presented as an independent new
final OOS or as a replacement for the selected Core35-Satellite65 candidate.

| Strategy | Status | CAGR | Sharpe | Maximum drawdown |
|:--|:--|--:|--:|--:|
| Core35-Satellite65 | Frozen registered OOS | 5.872% | 0.551 | -26.543% |
| Equal Active 3 ETFs | Retrospective, matched replay | 5.391% | 0.678 | -13.432% |
| Equal Active 4 ETFs | Retrospective, matched replay | 4.587% | 0.692 | -13.432% |
| Regime-Adaptive SMA200 | Retrospective, matched replay | 9.805% | 0.912 | -12.802% |
| SPY Buy & Hold | Frozen benchmark | 13.116% | 0.799 | -23.424% |
| Weekly 60/40 SPY/IEF | Frozen benchmark | 7.388% | 0.694 | -19.840% |

The new comparisons **independently reconcile the original Core and benchmark
NAV histories with archived daily results**, but reproduce only Gabriele's
allocation *concept*, not his original 2021-start weekly zero-fee backtest.
His original full implementation is preserved in the documented Git
history (collaboration snapshot `4f53268`), distinct from the scientific
contents of this final ZIP. In particular, neither 2021 rows nor his older
Sharpe calculations are included in the common-window comparison.

```bash
python scripts/run_strategy_comparison.py    # Read-only matched replay; prints current metrics
python scripts/run_strategy_comparison.py --output-dir ../group9_research_replay  # Optional CSV/plot export outside project
```

See [`docs/research_extensions_2026.md`](docs/research_extensions_2026.md)
for exact eligibility rules, execution conventions, limitations and provenance,
and [`docs/release_architecture_2026.md`](docs/release_architecture_2026.md)
for Git/release boundaries. The original temporal-robustness analysis is **retained
in the technical report**, but omitted as a standalone slide in the oral deck.

## 7. Final submission package

The ZIP root contains `project/` and `presentation.pptx`. The `project/` tree
contains exact frozen code/model sources, raw/processed data, original signals,
13 protected OOS release artefacts, 47-file freeze record, reproducible research
code and CSV/plots, documentation, supplementary full report, and final package
manifest. Historical scientific manifests were **never rewritten**. Only the
submission `PACKAGE_MANIFEST.json` is regenerated to reflect the completed package.

Recommended verification:

```bash
python main.py --release-action audit
python main.py --release-action verify-models
python main.py --release-action replay
python -m unittest discover -s tests -p 'test_*.py' -q
```

The GitHub repository is a collaboration record. The reproducible scientific
submission is the separately packaged ZIP, which must also be extracted and
audited from a clean folder.

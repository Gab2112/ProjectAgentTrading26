# Reproducibility and Final Group 9 Submission | 10 November 2026

**Project:** ProjectAgentTrading26 | **Course:** Data-Driven Models for Investment (LUISS)

This document describes the **current source branch** and the expected **complete
course submission**, not a claim that a new archive has already been built or tested.
The 7 November scientific freeze and original evaluation remain unchanged.

## 1. Evidence layers: do not confuse their roles

| Evidence | Location | Interpretation |
|---|---|---|
| Original scientific freeze | `reports/development_freeze_2021.json` | Checksums for **47 original code/model/input files** |
| Input integrity | `reports/oos_input_audit.json` | Historical raw-input checks |
| Model inference provenance | `reports/oos_signals/inference_provenance.json` | Frozen OOS agent-signal provenance |
| Registered final evaluation | `reports/oos_final_2022_2026/` | Official archived 2022–2026 NAV, metrics and weights |
| Release evidence | `reports/release_artifact_index.json` | Checksums for **13 release artifacts** |
| Submission manifest | `project/PACKAGE_MANIFEST.json` **inside the final ZIP** | Checks all project files *after* documentation, slides and report copies are finalized |

The 47-file freeze and the 13-artifact index are **historical records**: do not
regenerate or rewrite them when updating Markdown or packaging. The release
index intentionally refers to an earlier scientific integration commit, not
necessarily to the latest documentation commit. `release_config.yaml` is a
read-only specification of the selected strategy, while `config.yaml` retains
its historical development settings: neither is a prompt to retune the model.

## 2. Restoring a complete working project

Use **Python 3.11**. The Git repository contains tracked source and raw data,
but may **not** contain generated inputs (`data/processed/`, `data/signals/`),
frozen `.joblib` agent models or `reports/`. A source-only clone is **not** a
complete release. Obtain the finished `group_9.zip` rather than claiming a
source clone can replay the original evaluation unaided.

Expected course archive layout:

```text
group_9.zip
|-- project/
|   |-- README.md
|   |-- main.py
|   |-- config.yaml
|   |-- release_config.yaml
|   |-- requirements.txt
|   |-- PACKAGE_MANIFEST.json
|   |-- src/  data/  reports/  docs/  tests/
|   |-- presentation.pptx           # optional duplicate in project root
|   `-- final_report.tex            # optional, after final editorial review
`-- presentation.pptx              # REQUIRED by LUISS at the ZIP outer root
```

If the final deck is a PDF, the outer item may instead be `presentation.pdf`.
The optional report's eventual exact filename should match the actual release;
it is **not** yet guaranteed to exist. The `.github/workflows` directory may
be included for source provenance; it is not required to run tests locally.

For a fresh extraction:

```bash
unzip group_9.zip
cd project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --release-action audit
python main.py --release-action frozen-results
python main.py --release-action verify-models
python main.py --release-action replay
python -m unittest discover -s tests -p 'test_*.py' -q
```

The full unit suite includes tests for training and utilities; the audit and
replay specifically depend on the separately archived model/signal/report
artifacts. A short CI run on GitHub **does not** prove the packaged scientific
release is complete. MacOS and Linux may differ in solver/library behavior;
record the Python version and actual package installation results.

## 3. Correct interpretation of commands

| Command | What it does | What it does **not** do |
|---|---|---|
| `python main.py` | Audit and read archived final metrics | Fit or optimize models |
| `python main.py --release-action audit` | Verify freeze/input/release hashes, decision dates and optional package manifest | Run OOS training or backtesting |
| `python main.py --release-action frozen-results` | Read archived OOS comparison | Change the frozen strategy |
| `python main.py --release-action verify-models` | Check the four frozen Orchestrators can load | Retrain RF/SVM/GA |
| `python main.py --release-action replay` | Reconstruct **the same** OOS strategy in an isolated working copy and compare archived outputs | Re-select a candidate or claim a fresh independent OOS |
| `python main.py --level CORE` | Re-evaluate the archived 2010–2021 development strategy | Open the final OOS for tuning |
| `python main.py --level C` | Development-only historical B/C consensus study | Run the selected Core portfolio |
| `python main.py --analysis temporal-robustness` | Report existing continuous NAV series over 2022 and 2023–2026 slices | Train a 2022-extended model or create a new test |
| `python main.py --analysis regime-adaptive` | Retrospective conditional-SMA200 research on saved frozen agents and prices | Change archived Core OOS metrics or demonstrate validated alpha |

For optional research CSV output, `--analysis regime-adaptive --output-dir
exploratory_outputs` saves a file outside protected folders. The output
must not be placed in `data/`, `reports/`, `docs/`, `src/`, `tests/` or
`figures/` inside the checked-out project.

## 4. Source-of-truth results and limitations

The **official unchanged** OOS comparison covers 7 January 2022 to
9 October 2026 (1,193 daily observations, 249 weekly decisions):

| Strategy | CAGR | Sharpe | Maximum drawdown |
|---|---:|---:|---:|
| Frozen Core35–Satellite65 | 5.872% | 0.551 | −26.543% |
| SPY buy & hold | 13.116% | 0.799 | −23.424% |
| Weekly 60/40 SPY/IEF | 7.388% | 0.694 | −19.840% |

The 2023–2026 retrospective Core CAGR **14.343%** is a continuous historical
slice, **not** a 2023-start retrained OOS. The Regime-Adaptive SMA200 variant
showed 2022–2026 **9.805% CAGR**, **0.912 Sharpe**, **−12.802% MDD**, but is
post-freeze research; see `docs/regime_adaptive_exploratory_2026.md`.
Historical FRED vintages are not fully point-in-time verified. Fixed
commission/slippage/borrow fees and stylized execution are not live quotes.

The supplemental HAC alpha p-values (SPY 0.344; 60/40 0.200; Core 0.055) are
not produced by the public CLI and **not** part of the scientific 47-file
freeze. If the group includes them in the final report, retain and identify
the supplemental daily-return input and calculation script separately.

## 5. Final release verification (not yet completed)

After the last source/document/deck/report change, regenerate the distribution
`PACKAGE_MANIFEST.json` from the **actual files under `project/`**, excluding
`.git`, `.venv`, caches, transient logs and the manifest itself. The check is
strict: an added file can cause `audit_release` to fail until indexed.

From a **fresh extracted copy** of `group_9.zip`:

1. Check that `project/` and **outer** `presentation.pptx` or `.pdf` exist.
2. Run `python main.py --release-action audit` and `--release-action replay`.
3. Run `python -m unittest discover -s tests -p 'test_*.py' -q`.
4. Verify official and supplemental figures match their source files.
5. Independently verify the ZIP file list, manifest SHA-256 entries and the
   presentation/report binary integrity after packaging.

Keep `reports/` and `.joblib` files in the distributed scientific package even
if they remain ignored or untracked in Git; deleting them breaks
reproduction. The original `ProjectAgentTrading26_frozen_20261107.zip`, if
available, is an **external historical backup**, not a replacement for the
current complete LUISS submission. GitHub Actions verifies tracked-source
behavior, not the complete original OOS archive.

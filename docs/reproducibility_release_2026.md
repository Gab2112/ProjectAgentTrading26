# Final Release Reproducibility | 7 November 2026

This runbook documents the **frozen Core-Satellite** experiment, not the
historical 2010-2016 C_v1/C_v2 exploratory comparisons.

## Inputs and boundaries

- Development: 2010-2021; all agent and orchestrator fitting/selection.
- Protected OOS: January 2022 to 9 October 2026; frozen evaluation once.
- Frozen manifest: `reports/development_freeze_2021.json` (47 hashes).
- Data audit: `reports/oos_input_audit.json`.
- Signal provenance: `reports/oos_signals/inference_provenance.json`.
- Evaluation provenance: `reports/oos_final_2022_2026/evaluation_provenance.json`.
- Release index: `reports/release_artifact_index.json` (13 artifacts).
- Selected model: Core35_Satellite65, 35% SPY Core, 65% dynamic Satellite.

## Software environment

Python **3.11** and the dependencies from `requirements.txt` are required.
On macOS, verify the chosen interpreter using `python3.11 --version` and
avoid accidentally using the system's Python 3.13 for this release.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The CI environment independently installs dependencies from this file.
In practice, dependency resolution on a new date may differ if a package
is no longer available for that platform; record versions if divergence
occurs. FinBERT (`torch`, `transformers`) is optional and not required.

## Clean restoration: preferred single-archive method

Receive the separately distributed file
`ProjectAgentTrading26_RELEASE_20261107.zip` and its SHA-256 checksum.
The archive contains Git-tracked code and documents, frozen model binaries,
ignored development inputs, OOS reports and a `RELEASE_MANIFEST.json`.

On a machine with Python 3.11:

```bash
mkdir ProjectAgentTrading26_reproduced
cd ProjectAgentTrading26_reproduced
unzip ../ProjectAgentTrading26_RELEASE_20261107.zip
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --release-action audit
python main.py --release-action frozen-results
python main.py --core-development
python -m unittest discover -s tests -p 'test_*.py' -q
```

The archive is not itself a Git clone and does not contain `.git` history.
If Git history is required, clone the official repository and copy the
archive's *ignored data/model/report files* into the clone without
overwriting a different tracked source version. Check frozen hashes after
restoration. Never restore onto a dirty original `modulo-C` worktree.

## What each command verifies

**`--release-action audit`** hashes archived files and checks provenance;
it must be run after restoring the ignored artifact files. It does not
execute the OOS evaluator. The original scientific backup ZIP is optional
if absent from a separately unpacked distribution; the audit will note
its absence rather than claiming to verify it.

**`--release-action frozen-results`** reads archived evaluation metrics
after the audit. It does not recompute OOS forecasts, refit models or
backtest the final period.

**`--core-development`** recalculates the 2010-2021 development performance
using existing trained models/signals. This does run a development
backtest, but does not train agents or optimize the GA Orchestrator.

**Unit tests** cover pipeline behavior and release/development dispatch.
They do not certify macro data vintages or turn a historical simulation
into a live-executable strategy.

## Data distribution rules

Git tracks raw market/macro snapshots and source code. The following
local paths are not fully included in Git and must be restored through
the release artifact package: `data/processed/`, `data/signals/`,
`src/agents/config/*.joblib` and `reports/`. The packaging script checks
the 47 original frozen checksums, the raw-input audit and 13 release files
before collecting these paths. All archive member paths are relative to
the repository root.

Keep the original pre-OOS scientific backup
`ProjectAgentTrading26_frozen_20261107.zip` separately if long-term
provenance of the initial freeze is required. It predates later CLI/docs
commits. The consolidated `ProjectAgentTrading26_RELEASE_20261107.zip`
is the deliverable for code-and-artifacts restoration.

## What is and is not reproduced

**Reproduced:** archived strategy and benchmark NAV/metrics, checksum
checks, frozen development comparison, and local unit tests.

**Not claimed:** validation on true historical FRED macro vintages,
independently established live-trading profitability, or portability
to an environment with missing binary model dependencies.

The final OOS result was negative: CAGR 5.872% and Sharpe 0.551 for
Core35_Satellite65, below SPY (13.116%, 0.799) and 60/40 (7.388%,
0.694). Do not re-optimize the model using these test results.

## GitHub Actions limitation

`.github/workflows/unit-tests.yml` runs the ordinary Python unit suite
for pull requests. It prepares training-period data from tracked raw
sources, but does **not** install ignored frozen `.joblib` models or OOS
reports. Therefore a green PR build is **not** a full frozen-release
audit. That additional check must pass using the distributed release ZIP.

# Group 9 Release Structure and Integration Boundaries

**Scope:** `group_9.zip` packaged 13 November 2026. Project language: English.

## Entry points

- `main.py`: final multi-policy summary by default; the original frozen Core release remains available via `--release-action frozen-results`, `audit`, `verify-models` and `replay`.
- `scripts/evaluate_final_policies.py`: primary completed two-period analysis of Equal Active 4 ETFs and Regime-Adaptive SMA200, with the frozen Core reference.
- `final_policy_config.yaml`: explicit two-policy final research scope, separate from the immutable Core research contract.
- `src/development_policy_selection.py`: 2010-2021-only allocation-policy evaluation, with explicit cutoff and calendar checks.
- `scripts/run_strategy_comparison.py`: separate, opt-in original six-policy retrospective research. It imports frozen modules, never edits them, and is read-only by default; optional output goes outside `project/`.
- `src/research_strategy_comparison.py`: single shared research allocator, consistency checks, matched fee-aware execution and plots. This avoids duplicating Gabriele's experimental weekly backtester in several files.
- `src/agent_backtest.py`: **historically frozen** satellite-selection implementation, not overwritten by Equal Active experiments.
- `src/core_satellite.py`: **historically frozen** 35/65 policy. Research extensions call it but do not redefine the official selection.
- `src/orchestrator_loader.py`: original consistent JSON-to-agent loader retaining distinct long/short thresholds.
- `reports/`: unaltered 47-file and 13-artifact original scientific release, including provenance and OOS signals.
- `research_outputs/`: post-freeze matched comparison CSVs and figures; not a replacement for protected `reports/`.
- `Final_Project_Report.pdf`: scientific report, with the multi-policy conclusions retained in the completed supplemental policy-study documentation.
- `presentation.pptx` in the ZIP root and an identical copy in `project/`; no ZIP-relative path assumptions.

## Version-control and historical research boundaries

The distributed `project/` tree is the final, self-contained scientific
snapshot. The broader collaboration codebase is identified by Git commit
`4f532686e0e921cb3712fe4e0733eecfa3475f26`, with the team contributions
and earlier experimental implementations retained in version-control history.
The submission intentionally excludes duplicate weekly, zero-cost research
backtest implementations from the frozen production path. The authoritative
objects in this artifact are the archived freeze, saved OOS evidence, source
code, generated outputs and package manifest, not the name of a Git branch.

## Change acceptance policy

1. Freeze SHA256 map is authoritative for 47 code/model/signal assets and must match without normalization of comments.
2. Release-artifact index is authoritative for the 13 original OOS outputs and must not be rewritten.
3. New research comparisons have their own code, CSVs, provenance and plots; are labeled retrospective.
4. Cosmetics and nonessential refactoring cannot justify touching frozen scientific sources.
5. `PACKAGE_MANIFEST.json` is regenerated from every final file inside `project/`, distinct from the historical freeze.
6. Official and retrospective metrics are reconciled to their controlling saved CSV and NAV inputs in the verified final distribution.

## Unit test scope

The included minimal delivery tests are focused validation and regression checks; passing them does not imply the original 275-test development suite was executed on the package. The source-only GitHub CI is a different test context and does not verify the complete release archive. The frozen replay is required independently.

## Final study organization

The **two concluding allocation-policy alternatives** are Equal Active 4 ETFs
and Regime-Adaptive SMA200. The original Core35-Satellite65 policy is retained
as the historical reference that anchors the archive integrity checks. The
new development-only comparison uses 626 weekly decisions ending in 2021. The
post-2021 results use 249 weekly decisions from the saved historical period.
All model-training files are left unchanged: new allocation rules do not require
retraining the four original GA ensembles or their underlying RF/SVM agents.
The project distinguishes fitting cutoff, data-calendar causality, historical
selection records and post-hoc policy comparison rather than conflating them.

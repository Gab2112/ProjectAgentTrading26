"""Read-only release audit and archived OOS result viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hashes(mapping: dict, root: Path, label: str) -> None:
    """Verify that every listed file exists and matches its recorded SHA-256 hash."""
    for relative, expected in mapping.items():
        actual = digest(root / relative)
        if actual != expected:
            raise RuntimeError(
                f"{label}: integrity failure for {relative}"
            )


def verify_package_manifest(root: Path) -> int | None:
    """Guard all distributed files, including processed replay inputs.

    The manifest is generated only when the submission ZIP is assembled.
    Scientific Git repositories without the ZIP manifest remain supported.
    """
    manifest_path = root / 'PACKAGE_MANIFEST.json'
    if not manifest_path.exists():
        print('INFO: standalone Git tree; submission manifest not present.')
        return None
    manifest = read_json(manifest_path)
    mapping = manifest.get('files')
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError('Invalid submission manifest')
    for name, record in mapping.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise RuntimeError(f'Invalid submission path: {name}')
        if digest(path) != record['sha256']:
            raise RuntimeError(f'Submission manifest checksum mismatch: {name}')
        if path.stat().st_size != record['size_bytes']:
            raise RuntimeError(f'Submission manifest byte length mismatch: {name}')
    def part_of_release(path: Path) -> bool:
        relative = path.relative_to(root)
        ignored = {'.git', '.venv', '__pycache__', 'outputs', '.pytest_cache'}
        return not any(p in ignored for p in relative.parts)
    present = {
        p.relative_to(root).as_posix()
        for p in root.rglob('*')
        if p.is_file() and part_of_release(p) and p.name != 'PACKAGE_MANIFEST.json'
    }
    unexpected = present - set(mapping)
    if unexpected:
        raise RuntimeError('Unindexed files in submission: '+', '.join(sorted(unexpected)[:12]))
    print(f'OK: submission manifest: {len(mapping)} complete file checksums.')
    return len(mapping)


def audit_release(root: Path = ROOT) -> dict:
    """Verify the integrity of the frozen release without training or backtesting.

    Checks the development freeze record, the frozen code and model hashes, the
    raw input hashes, the archived release artifacts, the OOS signal and
    evaluation provenance, the optional submission manifest and the weekly
    decision calendars. Returns the release artifact index.
    """
    reports = root / "reports"

    freeze_path = reports / "development_freeze_2021.json"
    input_path = reports / "oos_input_audit.json"
    signals_path = reports / "oos_signals/inference_provenance.json"
    evaluation_path = (
        reports / "oos_final_2022_2026/evaluation_provenance.json"
    )
    index_path = reports / "release_artifact_index.json"

    freeze = read_json(freeze_path)
    inputs = read_json(input_path)
    signals = read_json(signals_path)
    evaluation = read_json(evaluation_path)
    index = read_json(index_path)

    if freeze["status"] != "FROZEN_BEFORE_FINAL_OOS":
        raise RuntimeError("Invalid development freeze")

    if freeze["selected_strategy"] != "Core35_Satellite65":
        raise RuntimeError("Unexpected selected strategy")

    if evaluation["status"] != "FINAL_OOS_EVALUATED_ONCE":
        raise RuntimeError("Final evaluation record missing")

    if index["record_type"] != "POST_OOS_RELEASE_ARCHIVE":
        raise RuntimeError("Invalid release index")

    if index["git_commit"] != "14ac63f" and not str(
        index["git_commit"]
    ).startswith("14ac63f"):
        raise RuntimeError("Unexpected original release commit")

    verify_hashes(
        freeze["code_and_model_sha256"],
        root,
        "Frozen models/code",
    )

    verify_hashes(
        inputs["raw_input_sha256"],
        root,
        "Raw inputs",
    )

    verify_hashes(
        index["artifacts"],
        root,
        "Archived results",
    )

    if digest(freeze_path) != signals["frozen_manifest_sha256"]:
        raise RuntimeError("Signal freeze provenance mismatch")

    if digest(freeze_path) != evaluation["frozen_manifest_sha256"]:
        raise RuntimeError("Evaluation freeze provenance mismatch")

    if digest(signals_path) != evaluation["signal_provenance_sha256"]:
        raise RuntimeError("Evaluation signal provenance mismatch")

    if digest(root / "src/frozen_oos_inference.py") != \
            signals["inference_code_sha256"]:
        raise RuntimeError("Inference code mismatch")

    if digest(root / "src/final_oos_backtest.py") != \
            evaluation["backtest_code_sha256"]:
        raise RuntimeError("OOS evaluation code mismatch")

    archive = root.parent / index["backup_archive"]

    if archive.exists():
        if digest(archive) != index["backup_sha256"]:
            raise RuntimeError("Backup archive checksum mismatch")
        print("OK: external ZIP backup verified.")
    else:
        print("INFO: external ZIP backup not present here.")

    verify_package_manifest(root)
    from src.submission_validation import verify_signal_calendar
    verify_signal_calendar(root, oos=False)
    verify_signal_calendar(root, oos=True)
    print("OK: complete development and OOS weekly decision calendars.")

    print(
        f"OK: {len(freeze['code_and_model_sha256'])} "
        "frozen files verified."
    )
    print("OK: historical raw inputs verified.")
    print(
        f"OK: {len(index['artifacts'])} "
        "release artifacts verified."
    )
    print("OK: OOS evaluation provenance verified.")
    print("OK: no training or backtesting executed.")

    return index


def show_frozen_results(root: Path = ROOT) -> None:
    """Audit the release and print the archived final OOS metrics (read-only)."""
    import pandas as pd

    audit_release(root)

    path = (
        root / "reports/oos_final_2022_2026"
        / "comparison_metrics.csv"
    )

    summary = pd.read_csv(path, index_col="strategy")

    strategies = [
        "Core35_Satellite65",
        "SPY_buy_hold",
        "Sixty_Forty",
    ]

    view = summary.loc[
        strategies,
        [
            "cagr",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "final_nav",
        ],
    ].copy()

    for col in ("cagr", "annualized_volatility", "max_drawdown"):
        view[col] *= 100.0

    view.columns = [
        "CAGR_%",
        "Volatility_%",
        "Sharpe",
        "MaxDD_%",
        "Final_NAV",
    ]

    print("\n=== ARCHIVED FINAL OOS RESULTS ===")
    print("Period: 2022-01-07 to 2026-10-09")
    print(view.round(3).to_string())

    print("\nREAD-ONLY: values loaded from archived CSV.")
    print("No model fitting, optimization or backtest performed.")


def main() -> None:
    """Command-line entry point for the audit and the archived results viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("audit", "frozen-results"),
    )

    args = parser.parse_args()

    if args.action == "audit":
        audit_release()
    elif args.action == "frozen-results":
        show_frozen_results()


if __name__ == "__main__":
    main()

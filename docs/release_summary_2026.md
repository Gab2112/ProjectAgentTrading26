# Pull Request Summary | Frozen Core-Satellite Research

## Scope

- Integrates frozen technical, macro and ML agent research with a
  Core35_Satellite65 final portfolio candidate.
- Preserves the 31 December 2021 development cutoff and one-shot
  protected OOS evaluation through 9 October 2026.
- Adds checksum-based release audit, archived OOS result display and
  development-only replay commands.
- Preserves historical A/B/C research documentation in an archived README.
- Adds a separate reproducibility guide and consolidated research package.

## Empirical finding

**No outperformance:** OOS Core35_Satellite65 CAGR 5.872%, Sharpe 0.551,
max drawdown -26.543%; SPY: 13.116%, 0.799, -23.424%; 60/40:
7.388%, 0.694, -19.840%. No post-OOS model retuning.

## Verification

- 47 frozen model/code/config/signal hashes verified.
- 13 release artifacts verified, with input and inference provenance.
- 272 unit tests passed locally with Python 3.11.
- Portable ZIP clean-reconstruction test to be recorded upon completion.
- GitHub Actions PR CI to be evaluated separately.

## Caveats

- The GA/strategy development scores are in-sample.
- Historic FRED macro vintages and exact announcement availability are
  not fully point-in-time certified.
- Fixed transaction and borrowing assumptions are not live broker quotes.
- Ignored model binaries and OOS reports are distributed in the release
  artifact ZIP, not in the Git commit.

## Review commands

```bash
python3.11 -m unittest discover -s tests -p 'test_*.py' -q
python3.11 main.py --release-action audit
python3.11 main.py --release-action frozen-results
python3.11 main.py --core-development
```

The latter three commands require the separately distributed scientific
artifacts. The first runs without the final OOS files when normal training
inputs are prepared by CI.

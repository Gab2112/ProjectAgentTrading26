# Decision Log

This file records only **material research and architecture decisions**. Routine coding changes are intentionally excluded.

---

## 2026-10-11 - Working research direction: Macro / Rates Regimes

**Decision**

Use a U.S. macro / rates regime framework as the working research direction.

**Research question**

> Can combinations of inflation, interest rates, yield-curve shape and rate volatility identify recurring regimes associated with different fixed-income exposures?

**Status**

Working hypothesis, not irrevocably final.

---

## 2026-10-11 - U.S.-only (atm)

**Decision**

Restrict the first implementation to U.S. fixed income.

---

## 2026-10-11 - Traded universe: SHY / IEF / TLT

**Decision**

- SHY for short duration;
- IEF for intermediate duration;
- TLT for long duration.

---

## 2026-10-11 - Weekly strategy frequency

**Decision**

Use daily raw market data, macro data at native frequency, and a weekly strategy / weekly rebalance.

---

## 2026-10-11 - Local raw data retained in repository

**Decision**

Keep downloaded raw market and macro data in `data/raw/`.

`main.py` skips re-downloading by default.

**Rationale**

The final project must be reproducible from repository contents without requiring a fresh download.

---

## 2026-10-11 - Publication-lag treatment for macro data

**Decision**

```text
CPI       45 calendar days
Core CPI  45 calendar days
FEDFUNDS  35 calendar days
```

**Known limitation**

Current FRED histories may contain later revisions.

**Future action**

Evaluate ALFRED / point-in-time vintage data or exact historical release calendars before final OOS evaluation if feasible.

---

## 2026-10-11 - CPI YoY calculated on monthly raw data

**Decision**

Calculate CPI YoY and Core CPI YoY using a true 12-month comparison on the original monthly series, then apply publication lag and weekly alignment.

---

## 2026-10-11 - Deterministic four-dimensional regime baseline

**Decision**

```text
inflation     rising / falling
rates         rising / falling
curve         normal / inverted
volatility    high / low
```

---

## 2026-10-11 - Regime thresholds

**Decision**

- inflation trend window: 13 weeks;
- rate trend window: 13 weeks;
- curve inversion threshold: 10Y - 2Y = 0;
- volatility threshold: cumulative historical median using at least 52 observations and shifted by one week.

---

## 2026-10-11 - Deterministic duration score

**Decision**

```text
rates falling      +2
rates rising       -2
inflation falling  +1
inflation rising   -1
curve inverted     +1
curve normal        0
```

High volatility moves the score one point toward zero.

**Status**

Baseline only. Not a claim of optimal weights.

---

## 2026-10-11 - Baseline signal thresholds

**Decision**

```text
score >= +2 -> long-duration regime
score <= -2 -> short-duration regime
otherwise   -> intermediate-duration regime
```

Asset mapping:

```text
long duration         -> TLT +1
intermediate duration -> IEF +1
short duration        -> SHY +1 and TLT -1
```

---

## 2026-10-11 - Portfolio allocation

**Decision**

Use `equal_active_signal` with maximum gross exposure of 1.0 and no additional leverage.

Example:

```text
SHY +1 and TLT -1
-> SHY +0.50
-> TLT -0.50
```

---

## 2026-10-11 - Execution timing

**Decision**

Apply target weights with a one-week lag.

```text
signal at t
-> target weights at t
-> applied position at t+1
```

---

## 2026-10-11 - Validation split

**Decision**

```text
TRAIN
2005-05-27 -> 2016-12-30

VALIDATION
2017-01-06 -> 2021-12-31

OOS
2022-01-01 onward
```

**Critical rule**

OOS remains locked until the model and relevant design choices are frozen.

---

## 2026-10-11 - Deterministic baseline retained despite weak performance

**Decision**

Keep the current deterministic economic system as the official Stage 1 baseline even though it underperforms the fixed-income benchmark in train and validation.

**Rationale**

Its purpose is comparison and interpretability, not retroactive optimization.

---

## 2026-10-11 - Agent layer deferred

**Decision**

Keep agents disabled until the quantitative core and at least one forecasting / ML model are established.

**Rationale**

Deterministic calculations should remain deterministic. Agents may later add value in interpretation, coordination, critique or decision aggregation.

---

## Next decision point

Before implementing `forecasting.py`, define:

1. the prediction target;
2. the prediction horizon;
3. whether Stage 2 is regression or classification;
4. the exact training/validation procedure for model selection.

No OOS data should be used in making these decisions.

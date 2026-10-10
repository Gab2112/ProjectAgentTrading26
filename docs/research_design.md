# Research Design

## LUISS Final Project — v0

**Course:** Data-Driven Models for Investment  
**Status:** Early research-design phase  
**Purpose of this document:** Define the first coherent research framework.

---

## 1. Project Objective

The project aims to develop and rigorously evaluate a **systematic trading strategy** driven by macro-financial information, with a current focus on:

- monetary policy;
- interest rates;
- inflation;
- yield-curve dynamics;
- fixed-income markets.

The final strategy must convert observable information into discrete trading decisions:

```text
+1 = Long
 0 = Flat / Neutral
-1 = Short
```

The project will be developed progressively, starting from a simple and interpretable baseline and introducing forecasting and machine-learning methods only when they can be tested against that baseline.

The central methodological principle is:

> **Increasing model complexity must justify itself through better out-of-sample performance, robustness, or economic interpretability.**

---

## 2. Current Research Direction

The project is currently moving toward a **monetary-policy / rates / fixed-income framework**.

The exact research question is intentionally not fixed yet.

The current area of interest includes relationships among:

```text
Policy rates
Inflation
Treasury / government-bond yields
Yield-curve shape
Real rates
Bond prices
Market volatility
```

Possible traded instruments may include government-bond ETFs, bond futures, or other liquid assets whose prices are directly affected by changes in interest-rate expectations.

At this stage, the project is deliberately broader than a single final strategy. The next research step is to identify a small set of precise, falsifiable hypotheses.

---

## 3. Research Philosophy

The project should not begin by choosing a sophisticated model.

The sequence should be:

```text
     Economic idea
           ↓
   Research hypothesis
           ↓
         Data
           ↓
       Features
           ↓
    Simple baseline
           ↓
   Forecasting model
           ↓
    Trading signal
           ↓
       Backtest
           ↓
      Validation
```

The model must serve the hypothesis, not the reverse.

Only after defining the prediction problem should the appropriate model be selected.

---

## 4. Candidate Research Questions

The following questions are currently examples, not final decisions.

### Candidate A — Bond Direction Forecasting

> Can changes in inflation, policy rates and the yield curve predict short-horizon returns across different bond maturities?

Possible output:

```text
Expected positive return  → +1
No strong expectation     →  0
Expected negative return  → -1
```

---

### Candidate B — Duration Allocation

> Can macroeconomic and interest-rate conditions determine when long-, intermediate-, or short-duration bond exposure is preferable?

Possible universe:

```text
Short-duration bonds
Intermediate-duration bonds
Long-duration bonds
```

The strategy could change exposure according to the predicted interest-rate environment.

---

### Candidate C — Yield Forecasting

> Does forecasting future changes in Treasury yields improve a systematic bond-trading strategy relative to simple rules?

Conceptually:

```text
Macro / rates features
        ↓
Yield forecasting model
        ↓
Expected yield change
        ↓
Bond trading signal
```

Because bond prices and yields move in opposite directions, a forecast of rising yields would generally imply pressure on bond prices, while falling yields would generally support bond prices.

---

### Candidate D — Macro / Rates Regimes -- `Chosen one`

> Can combinations of inflation, rates, yield-curve shape and volatility identify recurring regimes with different optimal fixed-income exposures?

Possible regime examples:

```text
Inflation rising / yields rising
Inflation falling / yields falling
Curve steepening
Curve flattening
High-rate volatility
Low-rate volatility
```

A later model could attempt to classify or forecast these regimes.

---

## 5. Candidate Asset Universe

The exact traded universe is still open.

Potential instruments may include:

- U.S. Treasury ETFs;
- short-, intermediate-, and long-duration bond ETFs;
- Treasury futures;
- European government-bond instruments;
- additional liquid assets used as benchmarks or cross-asset signals.

A multi-asset structure is preferred because it allows the system to compare different sensitivities to interest-rate and inflation dynamics.

The final universe should satisfy three conditions:

1. clear economic connection to the hypothesis;
2. sufficient historical data;
3. realistic tradability and benchmark construction.

---

## 6. Candidate Data

The first data architecture may include several groups of variables.

### 6.1 Market Data

Possible inputs:

- bond prices;
- bond ETF prices;
- Treasury yields;
- futures prices;
- market returns;
- trading volume;
- volatility measures.

---

### 6.2 Interest-Rate Data

Possible inputs:

- policy rates;
- 3-month yields;
- 2-year yields;
- 5-year yields;
- 10-year yields;
- 30-year yields.

These allow the construction of yield-curve features.

---

### 6.3 Yield-Curve Features

Examples:

```text
10Y - 2Y
10Y - 3M
30Y - 10Y
```

Possible additional features:

- slope;
- change in slope;
- steepening / flattening;
- inversion indicators.

---

### 6.4 Inflation Data

Possible variables:

- CPI;
- core CPI;
- inflation changes;
- inflation surprises;
- breakeven inflation;
- inflation expectations.

The distinction between low-frequency official inflation data and higher-frequency market-implied inflation expectations will need to be handled carefully.

---

### 6.5 Real Rates

A simplified real-rate concept is:

```text
Nominal yield
-
Expected inflation
=
Real yield
```

Real rates may be particularly informative for fixed-income and cross-asset behavior.

---

### 6.6 Additional Macro-Financial Variables

Possible later candidates:

- rate volatility;
- equity volatility;
- credit spreads;
- USD strength;
- economic-surprise indicators;
- central-bank expectations.

These should be included only if they have a clear economic role.

---

## 7. Proposed Architecture v0

The first target architecture is:

```text
ProjectAgentTrading26/
│
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
│
├── docs/
│   └── research_design.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── signals/
│   └── results/
│
└── src/
    ├── data_acquisition.py
    ├── data_transformation.py
    ├── feature_engineering.py
    ├── forecasting.py
    ├── signal_generation.py
    ├── backtester.py
    └── evaluator.py
```

The current repository already contains a functioning subset of this pipeline. The new structure should evolve gradually rather than through a complete rewrite.

---

## 8. Functional Pipeline

The target pipeline is:

```text
         RAW DATA
             ↓
 DATA CLEANING AND ALIGNMENT
             ↓
     FEATURE ENGINEERING
             ↓
       BASELINE MODEL
             ↓
    FORECASTING / ML MODEL
             ↓
      SIGNAL GENERATION
             ↓
        +1 / 0 / -1
             ↓
          BACKTEST
             ↓
TRAIN / VALIDATION / OUT-OF-SAMPLE
             ↓
   PERFORMANCE EVALUATION
```

Each stage should have a clear responsibility.

---

## 9. Module Responsibilities

### `data_acquisition.py`

Responsible for obtaining raw market and macro-financial data.

Future sources may include:

- Bloomberg;
- Yahoo Finance;
- FRED;
- central-bank databases;
- other validated financial-data sources.

The module should not perform feature engineering.

---

### `data_transformation.py`

Responsible for:

- date alignment;
- frequency alignment;
- missing values;
- data cleaning;
- chronological consistency;
- combining series from different sources.

This stage becomes especially important because macroeconomic variables may have different frequencies and release schedules.

---

### `feature_engineering.py`

Responsible for creating model inputs.

Possible examples:

```text
yield changes
yield-curve slopes
curve steepening / flattening
inflation changes
real-rate proxies
momentum
realized volatility
rolling statistics
```

Features must only use information that would have been available at the time.

---

### `forecasting.py`

Responsible for predictive models.

This module should support increasing complexity.

Possible development path:

```text
   Simple statistical model
              ↓
  Linear / Logistic Regression
              ↓
        Random Forest
              ↓
           Ensemble
              ↓
Neural Network, only if justified
```

Not every model needs to survive into the final project.

The purpose is comparison, not complexity for its own sake.

---

### `signal_generation.py`

Responsible for converting model outputs into:

```text
+1 = Long
 0 = Flat
-1 = Short
```

Example:

```text
Forecast probability > upper threshold
→ +1

Forecast probability < lower threshold
→ -1

Otherwise
→ 0
```

The exact rule will depend on the final target variable.

---

### `backtester.py`

Responsible for:

- applying positions;
- execution timing;
- strategy returns;
- transaction costs;
- slippage;
- equity curves;
- benchmarks.

The backtester should remain independent from any specific forecasting model.

---

### `evaluator.py`

Responsible for comparing models and strategies using metrics such as:

- cumulative return;
- CAGR (_Compound Annual Growth Rate_);
- annualized volatility;
- Sharpe ratio;
- maximum drawdown;
- profit factor;
- turnover;
- number of trades.

---

## 10. Development Strategy: Increasing Complexity

The project should progress through several layers.

### Stage 0 — Technical Baseline

The current SMA strategy remains useful as a technical proof of concept.

Its purpose is to verify that:

```text
data → feature → signal → backtest → evaluation
```

works end to end.

It is not the final research strategy.

---

### Stage 1 — Economic Baseline

Build a simple strategy based on the final macro / rates hypothesis.

Example:

```text
    Yield-curve change
            +
     Inflation trend
            +
        Rate trend
            ↓
  Simple deterministic rule
            ↓
          Signal
```

This gives an interpretable benchmark.

---

### Stage 2 — Simple Forecasting

Introduce a statistical forecasting model.

Examples:

- linear regression;
- logistic regression.

The purpose is to determine whether a fitted predictive relationship improves on the simple baseline.

---

### Stage 3 — Machine Learning

Introduce nonlinear models such as:

- Random Forest;
- gradient boosting;
- ensemble methods.

The key comparison becomes:

```text
 Simple baseline
       vs
   Regression
       vs
 Machine learning
```

---

### Stage 4 — Neural Models

A neural network should only be introduced if:

- the dataset is sufficiently large;
- the problem justifies nonlinear complexity;
- simpler models have already been established;
- strict validation is maintained.

A neural model is an optional extension, not a project requirement.

---

## 11. Validation Framework

The current train/test structure is not sufficient for the final project.

The target structure should be:

```text
        TRAIN
          ↓
      VALIDATION
          ↓
MODEL / PARAMETER FREEZE
          ↓
   OUT-OF-SAMPLE TEST
```

### Training

Used to estimate model parameters.

### Validation

Used to:

- choose features;
- select models;
- tune hyperparameters;
- choose thresholds.

### Out-of-Sample

Used only for final evaluation.

The OOS period must not influence model design.

---

## 12. Model Comparison

The project should not evaluate a model only in isolation.

The final analysis should answer questions such as:

```text
Does forecasting beat a simple rule?

Does Random Forest beat regression?

Does an ensemble improve robustness?

Does additional complexity survive out of sample?
```

Possible comparison table:

```text
Model                  Return   Sharpe   Max DD
------------------------------------------------
Simple baseline          ...
Regression               ...
Random Forest            ...
Ensemble                 ...
Benchmark                ...
```

This incremental comparison may become one of the central strengths of the final project.

---

## 13. Execution and Look-Ahead Control

All signals must be based only on information available at the time.

A generic daily rule could be:

```text
   Information available through day t
                 ↓
      Features calculated at t
                 ↓
       Forecast generated at t
                 ↓
        Signal generated at t
                 ↓
      Position applied from t+1
```

For macroeconomic data, publication dates are especially important.

For example, an inflation observation referring to one month may only be released several weeks later. The strategy must use the **release date**, not the period to which the statistic refers.

---

## 14. Open Decisions

The following decisions are intentionally still open:

- final research question;
- final traded assets;
- prediction horizon;
- target variable;
- exact macroeconomic features;
- exact historical sample;
- final data sources;
- baseline trading rule;
- forecasting models;
- hyperparameter ranges;
- signal thresholds;
- benchmark portfolio;
- transaction-cost assumptions.

These items should be resolved progressively rather than fixed prematurely.

---

## 15. Current Status

```text
Repository baseline tested         ✓
Current code inspected             ✓
Initial research architecture      ✓
Final hypothesis                   OPEN
Final asset universe               OPEN
Final dataset                      OPEN
Final forecasting model            OPEN
```

The next step is **???**

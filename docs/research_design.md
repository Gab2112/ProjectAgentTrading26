# Research Design

## LUISS Final Project — v1

**Course:** Data-Driven Models for Investment  
**Status:** Working research architecture  
**Working hypothesis:** Macro / Rates Regimes  
**Purpose of this document:** Define the current research question, expected software architecture, methodological constraints, and staged development plan.

---

## 1. Project Objective

The project aims to develop and rigorously evaluate a **systematic fixed-income trading system** driven by macro-financial information.
The current working focus is:

- monetary policy;
- interest rates;
- inflation;
- yield-curve dynamics;
- fixed-income markets;
- volatility and market-regime dynamics.

The system must ultimately convert information available at time t into discrete trading decisions:

```text
+1 = Long
 0 = Flat / Neutral
-1 = Short
```

The project will be developed progressively. A simple and interpretable economic baseline will be implemented first. Statistical forecasting, machine-learning models, and an agentic extension will be introduced only after the baseline and validation framework are stable.

The central methodological principle is:

> **Increasing model or system complexity must justify itself through better out-of-sample performance, robustness, interpretability, or decision quality.**

---

## 2. Working Research Question

The current working research question is:

> **Can combinations of inflation, interest rates, yield-curve shape, and volatility identify recurring macro/rates regimes associated with different optimal fixed-income exposures?**



It is not treated as fixed and irreversible. If later evidence or group's preferences suggests a better research question, the domain-specific components can be revised while preserving the reusable research infrastructure.

The project therefore distinguishes between:

### Reusable infrastructure

```text
data acquisition
data cleaning and alignment
time-series validation
backtesting
performance evaluation
configuration
results management
agent orchestration interfaces
```

### Hypothesis-specific components

```text
asset universe
regime definition
feature set
prediction target
forecast horizon
model choice
signal mapping
portfolio allocation logic
```

---

## 3. Economic Intuition

Fixed-income assets react differently depending on the prevailing macroeconomic and rates environment.

Possible regime dimensions include:

```text
Inflation rising / falling
Policy rates rising / falling
Yield levels rising / falling
Curve steepening / flattening
Curve inversion / normalization
Rate volatility high / low
```

The underlying hypothesis is that these states may correspond to systematically different risk-return profiles across bond maturities.

For example, long-duration bonds may react differently from short-duration bonds when:

- inflation accelerates;
- expected policy rates increase;
- the yield curve flattens;
- rate volatility rises.

The goal is not merely to label economic states. The regime representation must eventually support a tradable allocation rule.

---

## 4. Research Philosophy

The project should not begin by selecting a sophisticated model.

The intended sequence is:

```text
Economic hypothesis
        ↓
Point-in-time data
        ↓
Feature engineering
        ↓
Regime representation
        ↓
Simple economic baseline
        ↓
Statistical / ML model
        ↓
Agentic extension
        ↓
Trading signal / allocation
        ↓
    Backtest
        ↓
   Validation
```

The model must serve the hypothesis, not the reverse.

Likewise, the agentic system must serve a clear decision problem.

---

## 5. Candidate Asset Universe

The exact traded universe is still open.

Potential instruments include:

- U.S. Treasury ETFs;
- short-duration Treasury ETFs;
- intermediate-duration Treasury ETFs;
- long-duration Treasury ETFs;
- Treasury futures;
- European government-bond instruments;
- additional liquid fixed-income instruments used as benchmarks or comparative exposures.

A multi-asset structure is preferred because the research question explicitly concerns **different fixed-income exposures across regimes**.

The final universe should satisfy:

1. clear economic connection to the hypothesis;
2. sufficient historical depth;
3. realistic tradability;
4. enough variation in duration / rate sensitivity;
5. reliable benchmark construction.

---

## 6. Candidate Data

### 6.1 Market Data

Possible inputs:

- bond / ETF prices;
- Treasury futures prices;
- returns;
- trading volume;
- realized volatility;
- rate-volatility measures.

### 6.2 Interest-Rate Data

Possible inputs:

- policy rates;
- 3-month yields;
- 2-year yields;
- 5-year yields;
- 10-year yields;
- 30-year yields.

### 6.3 Yield-Curve Features

Examples:

```text
10Y - 2Y
10Y - 3M
30Y - 10Y
```

Possible derived features:

- slope;
- change in slope;
- steepening / flattening;
- inversion indicators;
- level / slope / curvature representations.

### 6.4 Inflation Data

Possible inputs:

- CPI;
- core CPI;
- inflation changes;
- inflation surprises;
- breakeven inflation;
- inflation expectations.

Official inflation releases and market-implied inflation measures must be treated differently because they have different publication frequencies and timing.

### 6.5 Real Rates

A simplified representation is:

```text
  Nominal yield
       -
Expected inflation
       =
   Real yield
```

### 6.6 Additional Macro-Financial Variables

Possible later candidates:

- rate volatility;
- equity volatility;
- credit spreads;
- USD strength;
- economic-surprise indicators;
- central-bank expectations.

Variables should be included only when their economic role can be justified.

---

## 7. Proposed Architecture v1

The repository should evolve toward:

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
    ├── regime_model.py
    ├── forecasting.py
    ├── signal_generation.py
    ├── portfolio.py
    ├── backtester.py
    ├── evaluator.py
    │
    └── agents/
        ├── __init__.py
        ├── orchestrator.py
        ├── macro_agent.py
        ├── regime_agent.py
        ├── strategy_agent.py
        ├── risk_agent.py
        └── critic_agent.py
```

This is a **target architecture**, not a requirement to implement every module immediately.

The deterministic research engine comes first. The agent layer is an extension built on top of that engine.

---

## 8. Two-Layer System Design

The architecture deliberately separates the project into two layers.

### Layer A — Deterministic Quantitative Core

```text
       RAW DATA
          ↓
  CLEANING / ALIGNMENT
          ↓
  FEATURE ENGINEERING
          ↓
     REGIME MODEL
          ↓
   FORECASTING / ML
          ↓
SIGNAL / PORTFOLIO RULE
          ↓
      BACKTEST
          ↓
     EVALUATION
```

This layer must be independently runnable and reproducible.

It provides:

- numerical features;
- model outputs;
- regime probabilities / labels;
- candidate positions;
- historical performance.

### Layer B — Agentic Decision Layer

The agentic layer consumes structured, point-in-time information produced by the quantitative core.

```text
                 ORCHESTRATOR
                      │
       ┌──────────────┼──────────────┐
       │              │              │
   Macro Agent    Regime Agent   Strategy Agent
       │              │              │
       └──────────────┼──────────────┘
                      ↓
                  Risk Agent
                      ↓
                 Critic Agent
                      ↓
               Final Decision
                      ↓
                 +1 / 0 / -1
```

The agent layer must not bypass the backtester or validation framework.

---

## 9. Agent Responsibilities

The exact agent design remains provisional.

### 9.1 Orchestrator Agent

Coordinates the agentic workflow.

Responsibilities may include:

- deciding which specialized agents should be called;
- passing structured information between agents;
- managing dependencies;
- collecting outputs;
- resolving missing or inconsistent outputs;
- producing the final structured decision package.

The orchestrator should not directly invent market data or overwrite deterministic model outputs.

### 9.2 Macro Agent

Interprets the macroeconomic state using only approved point-in-time inputs.

Possible inputs:

- inflation trend;
- policy-rate state;
- yield levels;
- real-rate proxies;
- macro surprises.

Possible output:

```text
macro_state
confidence
supporting_variables
```

### 9.3 Regime Agent

Consumes quantitative regime outputs and evaluates the current rate-regime interpretation.

Possible inputs:

- cluster / regime label;
- regime probabilities;
- yield-curve state;
- volatility state;
- transition probabilities.

Possible output:

```text
regime_interpretation
regime_confidence
transition_risk
```

This agent should complement, not replace, the underlying quantitative regime model.

### 9.4 Strategy Agent

Maps regime information into candidate fixed-income exposures.

Possible responsibilities:

- compare short-, intermediate-, and long-duration exposures;
- convert regime information into candidate directional positions;
- explain why a given exposure is consistent with the regime.

Possible output:

```text
candidate_position
candidate_weight
rationale
confidence
```

### 9.5 Risk Agent

Applies risk constraints independently from the Strategy Agent.

Possible checks:

- excessive exposure;
- volatility threshold;
- concentration;
- leverage constraints;
- turnover;
- drawdown / risk regime;
- conflicting signals.

The Risk Agent may reduce or reject a proposed position.

### 9.6 Critic Agent

Acts as an evaluator / challenger before the final decision.

Possible responsibilities:

- check whether the recommendation is supported by available data;
- detect inconsistencies between agents;
- identify unsupported reasoning;
- flag possible look-ahead or information-timing problems;
- request reconsideration when necessary.

The Critic Agent provides a controlled evaluator–optimizer mechanism rather than unrestricted debate.

---

## 10. Why the Agent Layer Is Not Implemented First

The project should first establish:

```text
reliable data
        ↓
point-in-time features
        ↓
valid regime representation
        ↓
baseline strategy
        ↓
proper backtest
```

Only then does it make sense to introduce autonomous decision components.

Otherwise, the agents would be reasoning over an unstable research pipeline and their contribution could not be measured cleanly.

The agentic extension therefore becomes a **later experimental layer**, not the initial implementation step.

---

## 11. Agentic Design Principles

### 11.1 Use Agents Only Where Agency Adds Value

Tasks with fixed and testable logic should remain deterministic.

Examples:

```text
return calculation
yield-curve spread calculation
data alignment
transaction-cost calculation
performance metrics
```

Agents are more appropriate for:

```text
context synthesis
specialized reasoning
conflict resolution
tool selection
structured critique
orchestration
```

### 11.2 Structured Inputs and Outputs

Agents should communicate through explicit schemas rather than unconstrained prose.

Example:

```json
{
  "regime": "flattening_high_volatility",
  "confidence": 0.72,
  "recommended_signal": -1,
  "risk_flag": false
}
```

### 11.3 Agent Output Must Be Testable

The project should compare:

```text
 Quantitative baseline
         vs
  ML-enhanced strategy
         vs
Agent-enhanced strategy
```

The agentic layer is useful only if its incremental contribution can be evaluated.

### 11.4 No Direct Access to Future Information

For historical backtests, every agent invocation must receive only information available at that historical timestamp.

### 11.5 Pretrained-Model Leakage Must Be Addressed

If LLM-based agents are used, the model may contain information in its pretrained weights about events occurring after the historical decision date.

Therefore the final methodology must explicitly discuss and mitigate pretrained-model leakage.

Potential safeguards include:

- restrict prompts to structured point-in-time inputs;
- avoid asking the LLM to recall historical events from memory;
- avoid giving future outcome information;
- log complete prompts and responses;
- test an agent-free quantitative benchmark;
- treat LLM-agent results as a distinct experimental layer.

---

## 12. Core Module Responsibilities

### `data_acquisition.py`

Obtains raw financial and macroeconomic data.

Potential sources:

- Bloomberg;
- Yahoo Finance;
- FRED;
- central-bank databases;
- other validated sources.

### `data_transformation.py`

Responsible for:

- date alignment;
- frequency alignment;
- missing values;
- chronological consistency;
- point-in-time release handling;
- combining series from different sources.

### `feature_engineering.py`

Creates model inputs such as:

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

### `regime_model.py`

Dedicated to regime identification.

Potential approaches may eventually include:

- deterministic regime rules;
- clustering;
- Gaussian mixture models;
- Hidden Markov Models;
- other state-classification techniques.

No final model is selected yet.

### `forecasting.py`

Handles models that forecast:

- future regime probabilities;
- regime transitions;
- returns;
- other intermediate variables if justified.

Potential development path:

```text
   Simple statistical model
              ↓
 Linear / Logistic Regression
              ↓
        Random Forest
              ↓
  Gradient Boosting / Ensemble
              ↓
 Neural model, only if justified
```

### `signal_generation.py`

Converts regime / forecasting outputs into:

```text
+1 = Long
 0 = Flat
-1 = Short
```

### `portfolio.py`

Maps asset-level signals into portfolio exposures.

This is especially important if several bond maturities are traded simultaneously.

### `backtester.py`

Responsible for:

- position application;
- execution timing;
- strategy returns;
- transaction costs;
- slippage;
- equity curves;
- benchmarks.

### `evaluator.py`

Computes and compares:

- cumulative return;
- annualized return / CAGR;
- annualized volatility;
- Sharpe ratio;
- maximum drawdown;
- profit factor;
- turnover;
- number of trades.

---

## 13. Development Roadmap

### Stage 0 — Existing Technical Baseline

The current SMA implementation has already verified that:

```text
data → feature → signal → backtest → evaluation
```

works end to end.

It is not part of the intended final strategy.

### Stage 1 — Macro / Rates Data Foundation

Implement:

- selected asset universe;
- yield data;
- inflation data;
- policy-rate data;
- volatility data;
- robust timestamp alignment.

### Stage 2 — Economic Regime Baseline

Create a simple and interpretable regime definition.

Example conceptually:

```text
     inflation trend
            +
       yield trend
            +
       curve shape
            +
      rate volatility
            ↓
simple regime classification
            ↓
   fixed-income exposure
```

This becomes the first meaningful benchmark.

### Stage 3 — Data-Driven Regime Model

Replace or complement deterministic regime rules with a statistical / ML regime model.

Possible approaches:

- clustering;
- HMM;
- classification models;
- ensemble methods.

### Stage 4 — Regime Forecasting

Test whether forecasting the next regime or regime transition adds value beyond contemporaneous regime identification.

### Stage 5 — Agentic Extension

Introduce the specialized agents and orchestrator.

The agentic system should be evaluated against the non-agentic strategy.

Possible comparison:

```text
Economic baseline
vs
Data-driven regime model
vs
Forecast-enhanced model
vs
Agent-enhanced system
```

### Stage 6 — Optional Advanced Models

Only if justified:

- neural networks;
- more complex ensembles;
- additional alternative data;
- more sophisticated multi-agent debate / consensus.

---

## 14. Validation Framework

The target structure is:

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

- select features;
- select regime methodology;
- tune hyperparameters;
- choose signal thresholds;
- compare candidate agent configurations if necessary.

### Out-of-Sample

Used only for final evaluation.

The OOS period must not influence model design.

Walk-forward validation may later be added if appropriate.

---

## 15. Model and System Comparison

The final analysis should not report only one model.

The project should evaluate the incremental contribution of each layer.

Example:

```text
System                         Return   Sharpe   Max DD
------------------------------------------------------
Economic regime baseline        ...
Statistical regime model        ...
ML regime model                 ...
Forecast-enhanced system        ...
Agent-enhanced system           ...
Benchmark                       ...
```

This structure allows the project to answer not only whether the final system works, but **which additional components actually add value**.

---

## 16. Execution and Look-Ahead Control

All signals must use only information available at the decision time.

A generic workflow is:

```text
Information available through t
            ↓
Point-in-time feature set
            ↓
Regime / forecast output
            ↓
Agentic analysis, if enabled
            ↓
Signal generated at t
            ↓
Position applied from t+1
```

For macroeconomic data, publication dates are essential.

An inflation value referring to month m cannot be used until its actual publication date.

The same point-in-time rule applies to every agent input.

---

## 17. Current Working Decisions

The following are now treated as working decisions:

```text
Domain                  Macro / rates / fixed income
Research candidate      Candidate D — Macro / Rates Regimes
System type             Multi-asset systematic trading
Core architecture       Quantitative engine + later agentic layer
Signal convention       +1 / 0 / -1
Validation philosophy   Train / validation / OOS
Development philosophy  Baseline first, complexity incrementally
Agent implementation    Required in expected architecture, later stage
```

---

## 18. Open Decisions

The following remain open:

- exact traded assets;
- U.S. vs European vs mixed market scope;
- prediction / holding horizon;
- precise regime definition;
- exact historical sample;
- exact macroeconomic features;
- final data sources;
- baseline exposure rule;
- regime-identification model;
- forecasting models;
- portfolio construction method;
- agent framework;
- exact agent count;
- LLM-based vs tool-based agents;
- agent consensus / orchestration method;
- signal thresholds;
- benchmark portfolio;
- transaction-cost assumptions.

These decisions should be resolved progressively.

---

## 19. Current Status

```text
Repository baseline tested               ✓
Current code inspected                   ✓
Research architecture defined            ✓
Working research candidate selected      ✓
Agentic layer included in architecture   ✓

Final asset universe                     OPEN
Final dataset                            OPEN
Regime definition                        OPEN
Baseline exposure rule                   OPEN
Final model                              OPEN
Agent implementation details             OPEN
```

---

## 20. Immediate Next Step

The next step is to define the **minimum viable specification of Candidate D** before changing the code.

The next design decision should specify four items:

```text
1. Market scope
   Example: U.S. rates first

2. Traded universe
   Example structure: short / intermediate / long duration

3. Initial regime dimensions
   Inflation + yield level + curve shape + rate volatility

4. Initial time scale
   Daily, weekly, or mixed-frequency framework
```

Once these four items are fixed provisionally, the repository can be refactored around a concrete v1 implementation without locking the project permanently to one final model.

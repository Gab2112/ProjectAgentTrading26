"""Trading agents of ProjectAgentTrading26.

Exports:
- BaseAgent: common abstract class with the standard ``predict()`` interface and the {-1, 0, 1} signal convention.
- SMAEMAAgent (alias EMA_EMW_Agent): trend-following agent based on an SMA/EMA crossover.
- SMAEMAOptimizer, WalkForwardTrainer: optimisation components for the SMA/EMA agent.
- ZScoreAgent: mean-reversion agent with hysteresis-based exit logic.
- ZScoreOptimizer, ZScoreWalkForwardTrainer: optimisation components for the Z-Score agent.
- MacroYieldCurveAgent: macroeconomic agent based on the 10Y-2Y curve slope, regimes and inflation.
- FinBERTAgent: sentiment agent based on the FinBERT transformer (lazily loaded).
- download_news_dataset: acquisition and structuring of the financial news dataset.
- SVMAgent: Support Vector Machine agent (annual walk-forward with purging, standardised decision function).
- SVMOptimizer, SVMWalkForwardTrainer: hyperparameter selection with temporal CV and out-of-fold signals.
- OrchestratorAgent: quantitative orchestrator that combines the agent signals.
- OrchestratorOptimizer: optimisation of the Orchestrator weights and decision thresholds.
- WalkForwardOrchestrator: rolling walk-forward validation of the Orchestrator.

Data, features and metrics are not defined here: see ``src/data_loader.py``,
``src/features.py`` and ``src/evaluator.py`` (``calculate_performance_metrics``,
``compute_objective_score``).
"""

from .base_agent import BaseAgent
from .EMA_EMW_agent import EMA_EMW_Agent, SMAEMAAgent, SMAEMAOptimizer, WalkForwardTrainer
from .ZScore_agent import ZScoreAgent, ZScoreOptimizer, ZScoreWalkForwardTrainer
from .macro_yield_curve_agent import MacroYieldCurveAgent
from .SVM_agent import SVMAgent, SVMOptimizer, SVMWalkForwardTrainer
from .orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorOptimizer,
    WalkForwardOrchestrator,
)


def __getattr__(name: str):
    """Lazily import components with a high import overhead (e.g. torch / transformers).
    """
    if name in ("FinBERTAgent", "download_news_dataset"):
        try:
            from .FinBERT_agent import FinBERTAgent as _FBAgent, download_news_dataset as _dn_dataset
            if name == "FinBERTAgent":
                return _FBAgent
            return _dn_dataset
        except ImportError:
            return None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseAgent",
    "SMAEMAAgent",
    "EMA_EMW_Agent",
    "SMAEMAOptimizer",
    "WalkForwardTrainer",
    "ZScoreAgent",
    "ZScoreOptimizer",
    "ZScoreWalkForwardTrainer",
    "MacroYieldCurveAgent",
    "SVMAgent",
    "SVMOptimizer",
    "SVMWalkForwardTrainer",
    "FinBERTAgent",
    "download_news_dataset",
    "OrchestratorAgent",
    "OrchestratorOptimizer",
    "WalkForwardOrchestrator",
]

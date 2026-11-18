"""Trading agents of ProjectAgentTrading26.

Exports:
- BaseAgent: abstract class with the ``predict()`` interface and the {-1, 0, 1} signal convention.
- SMAEMAAgent: trend-following agent based on an SMA/EMA crossover.
- SMAEMAOptimizer, WalkForwardTrainer: walk-forward optimisation of the SMA/EMA agent.
- ZScoreAgent: momentum / mean-reversion agent on a rolling z-score.
- ZScoreOptimizer, ZScoreWalkForwardTrainer: walk-forward optimisation of the Z-score agent.
- MacroYieldCurveAgent: rule-based agent on the 10Y-2Y curve slope and inflation.
- SVMAgent: support-vector classifier (yearly walk-forward with purging).
- SVMOptimizer, SVMWalkForwardTrainer: temporal cross-validation and out-of-fold signals.
- OrchestratorAgent: ensemble that combines the agent signals.
- OrchestratorOptimizer: genetic-algorithm fit of the Orchestrator weights and thresholds.
- walk_forward_orchestrator: yearly expanding walk-forward of the Orchestrator.

Data, features and metrics are defined in ``src/data_loader.py``,
``src/features.py`` and ``src/evaluator.py``.
"""

from .base_agent import BaseAgent
from .EMA_EMW_agent import SMAEMAAgent, SMAEMAOptimizer, WalkForwardTrainer
from .ZScore_agent import ZScoreAgent, ZScoreOptimizer, ZScoreWalkForwardTrainer
from .macro_yield_curve_agent import MacroYieldCurveAgent
from .SVM_agent import SVMAgent, SVMOptimizer, SVMWalkForwardTrainer
from .orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorOptimizer,
    walk_forward_orchestrator,
)

__all__ = [
    "BaseAgent",
    "SMAEMAAgent",
    "SMAEMAOptimizer",
    "WalkForwardTrainer",
    "ZScoreAgent",
    "ZScoreOptimizer",
    "ZScoreWalkForwardTrainer",
    "MacroYieldCurveAgent",
    "SVMAgent",
    "SVMOptimizer",
    "SVMWalkForwardTrainer",
    "OrchestratorAgent",
    "OrchestratorOptimizer",
    "walk_forward_orchestrator",
]

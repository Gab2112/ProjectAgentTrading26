"""
Modulo agenti per ProjectAgentTrading26.

Esporta:
- BaseAgent: classe astratta comune con interfaccia standard predict() e convenzione segnali {-1, 0, 1}
- SMAEMAAgent (e alias EMA_EMW_Agent): Agente Trend-Following basato su crossover SMA / EMA
- SMAEMAOptimizer, WalkForwardTrainer: Moduli di ottimizzazione per SMA/EMA
- ZScoreAgent: Agente Mean-Reversion con logica di uscita a isteresi
- ZScoreOptimizer, ZScoreWalkForwardTrainer: Moduli di ottimizzazione per Z-Score
- MacroYieldCurveAgent: Agente macroeconomico basato su pendenza di curva (10Y-2Y), regimi e inflazione
- FinBERTAgent: Agente sentiment basato su transformer FinBERT
- KalmanOrchestratorAgent (e alias OrchestratorAgent): Agente Orchestratore basato su Filtro di Kalman adattivo continuo (senza training esplicito)
- StaticOrchestratorAgent: Agente Orchestratore statico a griglia (benchmark comparativo)
- OrchestratorOptimizer: Modulo di ottimizzazione dei pesi e delle soglie decisionali per il modello statico
"""

from .base_agent import BaseAgent
from .EMA_EMW_agent import EMA_EMW_Agent, SMAEMAAgent, SMAEMAOptimizer, WalkForwardTrainer
from .ZScore_agent import ZScoreAgent, ZScoreOptimizer, ZScoreWalkForwardTrainer
from .macro_yield_curve_agent import MacroYieldCurveAgent
from .orchestrator_agent import (
    KalmanOrchestratorAgent,
    OrchestratorAgent,
    StaticOrchestratorAgent,
    OrchestratorOptimizer,
)

try:
    from .FinBERT_agent import FinBERTAgent
except ImportError:
    FinBERTAgent = None  # torch o transformers non installati

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
    "FinBERTAgent",
    "KalmanOrchestratorAgent",
    "OrchestratorAgent",
    "StaticOrchestratorAgent",
    "OrchestratorOptimizer",
]

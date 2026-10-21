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
- download_news_dataset: Funzione per acquisire e strutturare il database delle notizie finanziarie
- OrchestratorAgent: Agente Orchestratore quantitativo per la sintesi e combinazione ottima di segnali
- OrchestratorOptimizer: Modulo di ottimizzazione dei pesi e delle soglie decisionali per l'Orchestratore
- WalkForwardOrchestrator: Modulo per la validazione walk-forward rolling dell'Orchestratore
"""

from .base_agent import BaseAgent
from .EMA_EMW_agent import EMA_EMW_Agent, SMAEMAAgent, SMAEMAOptimizer, WalkForwardTrainer
from .ZScore_agent import ZScoreAgent, ZScoreOptimizer, ZScoreWalkForwardTrainer
from .macro_yield_curve_agent import MacroYieldCurveAgent
from .orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorOptimizer,
    WalkForwardOrchestrator,
)

try:
    from .FinBERT_agent import FinBERTAgent, download_news_dataset
except ImportError:
    FinBERTAgent = None  # torch o transformers non installati
    download_news_dataset = None

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
    "download_news_dataset",
    "OrchestratorAgent",
    "OrchestratorOptimizer",
    "WalkForwardOrchestrator",
]

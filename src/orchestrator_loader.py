"""Single canonical release loader for frozen GA Orchestrator JSON snapshots.

Do not call the legacy OrchestratorAgent.load_params: it does not handle the
nested best_params schema and would silently replace asymmetric thresholds.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.agents.orchestrator_agent import OrchestratorAgent
from src.utils import ASSETS, ROOT_DIR

AGENTS = ('SMA_EMA_Agent', 'ZScore_Agent', 'Macro_YieldCurve_Agent',
          'RandomForest_Agent', 'SVM_Agent')


def load_frozen_orchestrator(asset: str, root: Path = ROOT_DIR) -> OrchestratorAgent:
    asset = str(asset).upper()
    if asset not in ASSETS:
        raise ValueError('Invalid frozen ETF asset')
    path = Path(root)/'src/agents/config'/f'orchestrator_params_{asset.lower()}.json'
    snapshot = json.loads(path.read_text(encoding='utf-8'))
    if int(snapshot['max_train_year']) != 2021:
        raise ValueError(f'{asset}: GA was not frozen by 2021')
    p = snapshot['best_params']
    if set(p['weights']) != set(AGENTS) or set(p['alphas']) != set(AGENTS):
        raise ValueError(f'{asset}: incorrect model ensemble')
    if not 0 <= p['threshold_long'] <= 1 or not 0 <= p['threshold_short'] <= 1:
        raise ValueError(f'{asset}: invalid directional thresholds')
    model = OrchestratorAgent(
        weights=p['weights'], alphas=p['alphas'],
        theta_long=p['threshold_long'], theta_short=p['threshold_short'],
        target_asset=asset,
    )
    assert model.weights == p['weights']
    assert model.alphas == p['alphas']
    assert model.theta_long == p['threshold_long']
    assert model.theta_short == p['threshold_short']
    return model


def verify_all_orchestrators(root: Path = ROOT_DIR) -> int:
    for asset in ASSETS:
        load_frozen_orchestrator(asset, root)
    return len(ASSETS)

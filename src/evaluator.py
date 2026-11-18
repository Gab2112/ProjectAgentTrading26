"""Performance metrics.

Two families of functions are provided.

1. Reporting metrics (``calculate_metrics`` and helpers) evaluate the DAILY
   net-of-cost returns produced by the backtest engine:
   - cumulative return and CAGR = (final wealth) ** (252 / n) - 1;
   - annualised volatility = sample standard deviation * sqrt(252);
   - Sharpe ratio = mean daily excess return / std of daily excess return
     * sqrt(252), with the annual risk-free rate set in ``config.yaml``
     (0% by assumption: cash is unremunerated in the backtest as well);
   - Sortino ratio, maximum drawdown (including the initial wealth), Calmar
     ratio and profit factor (sum of positive daily returns divided by the
     absolute sum of negative daily returns);
   - market exposure: beta and correlation with a reference index, and the
     mean strategy return on days when the index falls.

2. Training metrics (``calculate_performance_metrics`` and
   ``compute_objective_score``) are the fitness functions used by the agent
   optimisers on WEEKLY signal returns. They are used for model selection only
   and never for reporting results.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# =============================================================================
# Reporting metrics on daily returns
# =============================================================================


def _validated_returns(returns: pd.Series) -> np.ndarray:
    values = pd.to_numeric(returns, errors="raise").dropna().to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite returns are required.")
    if np.any(values <= -1.0):
        raise ValueError("Returns of -100% or lower are not admissible.")
    return values


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown of the wealth path, including the initial wealth of 1."""
    r = _validated_returns(returns)
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    return float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))


def calculate_profit_factor(returns: pd.Series) -> float:
    """Sum of positive periodic returns divided by the absolute sum of negative ones.

    The measure is computed on periodic (daily) returns rather than on
    individual round-trip trades.
    """
    r = _validated_returns(returns)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else float("nan")


def sharpe_ratio(returns: np.ndarray, risk_free_daily: float, annualization_factor: int) -> float:
    """Annualised Sharpe ratio of a vector of periodic returns."""
    excess = np.asarray(returns, dtype=float) - risk_free_daily
    std = float(np.std(excess, ddof=1))
    return float(np.mean(excess) / std * np.sqrt(annualization_factor)) if std > 0 else float("nan")


def calculate_metrics(
    returns: pd.Series,
    risk_free_rate: float,
    annualization_factor: int,
    market_returns: pd.Series | None = None,
) -> dict:
    """Compute the reporting metrics of a series of daily net returns.

    ``risk_free_rate`` is annual and converted to a daily rate by
    compounding. When ``market_returns`` is supplied, beta, correlation and
    the mean return on down-market days are added.
    """
    if annualization_factor <= 1:
        raise ValueError("Invalid annualization factor.")
    r = _validated_returns(returns)
    rf_daily = (1.0 + float(risk_free_rate)) ** (1.0 / annualization_factor) - 1.0

    n = len(r)
    final_wealth = float(np.prod(1.0 + r))
    cagr = final_wealth ** (annualization_factor / n) - 1.0
    volatility = float(np.std(r, ddof=1) * np.sqrt(annualization_factor))

    excess = r - rf_daily
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    sortino = (
        float(np.mean(excess) / downside * np.sqrt(annualization_factor))
        if downside > 0 else float("nan")
    )
    max_dd = calculate_max_drawdown(returns)

    metrics = {
        "observations": int(n),
        "cumulative_return": final_wealth - 1.0,
        "cagr": float(cagr),
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe_ratio(r, rf_daily, annualization_factor),
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": float(cagr / abs(max_dd)) if max_dd < 0 else float("nan"),
        "profit_factor": calculate_profit_factor(returns),
        "hit_rate": float(np.mean(r[r != 0] > 0)) if np.any(r != 0) else float("nan"),
    }

    if market_returns is not None:
        metrics.update(market_exposure(returns, market_returns))
    return metrics


def market_exposure(returns: pd.Series, market_returns: pd.Series) -> dict:
    """Beta, correlation and down-market behaviour relative to a reference index."""
    joined = pd.concat([returns, market_returns], axis=1, join="inner").dropna()
    if len(joined) < 3:
        raise ValueError("Insufficient overlap with the market series.")
    strategy, market = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    variance = float(np.var(market, ddof=1))
    beta = float(np.cov(strategy, market, ddof=1)[0, 1] / variance) if variance > 0 else float("nan")
    down = market < 0
    return {
        "beta_to_market": beta,
        "correlation_to_market": float(np.corrcoef(strategy, market)[0, 1]),
        "mean_return_market_down_days": float(strategy[down].mean()) if down.any() else float("nan"),
    }


# =============================================================================
# Training (fitness) metrics on weekly signal returns
# =============================================================================


def calculate_performance_metrics(
    returns: pd.Series,
    annualization_factor: int = 52,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Fitness metrics of a series of periodic (weekly) signal returns.

    Used by the agent optimisers for model selection. The ratio
    ``sharpe_ratio`` here is the geometric return-to-volatility ratio
    (annualised return / annualised volatility), which penalises paths with
    large losses more strongly than the arithmetic Sharpe ratio used for
    reporting.
    """
    clean_ret = returns.fillna(0.0)
    n = len(clean_ret)
    if n == 0:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "calmar_ratio": 0.0,
        }

    cum_ret = float((1.0 + clean_ret).prod() - 1.0)
    ann_ret = -1.0 if cum_ret <= -1.0 else float((1.0 + cum_ret) ** (annualization_factor / n) - 1.0)
    vol = float(clean_ret.std(ddof=1) * np.sqrt(annualization_factor)) if n > 1 else 0.0

    excess_ret = ann_ret - risk_free_rate
    if vol > 1e-6:
        sharpe = float(excess_ret / vol)
    else:
        sharpe = 0.0 if excess_ret >= 0 else -99.0

    wealth = (1.0 + clean_ret).cumprod()
    mdd = abs(float((wealth / wealth.cummax() - 1.0).min()))

    gains = float(clean_ret[clean_ret > 0].sum())
    losses = float(abs(clean_ret[clean_ret < 0].sum()))
    profit_factor = float(gains / losses) if losses > 1e-6 else (10.0 if gains > 0 else 1.0)

    return {
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "profit_factor": profit_factor,
        "calmar_ratio": float(ann_ret / max(mdd, 0.01)),
    }


def compute_objective_score(metrics: Dict[str, float], metric_type: str = "sharpe_mdd") -> float:
    """Optimisation score balancing return against risk.

    - ``'sharpe_mdd'`` (default): Sharpe * (1 - MDD) if Sharpe >= 0,
      otherwise Sharpe * (1 + MDD); rewards smooth equity curves;
    - ``'sharpe'``: annualised return-to-volatility ratio;
    - ``'calmar'``: annualised return / maximum drawdown;
    - ``'return'``: cumulative return, without risk adjustment;
    - ``'profit'``: profit factor * (1 - MDD).
    """
    sharpe = metrics["sharpe_ratio"]
    mdd = metrics["max_drawdown"]
    ann_ret = metrics["annualized_return"]

    if metric_type == "sharpe_mdd":
        if sharpe >= 0:
            return float(sharpe * (1.0 - min(mdd, 0.99)))
        return float(sharpe * (1.0 + min(mdd, 1.0)))
    if metric_type == "sharpe":
        return float(sharpe)
    if metric_type == "calmar":
        return float(ann_ret / max(mdd, 0.02))
    if metric_type == "return":
        return float(metrics["cumulative_return"])
    if metric_type in ("profit_oriented", "profit"):
        return float(metrics.get("profit_factor", 1.0) * (1.0 - min(mdd, 0.99)))
    raise ValueError(f"Unknown objective metric '{metric_type}'.")

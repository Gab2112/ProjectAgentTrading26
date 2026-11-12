"""Abstract base class for all trading agents of ProjectAgentTrading26.

It defines the shared standard interface for:
- inference and signal generation: ``predict()`` and ``generate_signals()``;
- the signal convention: 1 (BUY/LONG), -1 (SELL/SHORT), 0 (HOLD/FLAT);
- data access through ``src/data_loader.py`` (``load_data``);
- automatic resolution of asset-specific price columns;
- parameter serialisation: ``load_params()``, ``save_params()``;
- uniform output formatting (numeric [-1, 0, 1] or string ['SELL', 'HOLD', 'BUY']).
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data_loader import load_agent_dataset, prepare_asset_data
from src.utils import ASSETS


class BaseAgent(ABC):
    """Abstract base class for all trading agents.

    Every derived agent must implement the abstract method:
        predict(df: pd.DataFrame) -> pd.Series
    """

    DEFAULT_ASSETS = ASSETS

    def __init__(
        self,
        name: str = "BaseAgent",
        target_asset: Optional[str] = None,
        price_col: Optional[str] = None,
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) or 'string' ('SELL', 'HOLD', 'BUY')
    ):
        self.name = name
        self.target_asset = target_asset.upper() if target_asset else None
        self.price_col = price_col or (f"{self.target_asset}_close" if self.target_asset else "close")
        self.output_format = output_format
        self.params: Dict[str, Any] = {}

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Compute the trading signals aligned with the DataFrame index.

        Args:
            df (pd.DataFrame): Historical data (prices, features or text).

        Returns:
            pd.Series: Time series of signals: -1 (SELL/SHORT), 0 (HOLD/FLAT),
            1 (BUY/LONG), or the corresponding strings if ``output_format='string'``.
        """
        pass

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Alias of ``predict()``, consistent with the pipeline signal-generation conventions.
        """
        return self.predict(df)

    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Compute both the discrete signals {-1, 0, 1} and a confidence estimate in [0.0, 1.0].

        By default, active signals (-1, 1) receive a confidence of 1.0 and HOLD (0)
        receives 0.0. Derived classes can refine the computation with probabilistic
        models or continuous metrics.
        """
        signals = self.predict(df)
        confidence = pd.Series(np.where(signals != 0, 1.0, 0.0), index=df.index, dtype=float)
        return signals, confidence

    def resolve_price_col(
        self,
        df: pd.DataFrame,
        price_col: Optional[str] = None,
        target_asset: Optional[str] = None,
    ) -> str:
        """Resolve the price column of the DataFrame in a tolerant way.

        Supports both DataFrames with generic columns (``'close'``, ``'Close'``)
        and multi-asset DataFrames with prefixed columns (``'SPY_close'``,
        ``'IEF_close'``, etc.).
        """
        candidate_col = price_col or self.price_col
        asset = (target_asset or self.target_asset or "").upper()

        # 1. The candidate is already present as is.
        if candidate_col and candidate_col in df.columns:
            return candidate_col

        # 2. If a target asset is specified, look for f"{asset}_close".
        if asset:
            for suffix in ["_close", "_Close", "_CLOSE"]:
                prefixed = f"{asset}{suffix}"
                if prefixed in df.columns:
                    return prefixed
            for suffix in ["_close", "_Close"]:
                prefixed = f"{asset.lower()}{suffix}"
                if prefixed in df.columns:
                    return prefixed

        # 3. Check the generic 'close' or 'Close' columns.
        for col_name in ["close", "Close", "Adj Close", "adj_close"]:
            if col_name in df.columns:
                return col_name

        # 4. The DataFrame has a single column ending in '_close'.
        close_like = [c for c in df.columns if str(c).lower().endswith("_close")]
        if len(close_like) == 1:
            return close_like[0]

        raise KeyError(
            f"[{self.name}] Impossibile determinare la colonna del prezzo nel DataFrame. "
            f"Candidato cercato: '{candidate_col}', target_asset: '{asset}'. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    # =====================================================================
    # DATA LOADING (delegated to src/data_loader.py)
    # =====================================================================


    def load_data(
        self,
        asset: Optional[str] = None,
        frequency: str = "weekly",
        include_macro: bool = True,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        root_dir: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Load the data prepared for the agent.

        If ``target_asset`` is set on the agent, the columns are adapted automatically.
        """
        target = (asset or self.target_asset or "").upper()
        df = load_agent_dataset(
            frequency=frequency,
            include_macro=include_macro,
            start_date=start_date,
            end_date=end_date,
            root_dir=root_dir,
        )

        if target:
            return prepare_asset_data(df, asset=target, price_col=self.price_col)
        return df

    # =====================================================================
    # PARAMETER MANAGEMENT AND SERIALISATION
    # =====================================================================
    def load_params(self, filepath: Union[str, Path]) -> "BaseAgent":
        """Load optimal parameters from a JSON file and update the agent state.

        Args:
            filepath: Path of the JSON file containing the parameters.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File parametri non trovato: {path}")

        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)

        self.params = params
        self._apply_params(params)
        return self

    def save_params(self, filepath: Union[str, Path]) -> None:
        """Save the current agent parameters to a JSON file.

        Args:
            filepath: Destination path of the JSON file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=4)

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Optional hook: update the derived agent attributes from the loaded parameter dictionary.
        """
        if "target_asset" in params:
            self.target_asset = str(params["target_asset"]).upper()
        if "price_col" in params:
            self.price_col = str(params["price_col"])

    def _format_signals(self, signals: pd.Series) -> pd.Series:
        """Normalise and validate the signal series.

        - Replaces NaN values with 0 (HOLD/FLAT).
        - Converts the values to integers {-1, 0, 1}.
        - Maps them to strings if ``output_format == 'string'``.
        """
        formatted = signals.fillna(0).astype(int)

        # Check that only the values allowed by the project (-1, 0, 1) are present.
        valid_values = {-1, 0, 1}
        observed = set(formatted.unique())
        if not observed.issubset(valid_values):
            raise ValueError(
                f"[{self.name}] Valori non ammessi rilevati nei segnali: {observed}. "
                f"I valori devono appartenere a {-1, 0, 1}."
            )

        if self.output_format == "string":
            mapping = {1: "BUY", -1: "SELL", 0: "HOLD"}
            return formatted.map(mapping)

        return formatted

    def __repr__(self) -> str:
        asset_str = f", target_asset='{self.target_asset}'" if self.target_asset else ""
        return f"{self.__class__.__name__}(name='{self.name}'{asset_str}, output_format='{self.output_format}', params={self.params})"


# =====================================================================
# PERFORMANCE AND STABILITY METRIC UTILITIES
# =====================================================================
def calculate_performance_metrics(
    returns: pd.Series,
    annualization_factor: int = 52,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Compute performance and risk/stability metrics for a series of periodic returns.

    Args:
        returns: Periodic (weekly) returns.
        annualization_factor: Annualisation factor (default 52 for weekly data).
        risk_free_rate: Annualised risk-free rate (default 0.0).

    Returns:
        Dictionary with cumulative_return, annualized_return, annualized_volatility,
        sharpe_ratio, max_drawdown, profit_factor and calmar_ratio.
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
    if cum_ret <= -1.0:
        ann_ret = -1.0
    else:
        ann_ret = float((1.0 + cum_ret) ** (annualization_factor / n) - 1.0)

    vol = float(clean_ret.std(ddof=1) * np.sqrt(annualization_factor)) if n > 1 else 0.0

    excess_ret = ann_ret - risk_free_rate
    if vol > 1e-6:
        sharpe = float(excess_ret / vol)
    else:
        sharpe = 0.0 if excess_ret >= 0 else -99.0

    # Maximum Drawdown
    wealth = (1.0 + clean_ret).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    mdd = abs(float(drawdown.min()))

    # Profit Factor
    gains = float(clean_ret[clean_ret > 0].sum())
    losses = float(abs(clean_ret[clean_ret < 0].sum()))
    profit_factor = float(gains / losses) if losses > 1e-6 else (10.0 if gains > 0 else 1.0)

    # Calmar Ratio
    calmar = float(ann_ret / max(mdd, 0.01))

    return {
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "profit_factor": profit_factor,
        "calmar_ratio": calmar,
    }


def compute_objective_score(
    metrics: Dict[str, float],
    metric_type: str = "sharpe_mdd",
) -> float:
    """Compute the optimisation score, balancing return against stability and risk.

    Supported options:
    - ``'sharpe_mdd'`` (default): Sharpe ratio penalised by the maximum drawdown,
      rewarding smooth equity curves with limited maximum losses.
      Score = Sharpe * (1 - MDD) if Sharpe >= 0, otherwise Sharpe * (1 + MDD).
    - ``'sharpe'``: plain annualised Sharpe ratio.
    - ``'calmar'``: Calmar ratio (annualised return / maximum drawdown).
    - ``'return'``: plain cumulative return, with no risk penalty.
    """
    sharpe = metrics["sharpe_ratio"]
    mdd = metrics["max_drawdown"]
    ann_ret = metrics["annualized_return"]

    if metric_type == "sharpe_mdd":
        if sharpe >= 0:
            return float(sharpe * (1.0 - min(mdd, 0.99)))
        else:
            return float(sharpe * (1.0 + min(mdd, 1.0)))
    elif metric_type == "sharpe":
        return float(sharpe)
    elif metric_type == "calmar":
        return float(ann_ret / max(mdd, 0.02))
    elif metric_type == "return":
        return float(metrics["cumulative_return"])
    elif metric_type in ("profit_oriented", "profit"):
        pf = metrics.get("profit_factor", 1.0)
        return float(pf * (1.0 - min(mdd, 0.99)))
    else:
        raise ValueError(f"Metrica obiettivo '{metric_type}' non riconosciuta.")

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

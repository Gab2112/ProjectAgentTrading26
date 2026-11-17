"""Macroeconomic agent based on the term structure of interest rates (yield curve) and on inflation.

The agent inherits from BaseAgent to comply with the standard project architecture.

Economic rationale:
Treasury prices and yields do not depend on corporate fundamentals but on
expectations about monetary policy, the business cycle and inflation.
The agent analyses:
1. Curve slope (10Y-2Y and 10Y-3M spreads): inverted versus steep curve.
2. The four curve-change regimes:
   - Bull flattening: falling rates led by the 10-year yield -> long TLT (+1)
   - Bull steepening: rapidly falling short-term rates (Fed cuts) -> long duration / TLT (+1)
   - Bear flattening: rising short-term rates due to monetary tightening -> short TLT (-1), long SHY (+1)
   - Bear steepening: rising long-term rates due to inflation or term premium -> short TLT (-1)
3. Inflation dynamics (CPI YoY or breakeven): accelerating inflation penalises duration.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent
from src.data_loader import load_agent_macro_data


class MacroYieldCurveAgent(BaseAgent):
    """Quantitative macro and yield-curve agent.

    - Combines curve slope, rate changes and inflation.
    - Produces a continuous macro score and discrete directional signals {-1, 0, 1}.
    """

    def __init__(
        self,
        slope_col: str = "curve_10y_2y",
        rate_change_col: str = "treasury_10y_change_1w",
        inflation_col: str = "cpi_yoy",
        slope_window: int = 4,
        rate_window: int = 4,
        score_threshold: float = 0.5,
        target_asset: Optional[str] = "IEF",
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) or 'string' ('SELL', 'HOLD', 'BUY')
    ):
        super().__init__(
            name="MacroYieldCurveAgent",
            target_asset=target_asset,
            output_format=output_format,
        )
        self.slope_col = slope_col
        self.rate_change_col = rate_change_col
        self.inflation_col = inflation_col
        self.slope_window = slope_window
        self.rate_window = rate_window
        self.score_threshold = score_threshold
        self.target_asset = (target_asset or "IEF").upper()

        self.params = {
            "slope_col": self.slope_col,
            "rate_change_col": self.rate_change_col,
            "inflation_col": self.inflation_col,
            "slope_window": int(self.slope_window),
            "rate_window": int(self.rate_window),
            "score_threshold": float(self.score_threshold),
            "target_asset": self.target_asset,
        }

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Update the agent parameters after loading them from JSON."""
        if "slope_col" in params:
            self.slope_col = str(params["slope_col"])
        if "rate_change_col" in params:
            self.rate_change_col = str(params["rate_change_col"])
        if "inflation_col" in params:
            self.inflation_col = str(params["inflation_col"])
        if "slope_window" in params:
            self.slope_window = int(params["slope_window"])
        if "rate_window" in params:
            self.rate_window = int(params["rate_window"])
        if "score_threshold" in params:
            self.score_threshold = float(params["score_threshold"])
        if "target_asset" in params:
            self.target_asset = str(params["target_asset"])

    def calculate_macro_score(self, df: pd.DataFrame) -> pd.Series:
        """Build a standardised macro score from curve slope, rates and inflation.

        Returns:
            pd.Series with the continuous score:
            - large positive values: environment favourable to falling yields and
              long duration (TLT);
            - large negative values: environment hostile to duration, with rising
              rates or high inflation.
        """
        # Check the input and, if needed, enrich it with macro columns through BaseAgent.
        if self.slope_col not in df.columns or self.rate_change_col not in df.columns:
            try:
                macro_df = load_agent_macro_data(decision_dates=df.index)
                for c in macro_df.columns:
                    if c not in df.columns:
                        df[c] = macro_df[c]
            except Exception:
                pass

        for col in [self.slope_col, self.rate_change_col]:
            if col not in df.columns:
                raise KeyError(
                    f"Colonna '{col}' non trovata nel DataFrame. Colonne disponibili: {list(df.columns)}"
                )

        slope = df[self.slope_col].astype(float)
        rate_change = df[self.rate_change_col].astype(float)

        # 1. Rate-change component (rolling-window smoothing).
        # If 10-year rates fall (rate_change < 0), bond prices rise -> POSITIVE score.
        smoothed_rate_change = rate_change.rolling(window=self.rate_window, min_periods=1).mean()
        # Normalise by the rolling standard deviation to make the score dimensionless.
        rate_change_std = rate_change.rolling(window=26, min_periods=4).std().replace(0, np.nan)
        rate_component = -1.0 * (smoothed_rate_change / rate_change_std).fillna(0.0)

        # 2. Curve slope and slope-change component (steepening versus flattening).
        delta_slope = slope.diff(self.slope_window).fillna(0.0)
        slope_std = delta_slope.rolling(window=26, min_periods=4).std().replace(0, np.nan)
        normalized_delta_slope = (delta_slope / slope_std).fillna(0.0)

        # Curve regime logic:
        # If rate_component > 0 (falling rates):
        # - delta_slope < 0 (bull flattening): maximum reward for long duration (TLT)
        # - delta_slope >= 0 (bull steepening): positive for duration
        # If rate_component < 0 (rising rates):
        # - delta_slope < 0 (bear flattening): Fed tightening, strongly negative for duration
        # - delta_slope >= 0 (bear steepening): rising term premium, negative for duration
        curve_component = np.where(
            rate_component > 0,
            1.0 + 0.5 * np.sign(-normalized_delta_slope),  # Bull regime
            -1.0 - 0.5 * np.sign(normalized_delta_slope),  # Bear regime
        )

        # Curve inversion: an inverted slope (< 0) adds a tailwind for duration.
        inversion_bonus = np.where(slope < 0, 0.5, 0.0)

        # 3. Inflation component (if available in the DataFrame).
        inflation_component = 0.0
        if self.inflation_col in df.columns:
            inf = df[self.inflation_col].dropna().astype(float)
            if len(inf) > 0:
                # 13-week (one-quarter) change in inflation.
                inf_delta = df[self.inflation_col].diff(13).fillna(0.0)
                # Falling inflation -> positive (+0.5); rising inflation -> negative (-0.5).
                inflation_component = np.where(inf_delta < 0, 0.5, np.where(inf_delta > 0, -0.5, 0.0))

        # Weighted combination of the total score.
        total_score = (
            0.50 * rate_component
            + 0.35 * pd.Series(curve_component, index=df.index)
            + 0.15 * pd.Series(inflation_component, index=df.index)
            + pd.Series(inversion_bonus, index=df.index)
        )

        return total_score

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Compute the discrete signals {-1, 0, 1} for the target asset (IEF by default, or as configured).

        - Score >= +score_threshold -> BUY (+1)
        - Score <= -score_threshold -> SELL (-1)
        - otherwise                 -> HOLD (0)
        """
        multi = self.predict_multi_asset(df, assets=[self.target_asset])
        sig_col = f"{self.target_asset}_signal"
        if sig_col in multi.columns:
            return self._format_signals(multi[sig_col])

        score = self.calculate_macro_score(df)
        raw_signals = np.where(
            score >= self.score_threshold,
            1,
            np.where(score <= -self.score_threshold, -1, 0),
        )
        signals = pd.Series(raw_signals, index=df.index, dtype=int)
        signals[score.isna()] = 0
        return self._format_signals(signals)

    def predict_multi_asset(
        self,
        df: pd.DataFrame,
        assets: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Generate signals for every requested asset (default: SPY, IEF, GLD, SHY).

        Full compatibility with the original universe (SHY, IEF, TLT) is preserved.
        """
        score = self.calculate_macro_score(df)
        multi = pd.DataFrame(index=df.index)

        target_assets = [str(a).upper() for a in assets] if assets is not None else list(self.DEFAULT_ASSETS)

        bull_mask = score >= self.score_threshold
        bear_mask = score <= -self.score_threshold
        neutral_mask = ~bull_mask & ~bear_mask & score.notna()

        slope = df[self.slope_col].astype(float) if self.slope_col in df.columns else pd.Series(0.0, index=df.index)

        for asset in target_assets:
            sig_col = f"{asset}_signal"
            multi[sig_col] = 0

            if asset == "IEF":
                # Intermediate duration: long when rates fall (bull), short on strong tightening, otherwise flat.
                multi.loc[bull_mask, sig_col] = 1
                multi.loc[bear_mask, sig_col] = -1
                multi.loc[neutral_mask, sig_col] = 1
            elif asset == "SHY":
                # Short duration / cash proxy: long when rates rise (safe haven), flat in bull regimes.
                multi.loc[bear_mask, sig_col] = 1
                multi.loc[bull_mask, sig_col] = 0
            elif asset == "TLT":
                # Long duration (historical universe): maximum sensitivity to falling rates.
                multi.loc[bull_mask, sig_col] = 1
                multi.loc[bear_mask, sig_col] = -1
            elif asset == "SPY":
                # US equities: favoured by accommodative conditions with a positive curve, cautious on curve inversion.
                multi.loc[bull_mask & (slope >= 0), sig_col] = 1
                multi.loc[bear_mask | (slope < 0), sig_col] = 0
            elif asset == "GLD":
                # Gold: inflation hedge and protection against negative real rates.
                multi.loc[bull_mask | (score > 0), sig_col] = 1
                multi.loc[bear_mask, sig_col] = 0

        return multi

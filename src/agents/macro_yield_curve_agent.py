"""
macro_yield_curve_agent.py

Agente Macroeconomico basato sulla Struttura a Termine dei Tassi (Yield Curve) e sull'Inflazione.
Eredita da BaseAgent per garantire piena conformità all'architettura standard del progetto.

Razionale Economico:
Nel mercato obbligazionario (Treasuries), i prezzi e i rendimenti non dipendono da dinamiche
aziendali, ma dalle aspettative su politica monetaria, ciclo economico e inflazione.
Questo agente analizza:
1. Pendenza della Curva (Spread 10Y - 2Y e 10Y - 3M): Inversione vs Curva Ripida.
2. I 4 Regimi di Variazione di Curva:
   - Bull Flattening: tassi in discesa guidati dal decennale -> Long TLT (+1)
   - Bull Steepening: tassi a breve in rapida discesa (tagli Fed) -> Long Duration / TLT (+1)
   - Bear Flattening: tassi a breve in salita per stretta monetaria -> Short TLT (-1), Long SHY (+1)
   - Bear Steepening: tassi a lungo in salita per inflazione/premio al rischio -> Short TLT (-1)
3. Dinamica dell'Inflazione (CPI YoY o Breakeven): accelerazione inflazionistica penalizza la duration.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .base_agent import BaseAgent


class MacroYieldCurveAgent(BaseAgent):
    """
    Agente Quantitativo Macro & Yield Curve:
    - Analizza la combinazione di slope di curva, variazione dei tassi e inflazione.
    - Genera uno score macro continuo e produce segnali direzionali discreti {-1, 0, 1}.
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
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) o 'string' ('SELL', 'HOLD', 'BUY')
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
        """Aggiorna i parametri dell'agente dopo il caricamento da JSON."""
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
        """
        Costruisce uno score macro standardizzato a partire da pendenza, tassi e inflazione.

        Ritorna:
            pd.Series con lo score continuo:
            - Valori positivi elevati: quadro favorevole a discesa dei rendimenti e TLT (Long Duration).
            - Valori negativi elevati: quadro ostile alla duration, tassi in salita o inflazione alta.
        """
        # Verifica ed eventuale arricchimento con colonne macro tramite BaseAgent
        if self.slope_col not in df.columns or self.rate_change_col not in df.columns:
            try:
                macro_df = self.load_macro_data(decision_dates=df.index)
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

        # 1. Componente Variazione Tassi (Smoothing a finestra mobile)
        # Se i tassi a 10 anni scendono (rate_change < 0), i prezzi dei bond salgono -> Score POSITIVO
        smoothed_rate_change = rate_change.rolling(window=self.rate_window, min_periods=1).mean()
        # Normalizzazione con deviazione standard rolling per rendere lo score adimensionale
        rate_change_std = rate_change.rolling(window=26, min_periods=4).std().replace(0, np.nan)
        rate_component = -1.0 * (smoothed_rate_change / rate_change_std).fillna(0.0)

        # 2. Componente Pendenza di Curva e Variazione dello Slope (Steepening vs Flattening)
        delta_slope = slope.diff(self.slope_window).fillna(0.0)
        slope_std = delta_slope.rolling(window=26, min_periods=4).std().replace(0, np.nan)
        normalized_delta_slope = (delta_slope / slope_std).fillna(0.0)

        # Logica dei Regimi di Curva:
        # Se rate_component > 0 (tassi in calo):
        #   - delta_slope < 0 (Bull Flattening): premio massimo alla duration lunga TLT
        #   - delta_slope >= 0 (Bull Steepening): positivo per la duration
        # Se rate_component < 0 (tassi in salita):
        #   - delta_slope < 0 (Bear Flattening): stretta Fed, molto negativo per duration
        #   - delta_slope >= 0 (Bear Steepening): premio a lungo termine sale, penalizzante
        curve_component = np.where(
            rate_component > 0,
            1.0 + 0.5 * np.sign(-normalized_delta_slope),  # Bull regime
            -1.0 - 0.5 * np.sign(normalized_delta_slope),  # Bear regime
        )

        # Inversione di curva: se lo slope assoluto è invertito (< 0), c'è una spinta favorevole alla duration
        inversion_bonus = np.where(slope < 0, 0.5, 0.0)

        # 3. Componente Inflazione (se disponibile nel DataFrame)
        inflation_component = 0.0
        if self.inflation_col in df.columns:
            inf = df[self.inflation_col].dropna().astype(float)
            if len(inf) > 0:
                # Variazione a 13 settimane (1 trimestre) dell'inflazione
                inf_delta = df[self.inflation_col].diff(13).fillna(0.0)
                # Inflazione in calo -> positivo (+0.5); inflazione in salita -> negativo (-0.5)
                inflation_component = np.where(inf_delta < 0, 0.5, np.where(inf_delta > 0, -0.5, 0.0))

        # Combinazione ponderata dello score totale
        total_score = (
            0.50 * rate_component
            + 0.35 * pd.Series(curve_component, index=df.index)
            + 0.15 * pd.Series(inflation_component, index=df.index)
            + pd.Series(inversion_bonus, index=df.index)
        )

        return total_score

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcola i segnali operativi discreti {-1, 0, 1} per l'asset target (default IEF o configurato).

        - Score >= +score_threshold -> BUY (+1)
        - Score <= -score_threshold -> SELL (-1)
        - Altrimenti                -> HOLD (0)
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
        """
        Genera i segnali per tutti gli asset specificati (default: SPY, IEF, GLD, SHY).
        Supporta e preserva piena compatibilità con l'universo originario (SHY, IEF, TLT).
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
                # Intermediate duration: Long su discesa tassi/bull, Short su stretta forte, Flat/Long neutrale
                multi.loc[bull_mask, sig_col] = 1
                multi.loc[bear_mask, sig_col] = -1
                multi.loc[neutral_mask, sig_col] = 1
            elif asset == "SHY":
                # Short duration / cash proxy: Long su salita tassi (rifugio), 0 su bull
                multi.loc[bear_mask, sig_col] = 1
                multi.loc[bull_mask, sig_col] = 0
            elif asset == "TLT":
                # Long duration (legacy): massima sensibilità a tassi in calo
                multi.loc[bull_mask, sig_col] = 1
                multi.loc[bear_mask, sig_col] = -1
            elif asset == "SPY":
                # Azionario USA: favorito da condizioni accomodanti con curva positiva, cauto/flat su inversione di curva
                multi.loc[bull_mask & (slope >= 0), sig_col] = 1
                multi.loc[bear_mask | (slope < 0), sig_col] = 0
            elif asset == "GLD":
                # Oro: bene rifugio anti-inflazione e protezione da tassi reali negativi
                multi.loc[bull_mask | (score > 0), sig_col] = 1
                multi.loc[bear_mask, sig_col] = 0

        return multi

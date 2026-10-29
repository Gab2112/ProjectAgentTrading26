"""
base_agent.py

Classe base astratta per tutti gli agenti di trading del framework ProjectAgentTrading26.
Definisce l'interfaccia standard e condivisa per:
- Inferenza e generazione segnali: predict() e generate_signals()
- Convenzione dei segnali: 1 (BUY/LONG), -1 (SELL/SHORT), 0 (HOLD/FLAT)
- Caricamento, validazione e allineamento dati multi-asset (SPY, IEF, GLD, SHY) e macro
- Risoluzione automatica delle colonne di prezzo per asset specifici
- Serializzazione e caricamento parametri: load_params(), save_params()
- Formattazione output uniforme (numeric [-1, 0, 1] o string ['SELL', 'HOLD', 'BUY'])
"""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


DEFAULT_ASSETS = ("SPY", "IEF", "GLD", "SHY")
DEFAULT_TRAIN_START = "2005-05-20"  # Inizio comune multi-asset (avvio quotazione GLD)
DEFAULT_TRAIN_END = "2016-12-31"


def get_project_root(start_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Risolve in modo robusto la directory radice del repository ProjectAgentTrading.
    """
    if start_path is not None:
        current = Path(start_path).resolve()
    else:
        current = Path(__file__).resolve().parent

    for candidate in [current] + list(current.parents):
        if (candidate / "config.yaml").is_file() and (candidate / "src").is_dir():
            return candidate

    # Fallback standard: due livelli sopra src/agents
    return Path(__file__).resolve().parents[2]


class BaseAgent(ABC):
    """
    Classe base astratta per tutti gli agenti di trading.

    Tutti gli agenti derivati devono implementare il metodo astratto:
        predict(df: pd.DataFrame) -> pd.Series
    """

    DEFAULT_ASSETS = DEFAULT_ASSETS

    def __init__(
        self,
        name: str = "BaseAgent",
        target_asset: Optional[str] = None,
        price_col: Optional[str] = None,
        output_format: str = "numeric",  # 'numeric' (-1, 0, 1) o 'string' ('SELL', 'HOLD', 'BUY')
    ):
        self.name = name
        self.target_asset = target_asset.upper() if target_asset else None
        self.price_col = price_col or (f"{self.target_asset}_close" if self.target_asset else "close")
        self.output_format = output_format
        self.params: Dict[str, Any] = {}

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcola e restituisce la serie dei segnali operativi allineata all'indice del DataFrame.

        Parametri:
            df (pd.DataFrame): DataFrame contenente i dati storici (prezzi, feature o testi).

        Ritorna:
            pd.Series: Serie temporale con i segnali generati:
                       -1 (SELL/SHORT), 0 (HOLD/FLAT), 1 (BUY/LONG)
                       oppure le corrispondenti stringhe se output_format='string'.
        """
        pass

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Alias intuitivo per il metodo predict().
        Garantisce coerenza con le convenzioni di signal_generation della pipeline.
        """
        return self.predict(df)

    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calcola sia i segnali operativi discreti {-1, 0, 1} sia la stima di confidenza / probabilità [0.0, 1.0].
        Di default assegna confidenza 1.0 ai segnali attivi (-1, 1) e 0.0 a HOLD (0).
        Le classi derivate possono specializzare il calcolo con modelli probabilistici o metriche continue.
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
        """
        Risolve in modo intelligente e tollerante la colonna dei prezzi nel DataFrame.
        Supporta sia DataFrame con colonne generiche ('close', 'Close') sia
        DataFrame multi-asset con prefisso ('SPY_close', 'IEF_close', ecc.).
        """
        candidate_col = price_col or self.price_col
        asset = (target_asset or self.target_asset or "").upper()

        # 1. Se il candidato è già esattamente presente
        if candidate_col and candidate_col in df.columns:
            return candidate_col

        # 2. Se è specificato un asset target, cerca la combinazione f"{asset}_close"
        if asset:
            for suffix in ["_close", "_Close", "_CLOSE"]:
                prefixed = f"{asset}{suffix}"
                if prefixed in df.columns:
                    return prefixed
            for suffix in ["_close", "_Close"]:
                prefixed = f"{asset.lower()}{suffix}"
                if prefixed in df.columns:
                    return prefixed

        # 3. Controllo colonne generiche 'close' o 'Close'
        for col_name in ["close", "Close", "Adj Close", "adj_close"]:
            if col_name in df.columns:
                return col_name

        # 4. Se df ha una sola colonna che finisce per '_close'
        close_like = [c for c in df.columns if str(c).lower().endswith("_close")]
        if len(close_like) == 1:
            return close_like[0]

        raise KeyError(
            f"[{self.name}] Impossibile determinare la colonna del prezzo nel DataFrame. "
            f"Candidato cercato: '{candidate_col}', target_asset: '{asset}'. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    # =====================================================================
    # METODI DI CARICAMENTO E PREPARAZIONE DATI (MULTI-ASSET READY)
    # =====================================================================
    @classmethod
    def load_market_data(
        cls,
        frequency: str = "weekly",
        source: Optional[Union[str, Path]] = None,
        assets: Optional[Sequence[str]] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        root_dir: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Carica i dati di mercato storici per gli asset del paniere (default: SPY, IEF, GLD, SHY).

        Parametri:
            frequency: 'weekly' (default) o 'daily'.
            source: Percorso opzionale a un file CSV specifico o alias ('master_weekly', 'market_daily').
            assets: Lista di ticker da selezionare (es. ['SPY', 'GLD']). Se None, carica tutti gli asset disponibili.
            start_date: Data minima (inclusa), es. '2005-05-20'.
            end_date: Data massima (inclusa).
            root_dir: Directory radice del progetto (risolta automaticamente se non fornita).

        Ritorna:
            pd.DataFrame: DataFrame indicizzato per Data con i prezzi storici.
        """
        root = get_project_root(root_dir)
        processed_dir = root / "data" / "processed"

        # Risoluzione del percorso sorgente
        if source is not None and Path(source).is_file():
            csv_path = Path(source)
        elif frequency == "daily" or str(source).lower() == "market_daily":
            csv_path = processed_dir / "market_daily.csv"
        else:
            csv_path = processed_dir / "master_weekly.csv"
            if not csv_path.exists():
                fallback = processed_dir / "decision_features_training.csv"
                if fallback.exists():
                    csv_path = fallback

        if not csv_path.exists():
            raise FileNotFoundError(
                f"File di dati di mercato non trovato in: {csv_path}. "
                f"Esegui la data transformation o verifica la directory data/processed."
            )

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Filtraggio temporale
        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.Timestamp(end_date)]

        # Filtraggio per asset specifici
        if assets is not None:
            req_assets = [a.upper() for a in assets]
            matched_cols = []
            for col in df.columns:
                col_upper = col.upper()
                if any(col_upper.startswith(f"{a}_") or col_upper == a for a in req_assets):
                    matched_cols.append(col)
                elif not any(col_upper.startswith(f"{x}_") for x in DEFAULT_ASSETS):
                    matched_cols.append(col)  # preserva feature non associate ad asset specifici
            if matched_cols:
                df = df[matched_cols]

        return df

    @classmethod
    def load_macro_data(
        cls,
        decision_dates: Optional[pd.DatetimeIndex] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        root_dir: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Carica e sintetizza le serie storiche macroeconomiche FRED (tassi, curve spreads, inflazione, VIX).
        Allinea causalmente le osservazioni sulle date fornite (evitando lookahead bias).
        """
        root = get_project_root(root_dir)
        macro_dir = root / "data" / "raw" / "macro"

        if not macro_dir.is_dir():
            raise FileNotFoundError(f"Directory dati macro non trovata in: {macro_dir}")

        macro_series: Dict[str, pd.Series] = {}
        for csv_file in macro_dir.glob("*.csv"):
            try:
                s_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                val_col = s_df.columns[0]
                num_series = pd.to_numeric(s_df[val_col], errors="coerce")
                macro_series[csv_file.stem] = num_series
            except Exception:
                continue

        # Costruzione DataFrame giornaliero dei tassi e indicatori base
        daily_macro = pd.DataFrame()
        for name, s in macro_series.items():
            daily_macro[name] = s

        daily_macro = daily_macro.sort_index().ffill(limit=5)

        # Calcolo indicatori derivati (curve spread, variazioni, inflazione)
        calc_df = pd.DataFrame(index=daily_macro.index)

        # 1. Pendenze di curva
        if "DGS10" in daily_macro and "DGS2" in daily_macro:
            calc_df["curve_10y_2y"] = daily_macro["DGS10"] - daily_macro["DGS2"]
            calc_df["curve_10y_2y_change_1w"] = calc_df["curve_10y_2y"].diff(5)

        if "DGS10" in daily_macro and "DGS3MO" in daily_macro:
            calc_df["curve_10y_3m"] = daily_macro["DGS10"] - daily_macro["DGS3MO"]
            calc_df["curve_10y_3m_change_1w"] = calc_df["curve_10y_3m"].diff(5)

        if "DGS30" in daily_macro and "DGS10" in daily_macro:
            calc_df["curve_30y_10y"] = daily_macro["DGS30"] - daily_macro["DGS10"]
            calc_df["curve_30y_10y_change_1w"] = calc_df["curve_30y_10y"].diff(5)

        # 2. Variazioni tassi 1w (~5 sedute lavorative)
        if "DGS10" in daily_macro:
            calc_df["treasury_10y_change_1w"] = daily_macro["DGS10"].diff(5)
            calc_df["treasury_10y"] = daily_macro["DGS10"]
        if "DGS2" in daily_macro:
            calc_df["treasury_2y_change_1w"] = daily_macro["DGS2"].diff(5)
            calc_df["treasury_2y"] = daily_macro["DGS2"]
        if "DGS3MO" in daily_macro:
            calc_df["treasury_3m_change_1w"] = daily_macro["DGS3MO"].diff(5)

        # 3. Policy rate Fed
        if "FEDFUNDS" in daily_macro:
            calc_df["rates__policy_rate"] = daily_macro["FEDFUNDS"].ffill()

        # 4. Inflazione YoY (CPIAUCSL shifted con ritardo prudenziale di 45 giorni per evitare lookahead)
        if "CPIAUCSL" in daily_macro:
            cpi_clean = daily_macro["CPIAUCSL"].dropna()
            cpi_yoy_raw = cpi_clean.pct_change(12)
            cpi_avail_idx = cpi_yoy_raw.index + pd.Timedelta(days=45)
            cpi_lagged = pd.Series(cpi_yoy_raw.values, index=cpi_avail_idx).sort_index()
            # Riallinea sull'indice giornaliero tramite forward fill
            calc_df["cpi_yoy"] = cpi_lagged.reindex(calc_df.index, method="ffill")

        # 5. Volatilità VIX
        if "VIXCLS" in daily_macro:
            calc_df["vix_level"] = daily_macro["VIXCLS"]
            calc_df["vix_change_4w"] = daily_macro["VIXCLS"].diff(20)

        # 6. Breakeven inflation
        if "T10YIE" in daily_macro:
            calc_df["breakeven_10y"] = daily_macro["T10YIE"]
        if "T5YIE" in daily_macro:
            calc_df["breakeven_5y_change_1w"] = daily_macro["T5YIE"].diff(5)

        # Allineamento con decision_dates (se fornite) o restituzione giornaliera
        if decision_dates is not None:
            # Allineamento causale merge_asof backwards
            dec_df = pd.DataFrame({"Date": decision_dates}).sort_values("Date")
            calc_clean = calc_df.dropna(how="all").reset_index().rename(columns={"index": "Date"})
            aligned = pd.merge_asof(dec_df, calc_clean, on="Date", direction="backward").set_index("Date")
            result = aligned
        else:
            result = calc_df

        if start_date is not None:
            result = result[result.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            result = result[result.index <= pd.Timestamp(end_date)]

        return result

    @classmethod
    def load_complete_dataset(
        cls,
        assets: Optional[Sequence[str]] = None,
        frequency: str = "weekly",
        include_macro: bool = True,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        root_dir: Optional[Union[str, Path]] = None,
        save_features_weekly_csv: bool = False,
    ) -> pd.DataFrame:
        """
        Carica il dataset completo congiunto di mercato (SPY, IEF, GLD, SHY) e feature macro allineate.
        Se save_features_weekly_csv=True o se data/processed/features_weekly.csv non esiste,
        può salvare una copia per compatibilità retroattiva con i moduli preesistenti.
        """
        market_df = cls.load_market_data(
            frequency=frequency,
            assets=assets,
            start_date=start_date,
            end_date=end_date,
            root_dir=root_dir,
        )

        if include_macro:
            try:
                macro_df = cls.load_macro_data(
                    decision_dates=market_df.index,
                    root_dir=root_dir,
                )
                complete_df = market_df.join(macro_df, how="left")
            except Exception as e:
                print(f"[BaseAgent.load_complete_dataset] Avviso: caricamento macro fallito ({e}), uso solo dati di mercato.")
                complete_df = market_df
        else:
            complete_df = market_df

        if save_features_weekly_csv:
            root = get_project_root(root_dir)
            target_path = root / "data" / "processed" / "features_weekly.csv"
            target_path.parent.mkdir(parents=True, exist_ok=True)
            complete_df.to_csv(target_path)
            print(f"[BaseAgent] Esportato dataset compatibile in: {target_path}")

        return complete_df

    @staticmethod
    def prepare_asset_data(
        df: pd.DataFrame,
        asset: str,
        price_col: str = "close",
    ) -> pd.DataFrame:
        """
        Estrae dal DataFrame multi-asset una vista standardizzata per un singolo asset (es. 'SPY'):
        - Crea colonne canoniche: 'open', 'high', 'low', 'close', 'volume'
        - Mantiene le colonne originarie con prefisso ('SPY_close', ecc.)
        - Mantiene tutte le colonne di feature globali o macro (es. 'curve_10y_2y', 'vix_level')
        """
        asset_u = asset.upper()
        out_df = df.copy()

        # Mappatura campi standard
        for standard_name in ["open", "high", "low", "close", "volume"]:
            source_col = f"{asset_u}_{standard_name}"
            if source_col in out_df.columns:
                out_df[standard_name] = out_df[source_col]
            elif f"{asset.lower()}_{standard_name}" in out_df.columns:
                out_df[standard_name] = out_df[f"{asset.lower()}_{standard_name}"]

        return out_df

    def load_data(
        self,
        asset: Optional[str] = None,
        frequency: str = "weekly",
        include_macro: bool = True,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        root_dir: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Metodo di istanza per caricare direttamente i dati preparati per l'agente.
        Se target_asset è configurato nell'agente, adatta automaticamente le colonne.
        """
        target = (asset or self.target_asset or "").upper()
        df = self.load_complete_dataset(
            frequency=frequency,
            include_macro=include_macro,
            start_date=start_date,
            end_date=end_date,
            root_dir=root_dir,
        )

        if target:
            return self.prepare_asset_data(df, asset=target, price_col=self.price_col)
        return df

    # =====================================================================
    # GESTIONE PARAMETRI E SERIALIZZAZIONE
    # =====================================================================
    def load_params(self, filepath: Union[str, Path]) -> "BaseAgent":
        """
        Carica i parametri ottimali da un file JSON e aggiorna lo stato dell'agente.

        Parametri:
            filepath: Percorso del file JSON contenente i parametri.
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
        """
        Salva i parametri attuali dell'agente in un file JSON.

        Parametri:
            filepath: Percorso di destinazione del file JSON.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=4)

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """
        Hook opzionale: aggiorna gli attributi specifici dell'agente derivato
        a partire dal dizionario dei parametri caricato.
        """
        if "target_asset" in params:
            self.target_asset = str(params["target_asset"]).upper()
        if "price_col" in params:
            self.price_col = str(params["price_col"])

    def _format_signals(self, signals: pd.Series) -> pd.Series:
        """
        Normalizza e valida la serie dei segnali:
        - Sostituisce eventuali NaN con 0 (HOLD/FLAT)
        - Converte in interi {-1, 0, 1}
        - Mappa nei valori stringa se output_format == 'string'
        """
        formatted = signals.fillna(0).astype(int)

        # Controllo di conformità sui valori ammessi dal progetto (-1, 0, 1)
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
# FUNZIONI DI UTILITÀ PER METRICHE DI PERFORMANCE E STABILITÀ
# =====================================================================
def calculate_performance_metrics(
    returns: pd.Series,
    annualization_factor: int = 52,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Calcola le metriche complete di performance e rischio/stabilità per una serie di rendimenti periodici.

    Parametri:
        returns: Serie di rendimenti periodici (settimanali).
        annualization_factor: Fattore di annualizzazione (default 52 per frequenza settimanale).
        risk_free_rate: Tasso privo di rischio annualizzato (default 0.0).

    Ritorna:
        Dizionario con metriche: cumulative_return, annualized_return, annualized_volatility,
        sharpe_ratio, max_drawdown, profit_factor, calmar_ratio.
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
    """
    Calcola lo score di ottimizzazione bilanciando rendimento e stabilità/rischio.

    Opzioni supportate:
    - 'sharpe_mdd' (default): Sharpe ratio penalizzato per il Maximum Drawdown.
        Premia strategie con curve di equity regolari e perdite massime contenute.
        Score = Sharpe * (1 - MDD) se Sharpe >= 0, altrimenti Sharpe * (1 + MDD).
    - 'sharpe': Sharpe ratio annualizzato puro.
    - 'calmar': Calmar ratio (rendimento annualizzato / max drawdown).
    - 'return': Rendimento cumulato puro (non include penalità sul rischio).
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
    else:
        raise ValueError(f"Metrica obiettivo '{metric_type}' non riconosciuta.")

"""
signal_generation.py 

Generazione dei segnali discreti per la baseline Macro / Rates Regimes.

Questo modulo trasforma i regimi economici identificati in regime_model.py in segnali di trading:

    - SHY: breve duration;
    - IEF: duration intermedia;
    - TLT: lunga duration.

La baseline è volutamente semplice, interpretabile e NON ottimizzata sui dati.

Di seguito la logica economica di fondo:
 1. tassi in calo e inflazione in calo -> maggiore preferenza per duration lunga;
 2. tassi in aumento e inflazione in aumento -> minore preferenza per duration lunga;
 3. segnali macro misti -> preferenza per duration intermedia;
 4. volatilità elevata -> riduzione della convinzione direzionale;
 5. curva invertita -> lieve supporto alla duration, perché può essere coerente
   con aspettative di rallentamento/easing, ma senza farne un driver dominante.

L'output finale per ciascun asset è sempre:
+1 = long
 0 = flat
-1 = short

"""

from pathlib import Path
from typing import Tuple

import pandas as pd


# =============================================================================
# Lettura del dataset dei regimi
# =============================================================================

def load_regime_dataset(config: dict, root_dir: Path) -> pd.DataFrame:
    """Carica il dataset settimanale prodotto da regime_model.py."""
    path = root_dir / config["paths"]["processed_data"] / "regimes_weekly.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Regime dataset not found: {path}"
        )

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    return df


# =============================================================================
# Parametri del baseline
# =============================================================================

def _get_signal_parameters(config: dict) -> dict:
    """
    Recupera i parametri del baseline dal config.yaml.

    I valori di default NON sono ottimizzati sul campione.
    Sono semplicemente regole economiche iniziali, utili come benchmark.
    """
    signal_config = config.get("signals", {})

    return {
        "long_duration_threshold": int(
            signal_config.get("long_duration_threshold", 2)
        ),
        "short_duration_threshold": int(
            signal_config.get("short_duration_threshold", -2)
        ),
    }


# =============================================================================
# Score macro di duration
# =============================================================================

def build_duration_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce uno score sintetico di preferenza per la duration.

    Interpretazione dello score:

    score positivo  -> contesto più favorevole alla duration lunga;
    score negativo  -> contesto meno favorevole alla duration lunga;
    score vicino a 0 -> quadro macro misto.

    Pesi del baseline:
    - rates falling  : +2
    - rates rising   : -2
    - inflation falling : +1
    - inflation rising  : -1
    - curve inverted : +1
    - curve normal   : 0

    La dinamica dei tassi riceve peso maggiore perché rappresenta il driver
    più diretto del prezzo dei bond a duration elevata.
    """
    signals = df.copy()

    required_columns = [
        "inflation_state",
        "rate_state",
        "curve_state",
        "volatility_state",
    ]

    missing = [
        column
        for column in required_columns
        if column not in signals.columns
    ]

    if missing:
        raise KeyError(
            f"Missing regime columns: {', '.join(missing)}"
        )

    # Lo score parte da zero e viene modificato solo quando tutti gli stati
    # necessari risultano disponibili.
    signals["duration_score_raw"] = 0.0

    valid = signals[required_columns].notna().all(axis=1)

    # -------------------------------------------------------------------------
    # Driver principale: direzione dei tassi
    # -------------------------------------------------------------------------
    signals.loc[
        valid & (signals["rate_state"] == "falling"),
        "duration_score_raw",
    ] += 2

    signals.loc[
        valid & (signals["rate_state"] == "rising"),
        "duration_score_raw",
    ] -= 2

    # -------------------------------------------------------------------------
    # Secondo driver: dinamica dell'inflazione
    # -------------------------------------------------------------------------
    signals.loc[
        valid & (signals["inflation_state"] == "falling"),
        "duration_score_raw",
    ] += 1

    signals.loc[
        valid & (signals["inflation_state"] == "rising"),
        "duration_score_raw",
    ] -= 1

    # -------------------------------------------------------------------------
    # Curva invertita: lieve supporto alla duration
    # -------------------------------------------------------------------------
    signals.loc[
        valid & (signals["curve_state"] == "inverted"),
        "duration_score_raw",
    ] += 1

    # -------------------------------------------------------------------------
    # Volatilità elevata: riduzione della convinzione
    # -------------------------------------------------------------------------
    # In presenza di alta volatilità non cambiamo il segno dello score:
    # lo avviciniamo semplicemente a zero di un punto.
    signals["duration_score"] = signals["duration_score_raw"]

    high_vol = valid & (signals["volatility_state"] == "high")

    positive_high_vol = high_vol & (signals["duration_score"] > 0)
    negative_high_vol = high_vol & (signals["duration_score"] < 0)

    signals.loc[
        positive_high_vol,
        "duration_score",
    ] -= 1

    signals.loc[
        negative_high_vol,
        "duration_score",
    ] += 1

    # Se non abbiamo un regime completo, lo score non deve essere utilizzato.
    signals.loc[~valid, "duration_score_raw"] = pd.NA
    signals.loc[~valid, "duration_score"] = pd.NA

    return signals


# =============================================================================
# Conversione dello score in segnali discreti
# =============================================================================

def generate_asset_signals(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    
    """
    Converte lo score macro in segnali discreti per SHY, IEF e TLT.

    Regola iniziale:

    score >= soglia positiva
        -> preferenza per lunga duration:
           TLT long

    score <= soglia negativa
        -> preferenza per breve duration:
           SHY long, TLT short

    score compreso tra le due soglie
        -> quadro intermedio:
           IEF long

    Questa è una scelta baseline e non rappresenta ancora il sistema finale.
    """
    signals = df.copy()
    params = _get_signal_parameters(config)

    long_threshold = params["long_duration_threshold"]
    short_threshold = params["short_duration_threshold"]

    if long_threshold <= 0:
        raise ValueError(
            "long_duration_threshold must be positive."
        )

    if short_threshold >= 0:
        raise ValueError(
            "short_duration_threshold must be negative."
        )

    # Inizializziamo i tre asset a flat.
    signals["SHY_signal"] = 0
    signals["IEF_signal"] = 0
    signals["TLT_signal"] = 0

    valid = signals["duration_score"].notna()

    long_duration = (
        valid
        & (signals["duration_score"] >= long_threshold)
    )

    short_duration = (
        valid
        & (signals["duration_score"] <= short_threshold)
    )

    intermediate_duration = (
        valid
        & ~long_duration
        & ~short_duration
    )

    # -------------------------------------------------------------------------
    # Regime favorevole alla lunga duration
    # -------------------------------------------------------------------------
    signals.loc[long_duration, "TLT_signal"] = 1

    # -------------------------------------------------------------------------
    # Regime ostile alla lunga duration
    # -------------------------------------------------------------------------
    signals.loc[short_duration, "SHY_signal"] = 1
    signals.loc[short_duration, "TLT_signal"] = -1

    # -------------------------------------------------------------------------
    # Regime macro misto
    # -------------------------------------------------------------------------
    signals.loc[intermediate_duration, "IEF_signal"] = 1

    # Verifica formale richiesta dalle istruzioni del prof: solo -1, 0, +1.
    allowed_values = {-1, 0, 1}

    for column in [
        "SHY_signal",
        "IEF_signal",
        "TLT_signal",
    ]:
        observed = set(signals[column].dropna().unique())

        if not observed.issubset(allowed_values):
            raise ValueError(
                f"Invalid values found in {column}: {observed}"
            )

    return signals


# =============================================================================
# Report dei segnali
# =============================================================================

def build_signal_summary(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Conta la frequenza dei segnali generati per ciascun asset.

    Il report permette di verificare rapidamente se il baseline produce
    comportamenti degeneri, ad esempio un asset sempre long o sempre flat.
    """
    rows = []

    for asset in ["SHY", "IEF", "TLT"]:
        column = f"{asset}_signal"

        # Per il report consideriamo solo le settimane in cui il regime
        # macro e il duration score sono effettivamente disponibili.
        valid = signals["duration_score"].notna()

        counts = (
            signals.loc[valid, column]
            .value_counts()
            .reindex([-1, 0, 1], fill_value=0)
        )

        total = counts.sum()

        for signal_value, count in counts.items():
            rows.append(
                {
                    "asset": asset,
                    "signal": int(signal_value),
                    "weeks": int(count),
                    "share_pct": (
                        count / total * 100
                        if total > 0
                        else 0.0
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Pipeline del modulo
# =============================================================================

def run_signal_generation(
    config: dict,
    root_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Esegue il baseline di signal generation e salva gli output.

    Output:
        data/signals/signals_weekly.csv
        data/signals/signal_summary.csv
    """
    signals_dir = root_dir / config["paths"]["signals"]
    signals_dir.mkdir(parents=True, exist_ok=True)

    regimes = load_regime_dataset(config, root_dir)

    signals = build_duration_score(regimes)
    signals = generate_asset_signals(signals, config)

    summary = build_signal_summary(signals)

    signals_path = signals_dir / "signals_weekly.csv"
    summary_path = signals_dir / "signal_summary.csv"

    signals.to_csv(signals_path)
    summary.to_csv(summary_path, index=False)

    valid_count = signals["duration_score"].notna().sum()

    print("\n[Signal Generation] Baseline completed successfully")
    print(f"  Dataset:     {signals_path}")
    print(f"  Valid weeks: {valid_count:,}")
    print(f"  Summary:     {summary_path}")

    return signals, summary

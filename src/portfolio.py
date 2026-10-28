"""Costruzione dei pesi target per le strategie Multi-Asset.

Questo modulo sostituisce la precedente logica Treasury basata sui segnali
SHY/IEF/TLT e su ``duration_score``.

Livelli supportati:
- A1: portafogli statici e a ribilanciamento settimanale definiti nel YAML;
- A2: controllo della volatilita' mediante ottimizzazione CVXPY.

Responsabilita':
1. leggere il pannello giornaliero prodotto da ``data_transformation.py``;
2. selezionare le date di decisione effettive, dopo la chiusura di mercato;
3. generare i pesi target senza utilizzare informazioni future;
4. salvare pesi, diagnostiche e riepiloghi separati per strategia.

Non esegue operazioni: il backtester dovra' applicare i pesi alla prima
apertura successiva. Non calcola rendimenti o metriche di performance.
Durante questa fase genera segnali solo nel periodo di TRAINING.

Interfaccia principale: ``run_portfolio(config, root_dir, level=None)``.
Restituisce un dizionario ``nome_strategia -> DataFrame dei pesi target``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_transformation import load_dataset
from src.time_split import get_time_windows, restrict_to_allowed_period
from src.utils import get_tickers


LEVEL_A_ASSETS = frozenset({"SPY", "IEF", "GLD", "SHY"})
REQUIRED_PRICE_FIELDS = ("open", "close")


def _validate_universe(config: dict) -> list[str]:
    """Controlla che A1/A2 utilizzino i quattro ETF e i vincoli concordati."""
    tickers = get_tickers(config)

    if len(tickers) != 4 or set(tickers) != LEVEL_A_ASSETS:
        raise ValueError(
            "I livelli A1/A2 richiedono esattamente SPY, IEF, GLD e SHY "
            "nell'universo investibile."
        )

    portfolio = config.get("portfolio", {})
    if not isinstance(portfolio, dict):
        raise TypeError("La sezione 'portfolio' deve essere un dizionario.")

    if (
        portfolio.get("long_only") is not True
        or portfolio.get("fully_invested") is not True
        or portfolio.get("allow_short") is not False
        or float(portfolio.get("leverage", 0)) != 1.0
    ):
        raise ValueError(
            "A1/A2 richiedono long_only=true, fully_invested=true, "
            "allow_short=false e leverage=1.0."
        )

    return tickers


def _validate_time_index(data: pd.DataFrame, name: str) -> None:
    """Esclude date ambigue, duplicate o non ordinate."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(f"{name}: l'indice deve contenere date.")
    if data.index.hasnans or data.index.tz is not None:
        raise ValueError(f"{name}: l'indice contiene date non valide.")
    if not data.index.is_monotonic_increasing or data.index.has_duplicates:
        raise ValueError(f"{name}: date duplicate o non ordinate.")


def load_portfolio_inputs(
    config: dict,
    root_dir: Path,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Legge i prezzi rettificati e il calendario delle decisioni settimanali.

    I dati antecedenti all'inizio del training restano disponibili soltanto
    per il calcolo delle finestre storiche, per esempio la covarianza A2.
    Le osservazioni di validation/OOS non vengono usate per creare i pesi.

    La validita' economica dei prezzi rettificati dipende dall'audit
    precedente di acquisizione e trasformazione dei dati.
    """
    root_dir = Path(root_dir)
    tickers = _validate_universe(config)
    windows = get_time_windows(config)
    train_start = windows["train_start"]
    train_end = windows["train_end"]

    if train_start is None:
        raise ValueError("Per il progetto Multi-Asset serve 'validation.train_start'.")

    processed = root_dir / config["paths"]["processed_data"]
    daily = load_dataset(processed / "market_daily.csv", strict=True)
    decisions_frame = load_dataset(
        processed / "weekly_decision_dates.csv", strict=True
    )

    _validate_time_index(daily, "market_daily.csv")
    _validate_time_index(decisions_frame, "weekly_decision_dates.csv")

    expected_columns = [
        f"{ticker}_{field}"
        for ticker in tickers
        for field in REQUIRED_PRICE_FIELDS
    ]
    missing = [column for column in expected_columns if column not in daily]
    if missing:
        raise KeyError(
            "market_daily.csv: mancano le colonne " + ", ".join(missing)
        )

    # Usa solo le colonne necessarie, senza inventare valori per sedute mancanti.
    daily = daily.loc[:, expected_columns].copy()
    for column in expected_columns:
        daily[column] = pd.to_numeric(daily[column], errors="raise")
    values = daily.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Prezzi giornalieri mancanti, non finiti o non positivi.")

    if "decision_date" not in decisions_frame.columns:
        raise KeyError("weekly_decision_dates.csv: manca 'decision_date'.")

    # L'indice e la colonna riportano la stessa seduta effettiva.
    decision_values = pd.to_datetime(
        decisions_frame["decision_date"], errors="raise"
    )
    if decision_values.isna().any() or not (
        pd.DatetimeIndex(decision_values) == decisions_frame.index
    ).all():
        raise ValueError("Date di decisione incoerenti nel file settimanale.")

    # Non basta disattivare l'OOS nel config: restringiamo anche esplicitamente
    # gli input al training, senza accedere alle osservazioni successive.
    daily = restrict_to_allowed_period(daily, config)
    daily = daily.loc[daily.index <= train_end].copy()
    decisions = decisions_frame.index[
        (decisions_frame.index >= train_start)
        & (decisions_frame.index <= train_end)
    ]
    decisions = pd.DatetimeIndex(decisions, name="Date")

    if daily.empty or decisions.empty:
        raise ValueError("Il training non contiene prezzi o date di decisione.")
    if daily.index.min() > train_start:
        raise ValueError("Dati storici insufficienti per iniziare il training.")
    if not decisions.isin(daily.index).all():
        raise ValueError("Una decisione non corrisponde a una seduta di mercato.")

    # Nel backtest ogni decisione sara' applicata alla seduta successiva.
    # Una decisione dell'ultima seduta di training potra' essere scartata
    # dal backtester, evitando operazioni nella validation.
    return daily, decisions


def _validated_weights(
    allocations: dict,
    tickers: list[str],
    strategy_name: str,
) -> pd.Series:
    """Verifica i pesi iniziali di un benchmark A1."""
    if not isinstance(allocations, dict):
        raise TypeError(f"{strategy_name}: 'weights' deve essere un dizionario.")

    unknown = set(allocations) - set(tickers)
    if unknown:
        raise ValueError(f"{strategy_name}: ticker non ammessi: {sorted(unknown)}")

    try:
        weights = pd.Series(
            {ticker: float(allocations.get(ticker, 0.0)) for ticker in tickers},
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{strategy_name}: pesi non numerici.") from exc

    values = weights.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
        or not np.isclose(values.sum(), 1.0, atol=1e-10, rtol=0)
    ):
        raise ValueError(
            f"{strategy_name}: i pesi devono essere compresi tra 0 e 1 "
            "e sommare esattamente a 1."
        )

    return weights


def build_a1_weights(
    config: dict,
    decision_dates: pd.DatetimeIndex,
    tickers: list[str],
) -> dict[str, pd.DataFrame]:
    """Genera i pesi A1 senza previsioni o informazioni macroeconomiche.

    ``rebalance: weekly`` produce un target a ogni decisione settimanale.
    ``rebalance: none`` produce un solo ordine iniziale (buy-and-hold):
    i pesi effettivi potranno poi variare naturalmente con i prezzi.
    """
    benchmark_config = config.get("benchmarks", {})
    if not isinstance(benchmark_config, dict) or not benchmark_config:
        raise ValueError("La sezione 'benchmarks' deve contenere i portafogli A1.")

    a1_config = config.get("strategies", {}).get("A1", {})
    selected_names = a1_config.get("benchmark_names", list(benchmark_config))
    if not isinstance(selected_names, list) or not selected_names:
        raise ValueError("La lista dei benchmark A1 non e' valida.")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("La lista dei benchmark A1 contiene duplicati.")

    portfolios: dict[str, pd.DataFrame] = {}
    for name in selected_names:
        if name not in benchmark_config:
            raise KeyError(f"Benchmark A1 non definito nel YAML: {name}")
        settings = benchmark_config[name]
        if not isinstance(settings, dict):
            raise TypeError(f"Benchmark A1 '{name}': impostazioni non valide.")

        # Accetta entrambe le denominazioni emerse durante la migrazione.
        allocations = settings.get("weights", settings.get("target_weights"))
        weight_vector = _validated_weights(allocations, tickers, name)
        rebalance = settings.get("rebalance")

        if rebalance == "weekly":
            dates = decision_dates
        elif rebalance == "none":
            dates = decision_dates[:1]
        else:
            raise ValueError(
                f"Benchmark '{name}': rebalance deve essere 'weekly' o 'none'."
            )

        weights = pd.DataFrame(
            np.tile(weight_vector.to_numpy(), (len(dates), 1)),
            index=pd.DatetimeIndex(dates, name="Date"),
            columns=tickers,
        )
        portfolios[name] = weights

    return portfolios


def _get_a2_parameters(config: dict) -> tuple[int, float]:
    """Legge A2 e verifica che i parametri coincidano con l'allocatore attuale.

    Evita che una modifica silenziosa al YAML dia l'impressione di aver
    modificato un vincolo che ``multiasset_allocator.py`` non supporta.
    """
    settings = config.get("strategies", {}).get("A2", {})
    if not isinstance(settings, dict):
        raise TypeError("La sezione 'strategies.A2' deve essere un dizionario.")

    window = int(
        settings.get(
            "covariance_window",
            settings.get("covariance_lookback_trading_days", 126),
        )
    )
    volatility = float(
        settings.get(
            "target_volatility",
            settings.get("maximum_estimated_annual_volatility", 0.08),
        )
    )
    annualization = int(
        settings.get(
            "annualization_factor",
            settings.get("covariance_annualization_factor", 252),
        )
    )
    optimizer = str(settings.get("optimizer", settings.get("optimization_library", "cvxpy")))
    solver = str(settings.get("solver", "CLARABEL"))
    fallback = str(
        settings.get(
            "fallback", settings.get("infeasible_fallback", "minimum_variance")
        )
    )
    objective = str(
        settings.get("objective", "minimum_squared_distance")
    )
    reference = settings.get(
        "reference_portfolio",
        settings.get("reference_benchmark", "equal_weight"),
    )

    # L'allocatore corrente parte sempre da pesi 25/25/25/25.
    # Una variazione del riferimento nel YAML deve essere segnalata,
    # non ignorata durante il calcolo.
    benchmarks = config.get("benchmarks", {})
    if reference not in benchmarks:
        raise ValueError(f"A2: benchmark di riferimento non trovato: {reference}")
    reference_config = benchmarks[reference]
    reference_weights = reference_config.get(
        "weights", reference_config.get("target_weights")
    )
    if (
        reference_config.get("rebalance") != "weekly"
        or not isinstance(reference_weights, dict)
        or any(
            not np.isclose(float(reference_weights.get(ticker, 0)), 0.25)
            for ticker in LEVEL_A_ASSETS
        )
    ):
        raise ValueError(
            "A2 richiede come riferimento un Equal Weight settimanale "
            "con peso 25% per ciascun ETF."
        )

    if objective not in {
        "minimum_squared_distance", "minimum_squared_distance_from_reference"
    }:
        raise ValueError("A2: funzione obiettivo non implementata.")

    if window < 2 or not np.isfinite(volatility) or volatility <= 0:
        raise ValueError("Parametri di finestra o volatilita' A2 non validi.")
    if annualization != 252 or optimizer.lower() != "cvxpy" or solver.upper() != "CLARABEL":
        raise ValueError(
            "L'allocatore A2 attuale richiede annualizzazione 252, "
            "CVXPY e risolutore CLARABEL."
        )
    if fallback != "minimum_variance":
        raise ValueError("L'allocatore A2 attuale supporta solo il fallback minimum_variance.")

    return window, volatility


# A2 validators and constants restored from the original allocator.
ASSET = ("SPY", "IEF", "GLD", "SHY")
SEDUTE_ANNUALI = 252
TOLLERANZA_PESI = 1e-7
TOLLERANZA_VOLATILITA = 1e-7


def _valida_rendimenti(
    rendimenti: pd.DataFrame,
    finestra: int,
    volatilita_massima: float,
) -> pd.DataFrame:
    """Controlla gli input senza modificare il DataFrame originale."""
    if not isinstance(rendimenti, pd.DataFrame):
        raise TypeError("I rendimenti devono essere un DataFrame pandas.")

    if isinstance(finestra, (bool, np.bool_)) or not isinstance(
        finestra, (int, np.integer)
    ):
        raise TypeError("La finestra deve essere un numero intero di sedute.")

    if finestra < 2:
        raise ValueError("La finestra deve contenere almeno due sedute.")

    if isinstance(volatilita_massima, (bool, np.bool_)):
        raise TypeError("Il limite di volatilità deve essere numerico.")

    try:
        vol_limite = float(volatilita_massima)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Il limite di volatilità non è valido.") from exc

    if not np.isfinite(vol_limite) or vol_limite <= 0:
        raise ValueError("Il limite di volatilità deve essere finito e positivo.")

    if len(rendimenti) < finestra:
        raise ValueError("Osservazioni insufficienti per la finestra indicata.")

    if not isinstance(rendimenti.index, pd.DatetimeIndex):
        raise TypeError("È richiesto un indice temporale DatetimeIndex.")

    if rendimenti.index.hasnans or rendimenti.index.tz is not None:
        raise ValueError("Le date devono essere valide e senza fuso orario.")

    if not rendimenti.index.is_monotonic_increasing:
        raise ValueError("Le date non sono ordinate.")

    if rendimenti.index.has_duplicates:
        raise ValueError("Sono presenti date duplicate.")

    if rendimenti.columns.has_duplicates:
        raise ValueError("Sono presenti ticker duplicati.")

    if not set(ASSET).issubset(rendimenti.columns):
        raise ValueError("Mancano uno o più ETF richiesti: SPY, IEF, GLD, SHY.")

    # La selezione avviene dopo il controllo delle date: nessuna risistemazione
    # automatica dell'ordine o riempimento di valori mancanti.
    dati = rendimenti.loc[:, list(ASSET)].tail(finestra).copy()

    try:
        valori = dati.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("I rendimenti devono essere numerici.") from exc

    if not np.isfinite(valori).all():
        raise ValueError("Rendimenti mancanti o non validi.")

    return dati.astype(float)


def _verifica_soluzione(pesi: np.ndarray) -> np.ndarray:
    """Valida i pesi restituiti dal risolutore, senza alterarli."""
    if pesi is None:
        raise RuntimeError("Il risolutore non ha restituito alcuna soluzione.")

    try:
        valori = np.asarray(pesi, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Pesi restituiti dal risolutore non numerici.") from exc

    if (
        len(valori) != len(ASSET)
        or not np.isfinite(valori).all()
        or abs(valori.sum() - 1.0) > TOLLERANZA_PESI
        or np.any(valori < -TOLLERANZA_PESI)
        or np.any(valori > 1.0 + TOLLERANZA_PESI)
    ):
        raise RuntimeError("La soluzione viola i vincoli sui pesi.")

    # Non normalizziamo automaticamente: correggere una soluzione errata
    # nasconderebbe problemi di convergenza del risolutore.
    return valori


def calcola_pesi_a2(
    rendimenti: pd.DataFrame,
    finestra: int = 126,
    volatilita_massima: float = 0.08,
) -> tuple[pd.Series, dict]:
    """Restituisce i pesi target A2 e le diagnostiche dell'ottimizzazione.

    I rendimenti devono essere giornalieri e disponibili alla decisione.
    La funzione usa solo le ultime ``finestra`` osservazioni ricevute.

    La volatilità stimata è ``sqrt(w.T @ Sigma @ w)``, con Sigma annualizzata.
    Se Equal Weight rispetta il limite, i suoi pesi rimangono invariati.
    Altrimenti si minimizza ``||w - w_EW||^2`` rispettando i vincoli.

    La diagnostica ``limite_rispettato`` può essere False esclusivamente
    quando il vincolo è non fattibile e si ricorre alla varianza minima.
    """
    # A1 does not need the optional convex optimizer dependency.
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError("A2 requires cvxpy and clarabel") from exc
    ASSET = ("SPY", "IEF", "GLD", "SHY")

    dati = _valida_rendimenti(rendimenti, finestra, volatilita_massima)
    limite = float(volatilita_massima)

    # La definizione della covarianza resta identica alla ricerca iniziale:
    # campionaria, usando 126 sedute e 252 giornate di negoziazione annue.
    cov = dati.cov().to_numpy(dtype=float) * SEDUTE_ANNUALI
    cov = (cov + cov.T) / 2.0
    if not np.isfinite(cov).all():
        raise ValueError("La matrice di covarianza non contiene valori finiti.")

    riferimento = np.full(len(ASSET), 1.0 / len(ASSET))

    def volatilita(pesi: np.ndarray) -> float:
        """Calcola la volatilità annualizzata della combinazione di pesi."""
        varianza = float(pesi @ cov @ pesi)
        return float(np.sqrt(max(varianza, 0.0)))

    vol_ew = volatilita(riferimento)

    if vol_ew <= limite:
        return (
            pd.Series(riferimento, index=ASSET),
            {
                "stato": "equal_weight",
                "volatilita_ew": vol_ew,
                "volatilita_target": vol_ew,
                "limite_rispettato": True,
            },
        )

    # La fattorizzazione PSD trasforma il vincolo di varianza in un vincolo
    # di norma, adatto a un problema convesso risolto da CLARABEL.
    autovalori, autovettori = np.linalg.eigh(cov)
    if autovalori.min() < -1e-10:
        raise ValueError("La matrice di covarianza non è semidefinita positiva.")

    autovalori = np.maximum(autovalori, 0.0)
    fattore = np.diag(np.sqrt(autovalori)) @ autovettori.T

    w = cp.Variable(len(ASSET))
    vincoli_base = [cp.sum(w) == 1, w >= 0]

    # La riscalatura in punti percentuali riduce i problemi numerici.
    rischio = cp.norm(100.0 * (fattore @ w), 2)
    problema = cp.Problem(
        cp.Minimize(cp.sum_squares(w - riferimento)),
        vincoli_base + [rischio <= 100.0 * limite],
    )

    try:
        problema.solve(solver=cp.CLARABEL)
    except cp.error.SolverError as exc:
        raise RuntimeError("CLARABEL non ha completato l'ottimizzazione A2.") from exc

    if problema.status in ("optimal", "optimal_inaccurate"):
        pesi = _verifica_soluzione(w.value)
        stato = "ottimizzato"
    elif problema.status in ("infeasible", "infeasible_inaccurate"):
        # Quando nessun portafoglio completamente investito rispetta l'8%,
        # scegliamo il minimo rischio ottenibile con gli stessi vincoli.
        minimo = cp.Problem(
            cp.Minimize(cp.sum_squares(100.0 * (fattore @ w))),
            vincoli_base,
        )
        try:
            minimo.solve(solver=cp.CLARABEL)
        except cp.error.SolverError as exc:
            raise RuntimeError("CLARABEL non ha completato il fallback A2.") from exc

        if minimo.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"Fallback Minimum Variance non risolto: {minimo.status}"
            )
        pesi = _verifica_soluzione(w.value)
        stato = "fallback_minimum_variance"
    else:
        raise RuntimeError(
            f"Ottimizzazione A2 non risolta: {problema.status}"
        )

    vol_finale = volatilita(pesi)
    limite_rispettato = vol_finale <= limite + TOLLERANZA_VOLATILITA

    # La varianza minima può superare il limite se il vincolo è impossibile;
    # una soluzione della prima ottimizzazione, invece, non può violarlo.
    if stato == "ottimizzato" and not limite_rispettato:
        raise RuntimeError("La soluzione viola il limite di volatilità.")

    return (
        pd.Series(pesi, index=ASSET),
        {
            "stato": stato,
            "volatilita_ew": vol_ew,
            "volatilita_target": vol_finale,
            "limite_rispettato": limite_rispettato,
        },
    )


def build_a2_weights(
    market_daily: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    config: dict,
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcola i target A2 usando solo rendimenti osservati entro ogni decisione.

    Il calcolo include il rendimento della seduta di decisione soltanto
    dopo la sua chiusura. Nessuna informazione successiva alla data corrente
    viene trasmessa all'ottimizzatore.
    """
    # The A2 optimizer is defined below; CVXPY loads only when needed.

    window, target_vol = _get_a2_parameters(config)
    closes = market_daily[[f"{ticker}_close" for ticker in tickers]].copy()
    closes.columns = tickers
    daily_returns = closes.pct_change(fill_method=None).iloc[1:]

    weights_rows = []
    diagnostics_rows = []
    for decision in decision_dates:
        past_returns = daily_returns.loc[:decision].tail(window)
        if len(past_returns) < window:
            raise ValueError(
                f"Storico insufficiente alla decisione {decision.date()}: "
                f"servono {window} rendimenti giornalieri."
            )

        weights, diagnostics = calcola_pesi_a2(
            past_returns,
            finestra=window,
            volatilita_massima=target_vol,
        )
        weights = weights.reindex(tickers)
        values = weights.to_numpy(dtype=float)

        if (
            not np.isfinite(values).all()
            or np.any(values < -1e-7)
            or np.any(values > 1 + 1e-7)
            or not np.isclose(values.sum(), 1.0, atol=1e-7)
        ):
            raise RuntimeError(f"Pesi A2 non validi alla data {decision.date()}.")

        weights_rows.append(pd.Series(values, index=tickers, name=decision))
        diagnostics_rows.append(
            {
                "Date": decision,
                "stato": diagnostics["stato"],
                "volatilita_ew": diagnostics["volatilita_ew"],
                "volatilita_target": diagnostics["volatilita_target"],
                "limite_rispettato": diagnostics["limite_rispettato"],
                "osservazioni_covarianza": len(past_returns),
            }
        )

    weights_frame = pd.DataFrame(weights_rows)
    weights_frame.index.name = "Date"
    diagnostics_frame = pd.DataFrame(diagnostics_rows).set_index("Date")
    return weights_frame, diagnostics_frame


def build_portfolio_summary(weights: pd.DataFrame) -> pd.DataFrame:
    """Riepiloga i pesi TARGET, senza confonderli con quelli effettivi."""
    if weights.empty:
        raise ValueError("Impossibile riepilogare un portafoglio senza decisioni.")

    rows = []
    for ticker in weights.columns:
        values = weights[ticker]
        rows.append(
            {
                "ticker": ticker,
                "peso_target_medio": float(values.mean()),
                "peso_target_minimo": float(values.min()),
                "peso_target_massimo": float(values.max()),
                "decisioni_con_esposizione": int((values > 1e-10).sum()),
            }
        )
    return pd.DataFrame(rows)


def run_portfolio(
    config: dict,
    root_dir: Path,
    level: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Esegue la costruzione dei target del livello A1 oppure A2.

    Salva ogni strategia in una cartella distinta:
        data/signals/A1/<nome>/target_weights.csv
        data/signals/A2/volatility_control/target_weights.csv

    In questo stadio la validation non viene elaborata e l'OOS e' bloccato.
    B e C non vengono eseguiti finche' non saranno realmente implementati.
    """
    selected_level = str(
        level or config.get("pipeline", {}).get("active_level", "A1")
    ).upper()

    if selected_level not in {"A1", "A2"}:
        raise NotImplementedError(
            f"Livello {selected_level} non ancora implementato: "
            "non vengono generati pesi fittizi per B/C."
        )

    level_config = config.get("strategies", {}).get(selected_level)
    if not isinstance(level_config, dict) or level_config.get("enabled") is not True:
        raise ValueError(f"Il livello {selected_level} non e' abilitato nel YAML.")

    tickers = _validate_universe(config)
    daily, decisions = load_portfolio_inputs(config, root_dir)

    diagnostics: dict[str, pd.DataFrame] = {}
    if selected_level == "A1":
        portfolios = build_a1_weights(config, decisions, tickers)
    else:
        weights, details = build_a2_weights(daily, decisions, config, tickers)
        portfolios = {"volatility_control": weights}
        diagnostics["volatility_control"] = details

    root_dir = Path(root_dir)
    signals_root = root_dir / config["paths"]["signals"] / selected_level

    for name, weights in portfolios.items():
        output = signals_root / name
        output.mkdir(parents=True, exist_ok=True)

        # Nomi distinti impediscono che una strategia sovrascriva un'altra.
        weights.to_csv(output / "target_weights.csv", index_label="Date")
        build_portfolio_summary(weights).to_csv(
            output / "target_summary.csv", index=False
        )
        if name in diagnostics:
            diagnostics[name].to_csv(output / "diagnostics.csv", index_label="Date")

        print(
            f"[Portfolio] {selected_level} / {name}: "
            f"{len(weights)} decisioni di training -> {output}"
        )

    return portfolios

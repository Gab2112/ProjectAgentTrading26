"""Target-weight construction for the multi-asset strategies.

This module replaces the earlier Treasury logic based on SHY/IEF/TLT
signals and on ``duration_score``.

Supported levels:
- A1: static and weekly-rebalanced portfolios defined in the YAML file;
- A2: volatility control through CVXPY optimisation;
- B: volatility cap for signed weights (``apply_volatility_cap_panel``).

Responsibilities:
1. read the daily panel produced by ``data_transformation.py``;
2. select the actual decision dates, after the market close;
3. generate target weights without using future information;
4. save weights, diagnostics and summaries separately for each strategy.

No trades are executed here: the backtester applies the weights at the
next available open. Returns and performance metrics are not computed.
In this phase signals are generated only for the TRAINING period.

Main interface: ``run_portfolio(config, root_dir, level=None)``.
Returns a dictionary ``strategy_name -> DataFrame of target weights``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.time_split import get_time_windows, restrict_to_allowed_period
from src.utils import ASSETS, get_tickers


REQUIRED_PRICE_FIELDS = ("open", "close")


def _validate_universe(config: dict) -> list[str]:
    """Check that A1/A2 use the four ETFs and the agreed constraints."""
    tickers = get_tickers(config)

    if len(tickers) != 4 or set(tickers) != set(ASSETS):
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
    """Reject ambiguous, duplicated or unsorted dates."""
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
    """Read adjusted prices and the weekly decision calendar.

    Data preceding the start of training remains available only to compute
    look-back windows, for example the A2 covariance. Validation and OOS
    observations are never used to build weights.

    The economic validity of adjusted prices depends on the preceding audit
    of data acquisition and transformation.
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

    # Use only the required columns, without inventing values for missing sessions.
    daily = daily.loc[:, expected_columns].copy()
    for column in expected_columns:
        daily[column] = pd.to_numeric(daily[column], errors="raise")
    values = daily.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Prezzi giornalieri mancanti, non finiti o non positivi.")

    if "decision_date" not in decisions_frame.columns:
        raise KeyError("weekly_decision_dates.csv: manca 'decision_date'.")

    # The index and the column refer to the same actual trading session.
    decision_values = pd.to_datetime(
        decisions_frame["decision_date"], errors="raise"
    )
    if decision_values.isna().any() or not (
        pd.DatetimeIndex(decision_values) == decisions_frame.index
    ).all():
        raise ValueError("Date di decisione incoerenti nel file settimanale.")

    # Disabling OOS in the configuration is not enough: inputs are also explicitly
    # restricted to the training period, without accessing later observations.
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

    # In the backtest, each decision is applied at the following session.
    # A decision taken on the last training session may therefore be discarded
    # by the backtester, avoiding trades inside the validation period.
    return daily, decisions


def _validated_weights(
    allocations: dict,
    tickers: list[str],
    strategy_name: str,
) -> pd.Series:
    """Validate the initial weights of an A1 benchmark."""
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
    """Generate A1 weights without forecasts or macroeconomic information.

    ``rebalance: weekly`` produces a target at every weekly decision.
    ``rebalance: none`` produces a single initial order (buy-and-hold):
    the effective weights then drift naturally with prices.
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

        # Accept both naming conventions used in the configuration.
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
    """Read the A2 settings and check that they match the current allocator.

    This prevents a silent change in the YAML file from suggesting that a
    constraint unsupported by the allocator has been modified.
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

    # The current allocator always starts from 25/25/25/25 weights.
    # A change of the reference weights in the YAML file must be reported,
    # not silently ignored during the computation.
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
            for ticker in ASSETS
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
SEDUTE_ANNUALI = 252
TOLLERANZA_PESI = 1e-7
TOLLERANZA_VOLATILITA = 1e-7


def _valida_rendimenti(
    rendimenti: pd.DataFrame,
    finestra: int,
    volatilita_massima: float,
) -> pd.DataFrame:
    """Validate the inputs without modifying the original DataFrame."""
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

    if not set(ASSETS).issubset(rendimenti.columns):
        raise ValueError("Mancano uno o più ETF richiesti: SPY, IEF, GLD, SHY.")

    # Columns are selected after the date checks: no automatic reordering
    # and no filling of missing values.
    dati = rendimenti.loc[:, list(ASSETS)].tail(finestra).copy()

    try:
        valori = dati.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("I rendimenti devono essere numerici.") from exc

    if not np.isfinite(valori).all():
        raise ValueError("Rendimenti mancanti o non validi.")

    return dati.astype(float)


def _verifica_soluzione(pesi: np.ndarray) -> np.ndarray:
    """Validate the weights returned by the solver without altering them."""
    if pesi is None:
        raise RuntimeError("Il risolutore non ha restituito alcuna soluzione.")

    try:
        valori = np.asarray(pesi, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Pesi restituiti dal risolutore non numerici.") from exc

    if (
        len(valori) != len(ASSETS)
        or not np.isfinite(valori).all()
        or abs(valori.sum() - 1.0) > TOLLERANZA_PESI
        or np.any(valori < -TOLLERANZA_PESI)
        or np.any(valori > 1.0 + TOLLERANZA_PESI)
    ):
        raise RuntimeError("La soluzione viola i vincoli sui pesi.")

    # Weights are not normalised automatically: correcting a wrong solution
    # would hide solver convergence problems.
    return valori


def calcola_pesi_a2(
    rendimenti: pd.DataFrame,
    finestra: int = 126,
    volatilita_massima: float = 0.08,
) -> tuple[pd.Series, dict]:
    """Return the A2 target weights and the optimisation diagnostics.

    Returns must be daily and available at the decision date. Only the last
    ``finestra`` observations received are used.

    Estimated volatility is ``sqrt(w.T @ Sigma @ w)``, with Sigma annualised.
    If the equal-weight portfolio satisfies the limit, its weights are kept.
    Otherwise ``||w - w_EW||^2`` is minimised subject to the constraints.

    The ``limite_rispettato`` diagnostic can be False only when the constraint
    is infeasible and the minimum-variance portfolio is used instead.
    """
    # A1 does not need the optional convex optimizer dependency.
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError("A2 requires cvxpy and clarabel") from exc
    
    dati = _valida_rendimenti(rendimenti, finestra, volatilita_massima)
    limite = float(volatilita_massima)

    # The covariance definition is identical to the initial research:
    # sample covariance over 126 sessions, annualised with 252 trading days.
    cov = dati.cov().to_numpy(dtype=float) * SEDUTE_ANNUALI
    cov = (cov + cov.T) / 2.0
    if not np.isfinite(cov).all():
        raise ValueError("La matrice di covarianza non contiene valori finiti.")

    riferimento = np.full(len(ASSETS), 1.0 / len(ASSETS))

    def volatilita(pesi: np.ndarray) -> float:
        """Compute the annualised volatility of the weight combination."""
        varianza = float(pesi @ cov @ pesi)
        return float(np.sqrt(max(varianza, 0.0)))

    vol_ew = volatilita(riferimento)

    if vol_ew <= limite:
        return (
            pd.Series(riferimento, index=ASSETS),
            {
                "stato": "equal_weight",
                "volatilita_ew": vol_ew,
                "volatilita_target": vol_ew,
                "limite_rispettato": True,
            },
        )

    # The PSD factorisation turns the variance constraint into a norm
    # constraint, suitable for a convex problem solved with CLARABEL.
    autovalori, autovettori = np.linalg.eigh(cov)
    if autovalori.min() < -1e-10:
        raise ValueError("La matrice di covarianza non è semidefinita positiva.")

    autovalori = np.maximum(autovalori, 0.0)
    fattore = np.diag(np.sqrt(autovalori)) @ autovettori.T

    w = cp.Variable(len(ASSETS))
    vincoli_base = [cp.sum(w) == 1, w >= 0]

    # Rescaling to percentage points reduces numerical issues.
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
        # When no fully invested portfolio satisfies the 8% limit,
        # the minimum risk achievable under the same constraints is selected.
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

    # Minimum variance may exceed the limit when the constraint is infeasible;
    # a solution of the first optimisation, instead, cannot violate it.
    if stato == "ottimizzato" and not limite_rispettato:
        raise RuntimeError("La soluzione viola il limite di volatilità.")

    return (
        pd.Series(pesi, index=ASSETS),
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
    """Compute A2 targets using only returns observed up to each decision.

    The return of the decision session is included only after its close.
    No information later than the current date reaches the optimiser.
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


# Volatility cap for signed weights (Level B): proportional scaling only,
# with no optimisation and no change in signal direction.

def apply_signed_volatility_cap(
    weights: pd.Series,
    daily_returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    window: int = 126,
    target_volatility: float = 0.08,
) -> tuple[pd.Series, dict]:
    """Cap estimated annualized volatility without changing signal directions."""

    if list(weights.index) != list(ASSETS):
        raise ValueError("Invalid weight universe")

    if list(daily_returns.columns) != list(ASSETS):
        raise ValueError("Invalid return universe")

    if (
        not isinstance(daily_returns.index, pd.DatetimeIndex)
        or daily_returns.index.has_duplicates
        or not daily_returns.index.is_monotonic_increasing
        or daily_returns.index.tz is not None
    ):
        raise ValueError("Invalid return dates")

    decision_date = pd.Timestamp(decision_date)

    if (
        decision_date.tz is not None
        or decision_date not in daily_returns.index
    ):
        raise ValueError("Invalid decision date")

    if window < 2 or not np.isfinite(target_volatility) or target_volatility <= 0:
        raise ValueError("Invalid risk parameters")

    w = weights.to_numpy(dtype=float)

    if (
        not np.isfinite(w).all()
        or np.abs(w).sum() > 1.0 + 1e-10
    ):
        raise ValueError("Invalid signed weights")

    # Only historical observations available at the decision close.
    history = daily_returns.loc[:decision_date].tail(window)

    if len(history) != window:
        raise ValueError("Insufficient return history")

    history = history.astype(float)

    if not np.isfinite(history.to_numpy()).all():
        raise ValueError("Invalid historical returns")

    covariance = history.cov().to_numpy(dtype=float) * 252

    variance = float(w @ covariance @ w)
    volatility_before = float(np.sqrt(max(variance, 0.0)))

    factor = (
        min(1.0, target_volatility / volatility_before)
        if volatility_before > 0
        else 1.0
    )

    adjusted = pd.Series(
        w * factor,
        index=ASSETS,
        dtype=float,
    )

    diagnostics = {
        "vol_before": volatility_before,
        "vol_after": volatility_before * factor,
        "scaling_factor": factor,
    }

    return adjusted, diagnostics


def apply_volatility_cap_panel(
    raw_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    *,
    window: int = 126,
    target_volatility: float = 0.08,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the existing point-in-time risk cap to every decision.

    This is an orchestration helper: the numerical algorithm and the three
    legacy risk-diagnostic columns stay unchanged for frozen B_v1/B_v2 CSVs.
    """
    if not isinstance(raw_weights, pd.DataFrame) or raw_weights.empty:
        raise ValueError("Expected nonempty target-weight panel")
    if list(raw_weights.columns) != list(ASSETS):
        raise ValueError("Invalid target-weight asset order")
    if not isinstance(raw_weights.index, pd.DatetimeIndex):
        raise ValueError("Target weights require dated decisions")
    if (
        raw_weights.index.hasnans
        or raw_weights.index.has_duplicates
        or not raw_weights.index.is_monotonic_increasing
        or raw_weights.index.tz is not None
    ):
        raise ValueError("Invalid target-weight decision dates")

    adjusted_rows: list[np.ndarray] = []
    risk_rows: list[dict] = []
    for date in raw_weights.index:
        adjusted, diagnostics = apply_signed_volatility_cap(
            weights=raw_weights.loc[date],
            daily_returns=daily_returns,
            decision_date=date,
            window=window,
            target_volatility=target_volatility,
        )
        adjusted_rows.append(adjusted.to_numpy(dtype=float))
        risk_rows.append(diagnostics)

    final_weights = pd.DataFrame(
        adjusted_rows, index=raw_weights.index, columns=ASSETS,
    )
    # Keep the frozen CSV column order: vol_before, vol_after, scaling_factor.
    risk = pd.DataFrame(risk_rows, index=raw_weights.index)
    return final_weights, risk


def build_portfolio_summary(weights: pd.DataFrame) -> pd.DataFrame:
    """Summarise the TARGET weights, keeping them distinct from the effective weights.
    """
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
    """Build the targets for Level A1 or A2.

    Each strategy is saved in a separate folder:
        data/signals/A1/<name>/target_weights.csv
        data/signals/A2/volatility_control/target_weights.csv

    At this stage validation is not processed and OOS is locked.
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

        # Distinct names prevent one strategy from overwriting another.
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

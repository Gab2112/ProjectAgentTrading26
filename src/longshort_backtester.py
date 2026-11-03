"""Shared A/B portfolio backtesting engine; one execution/accounting model."""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.regime_allocator import ASSETS


def schedule_next_open_executions(
    target_weights: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Schedule target weights for the next available trading Open."""

    if not isinstance(target_weights, pd.DataFrame):
        raise TypeError("Target weights must be a DataFrame")

    if list(target_weights.columns) != list(ASSETS):
        raise ValueError("Invalid asset universe")

    for index in (target_weights.index, trading_dates):
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("Expected DatetimeIndex")

        if (
            index.hasnans
            or index.has_duplicates
            or index.tz is not None
            or not index.is_monotonic_increasing
        ):
            raise ValueError("Invalid trading dates")

    if target_weights.empty or trading_dates.empty:
        raise ValueError("Empty input")

    values = target_weights.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Non-finite target weights")

    if (np.abs(values).sum(axis=1) > 1.0 + 1e-10).any():
        raise ValueError("Gross exposure exceeds 100%")

    positions = trading_dates.get_indexer(target_weights.index)

    if (positions < 0).any():
        raise ValueError("Decision date is not a trading session")

    # Execution happens at the next available session.
    execution_positions = positions + 1

    executable = execution_positions < len(trading_dates)

    if not executable.any():
        raise ValueError("No executable decisions")

    scheduled = target_weights.iloc[
        np.flatnonzero(executable)
    ].astype(float).copy()

    scheduled.index = pd.DatetimeIndex(
        trading_dates[execution_positions[executable]],
        name="Date",
    )

    return scheduled


def rebalance_at_open(
    shares: pd.Series,
    cash: float,
    open_prices: pd.Series,
    target_weights: pd.Series,
    commission_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> tuple[pd.Series, float, dict]:
    """Execute a self-financing Long/Short rebalance at market Open."""

    for name, series in (
        ("shares", shares),
        ("open_prices", open_prices),
        ("target_weights", target_weights),
    ):
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name} must be a Series")
        if list(series.index) != list(ASSETS):
            raise ValueError(f"Invalid asset order in {name}")

    q = shares.to_numpy(dtype=float)
    p = open_prices.to_numpy(dtype=float)
    w = target_weights.to_numpy(dtype=float)

    if (
        not np.isfinite(q).all()
        or not np.isfinite(p).all()
        or not np.isfinite(w).all()
        or not np.isfinite(cash)
        or (p <= 0).any()
        or np.abs(w).sum() > 1.0 + 1e-10
    ):
        raise ValueError("Invalid portfolio inputs")

    if (
        not np.isfinite(commission_bps)
        or not np.isfinite(slippage_bps)
        or commission_bps < 0
        or slippage_bps < 0
    ):
        raise ValueError("Invalid transaction costs")

    cost_rate = (commission_bps + slippage_bps) / 10000

    if cost_rate >= 1:
        raise ValueError("Transaction costs are excessive")

    current_notional = q * p
    nav_before = float(cash + current_notional.sum())

    if nav_before <= 0:
        raise ValueError("Non-positive portfolio NAV")

    # Exact self-financing equation:
    # NAV_after + trading_cost(NAV_after) = NAV_before.
    def balance(nav_after):
        traded = np.abs(nav_after * w - current_notional).sum()
        return nav_after + cost_rate * traded - nav_before

    if balance(0.0) >= 0:
        raise ValueError("No positive self-financing solution")

    nav_after = brentq(
        balance,
        0.0,
        nav_before,
        xtol=1e-10,
    )

    new_shares = nav_after * w / p
    traded_notional = float(
        np.abs((new_shares - q) * p).sum()
    )
    trading_cost = cost_rate * traded_notional

    # Short-sale proceeds enter the accounting cash ledger.
    new_cash = float(
        cash - ((new_shares - q) * p).sum() - trading_cost
    )

    calculated_nav = new_cash + float((new_shares * p).sum())

    if not np.isclose(
        calculated_nav, nav_after, atol=1e-7, rtol=1e-10
    ):
        raise RuntimeError("Self-financing accounting mismatch")

    diagnostics = {
        "nav_before": nav_before,
        "nav_after": nav_after,
        "traded_notional": traded_notional,
        "transaction_cost": trading_cost,
        "turnover": traded_notional / nav_before,
    }

    return (
        pd.Series(new_shares, index=ASSETS, dtype=float),
        new_cash,
        diagnostics,
    )

def inspect_short_collateral(
    shares: pd.Series,
    cash: float,
    prices: pd.Series,
    additional_margin_rate: float = 0.50,
) -> dict:
    """Inspect cash segregation and simplified short margin requirements."""

    if not isinstance(shares, pd.Series):
        raise TypeError("Shares must be a Series")

    if not isinstance(prices, pd.Series):
        raise TypeError("Prices must be a Series")

    if (
        list(shares.index) != list(ASSETS)
        or list(prices.index) != list(ASSETS)
    ):
        raise ValueError("Invalid asset universe")

    q = shares.to_numpy(dtype=float)
    p = prices.to_numpy(dtype=float)

    if (
        not np.isfinite(q).all()
        or not np.isfinite(p).all()
        or not np.isfinite(cash)
        or (p <= 0).any()
    ):
        raise ValueError("Invalid portfolio inputs")

    if (
        not np.isfinite(additional_margin_rate)
        or not 0 <= additional_margin_rate <= 1
    ):
        raise ValueError("Invalid margin rate")

    values = q * p

    long_notional = float(
        np.maximum(values, 0).sum()
    )

    short_notional = float(
        -np.minimum(values, 0).sum()
    )

    nav = float(
        cash + long_notional - short_notional
    )

    if nav <= 0:
        raise ValueError("Non-positive portfolio NAV")

    # Simplified collateral convention.
    short_collateral = short_notional

    additional_margin = (
        additional_margin_rate * short_notional
    )

    unrestricted_cash = (
        cash - short_collateral - additional_margin
    )

    return {
        "nav": nav,
        "long_notional": long_notional,
        "short_notional": short_notional,
        "cash_ledger": float(cash),
        "short_collateral": short_collateral,
        "additional_margin": additional_margin,
        "unrestricted_cash": unrestricted_cash,
        "margin_shortfall": max(0.0, -unrestricted_cash),
    }

def accrue_short_borrow_fee(
    shares: pd.Series,
    cash: float,
    reference_prices: pd.Series,
    elapsed_calendar_days: int,
    annual_borrow_rate: float = 0.01,
) -> tuple[float, dict]:
    """Charge simplified short borrowing costs using ACT/365."""

    if not isinstance(shares, pd.Series):
        raise TypeError("Shares must be a Series")

    if not isinstance(reference_prices, pd.Series):
        raise TypeError("Prices must be a Series")

    if (
        list(shares.index) != list(ASSETS)
        or list(reference_prices.index) != list(ASSETS)
    ):
        raise ValueError("Invalid asset universe")

    if (
        isinstance(elapsed_calendar_days, (bool, np.bool_))
        or not isinstance(elapsed_calendar_days, (int, np.integer))
        or elapsed_calendar_days < 1
    ):
        raise ValueError("Invalid elapsed days")

    rate = float(annual_borrow_rate)

    if not np.isfinite(rate) or rate < 0:
        raise ValueError("Invalid borrowing rate")

    q = shares.to_numpy(dtype=float)
    p = reference_prices.to_numpy(dtype=float)

    if (
        not np.isfinite(q).all()
        or not np.isfinite(p).all()
        or not np.isfinite(cash)
        or (p <= 0).any()
    ):
        raise ValueError("Invalid portfolio inputs")

    nav_before = float(cash + q @ p)

    if nav_before <= 0:
        raise ValueError("Non-positive NAV")

    # Negative quantities represent borrowed securities.
    short_notional = float(
        np.maximum(-(q * p), 0.0).sum()
    )

    borrow_fee = (
        short_notional
        * rate
        * elapsed_calendar_days
        / 365.0
    )

    new_cash = float(cash - borrow_fee)
    nav_after = float(new_cash + q @ p)

    if nav_after <= 0:
        raise ValueError("Borrowing fees exhausted portfolio equity")

    diagnostics = {
        "short_notional": short_notional,
        "borrow_fee": borrow_fee,
        "nav_before": nav_before,
        "nav_after": nav_after,
        "elapsed_days": elapsed_calendar_days,
    }

    return new_cash, diagnostics

def mark_to_market_close(
    shares: pd.Series,
    cash: float,
    close_prices: pd.Series,
    previous_close_nav: float,
    additional_margin_rate: float = 0.50,
) -> dict:
    """Value the portfolio at Close and calculate daily return."""

    previous_close_nav = float(previous_close_nav)

    if (
        not np.isfinite(previous_close_nav)
        or previous_close_nav <= 0
    ):
        raise ValueError("Invalid previous NAV")

    # Reuse our existing accounting and margin checks.
    report = inspect_short_collateral(
        shares=shares,
        cash=cash,
        prices=close_prices,
        additional_margin_rate=additional_margin_rate,
    )

    nav = report["nav"]

    report["daily_return"] = (
        nav / previous_close_nav - 1.0
    )

    report["long_exposure"] = (
        report["long_notional"] / nav
    )

    report["short_exposure"] = (
        report["short_notional"] / nav
    )

    report["gross_exposure"] = (
        report["long_exposure"]
        + report["short_exposure"]
    )

    report["net_exposure"] = (
        report["long_exposure"]
        - report["short_exposure"]
    )

    report["margin_ok"] = (
        report["margin_shortfall"] <= 1e-8
    )

    return report

def simulate_longshort_daily(
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission_bps: float = 2.0,
    slippage_bps: float = 1.0,
    annual_borrow_rate: float = 0.01,
    additional_margin_rate: float = 0.50,
) -> pd.DataFrame:
    """Simulate daily Long/Short portfolio accounting."""

    for name, panel in (
        ("Open", open_prices),
        ("Close", close_prices),
    ):
        if not isinstance(panel, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame")

        if list(panel.columns) != list(ASSETS):
            raise ValueError(f"Invalid {name} asset columns")

        index = panel.index

        if (
            not isinstance(index, pd.DatetimeIndex)
            or index.hasnans
            or index.has_duplicates
            or index.tz is not None
            or not index.is_monotonic_increasing
            or not index.normalize().equals(index)
        ):
            raise ValueError(f"Invalid {name} trading dates")

        values = panel.to_numpy(dtype=float)

        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Invalid {name} prices")

    if (
        len(open_prices) < 2
        or not open_prices.index.equals(close_prices.index)
    ):
        raise ValueError("Incompatible trading calendars")

    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Invalid initial capital")

    # Decisions are executed at the next available Open.
    scheduled = schedule_next_open_executions(
        target_weights,
        open_prices.index,
    )

    shares = pd.Series(0.0, index=ASSETS)
    cash = float(initial_capital)
    previous_nav = float(initial_capital)

    rows = []

    for i, date in enumerate(open_prices.index):

        borrow_fee = 0.0
        trading_cost = 0.0
        turnover = 0.0
        executed = False

        # Charge overnight borrowing on positions held
        # at the previous Close.
        if i > 0:
            previous_date = open_prices.index[i - 1]

            elapsed_days = (date - previous_date).days

            cash, borrowing = accrue_short_borrow_fee(
                shares=shares,
                cash=cash,
                reference_prices=close_prices.iloc[i - 1],
                elapsed_calendar_days=elapsed_days,
                annual_borrow_rate=annual_borrow_rate,
            )

            borrow_fee = borrowing["borrow_fee"]

        # Check margin BEFORE any scheduled rebalance.
        pretrade = inspect_short_collateral(
            shares=shares,
            cash=cash,
            prices=open_prices.loc[date],
            additional_margin_rate=additional_margin_rate,
        )

        if pretrade["margin_shortfall"] > 1e-8:
            raise RuntimeError(
                f"Margin shortfall before execution at "
                f"{date.date()}: "
                f"{pretrade['margin_shortfall']:.2f}"
            )

        # Execute previously generated signals at today's Open.
        if date in scheduled.index:
            shares, cash, execution = rebalance_at_open(
                shares=shares,
                cash=cash,
                open_prices=open_prices.loc[date],
                target_weights=scheduled.loc[date],
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
            )

            trading_cost = execution["transaction_cost"]
            turnover = execution["turnover"]
            executed = execution["traded_notional"] > 1e-10

        # Value the resulting portfolio at today's Close.
        report = mark_to_market_close(
            shares=shares,
            cash=cash,
            close_prices=close_prices.loc[date],
            previous_close_nav=previous_nav,
            additional_margin_rate=additional_margin_rate,
        )

        if not report["margin_ok"]:
            raise RuntimeError(
                f"Margin shortfall at {date.date()}: "
                f"{report['margin_shortfall']:.2f}"
            )

        row = {
            "Date": date,
            "nav": report["nav"],
            "daily_return": report["daily_return"],
            "cash": cash,
            "borrow_fee": borrow_fee,
            "transaction_cost": trading_cost,
            "turnover": turnover,
            "executed": executed,
            "long_exposure": report["long_exposure"],
            "short_exposure": report["short_exposure"],
            "gross_exposure": report["gross_exposure"],
            "net_exposure": report["net_exposure"],
            "margin_shortfall": report["margin_shortfall"],
        }

        for asset in ASSETS:
            row[f"{asset}_weight"] = (
                shares[asset]
                * close_prices.loc[date, asset]
                / report["nav"]
            )

        rows.append(row)
        previous_nav = report["nav"]

    return pd.DataFrame(rows).set_index("Date")


# A1/A2 input validation and legacy result schema.
# Core execution remains in simulate_longshort_daily above.

def _prepare_long_only_targets(
    aperture: pd.DataFrame,
    chiusure: pd.DataFrame,
    pesi_target: pd.DataFrame,
) -> pd.DataFrame:
    """Validate A1/A2 restrictions, preserving numerical clipping tolerance."""
    for name, panel in (
        ("aperture", aperture),
        ("chiusure", chiusure),
        ("pesi_target", pesi_target),
    ):
        if not isinstance(panel, pd.DataFrame) or panel.empty:
            raise ValueError(f"{name}: DataFrame mancante o vuoto")
        if not isinstance(panel.index, pd.DatetimeIndex):
            raise TypeError(f"{name}: richiesto DatetimeIndex")
        if (
            panel.index.hasnans
            or panel.index.has_duplicates
            or not panel.index.is_monotonic_increasing
            or panel.index.tz is not None
        ):
            raise ValueError(f"{name}: date non valide")
        if panel.columns.has_duplicates:
            raise ValueError(f"{name}: colonne duplicate")

    if not aperture.index.equals(chiusure.index):
        raise ValueError("Open e Close non hanno lo stesso calendario")
    if not aperture.columns.equals(chiusure.columns):
        raise ValueError("Open e Close non hanno lo stesso universo")
    if set(aperture.columns) != set(ASSETS) or set(pesi_target.columns) != set(ASSETS):
        raise ValueError("L'universo A1/A2 deve essere SPY, IEF, GLD, SHY")
    if not pesi_target.index.isin(aperture.index).all():
        raise ValueError("Le decisioni devono corrispondere a sedute effettive")

    for name, panel in (("Open", aperture), ("Close", chiusure)):
        try:
            values = panel.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: prezzi non numerici") from exc
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{name}: prezzi mancanti o non positivi")

    try:
        weights = pesi_target.loc[:, list(ASSETS)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Pesi target non numerici") from exc
    if (
        not np.isfinite(weights).all()
        or np.any(weights < -1e-7)
        or np.any(weights > 1.0 + 1e-7)
        or np.any(np.abs(weights.sum(axis=1) - 1.0) > 1e-7)
    ):
        raise ValueError("A1/A2 richiedono pesi long-only interamente investiti")

    # Same small numerical tolerance and normalization as the original A engine.
    weights = np.clip(weights, 0.0, 1.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return pd.DataFrame(weights, index=pesi_target.index, columns=list(ASSETS))


def simula_portafoglio(
    aperture: pd.DataFrame,
    chiusure: pd.DataFrame,
    pesi_target: pd.DataFrame,
    capitale_iniziale: float = 100000.0,
    commission_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> pd.DataFrame:
    """Run A1/A2 on the shared engine; preserve the original return schema."""
    for label, value, strictly_positive in (
        ("capitale_iniziale", capitale_iniziale, True),
        ("commission_bps", commission_bps, False),
        ("slippage_bps", slippage_bps, False),
    ):
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label}: parametro non numerico") from exc
        if not np.isfinite(numeric) or (numeric <= 0 if strictly_positive else numeric < 0):
            raise ValueError(f"{label}: parametro non valido")

    targets = _prepare_long_only_targets(aperture, chiusure, pesi_target)
    opens = aperture.loc[:, list(ASSETS)]
    closes = chiusure.loc[:, list(ASSETS)]

    # The shared scheduling function also rejects unexecutable-only decisions.
    scheduled = schedule_next_open_executions(targets, opens.index)
    common = simulate_longshort_daily(
        open_prices=opens,
        close_prices=closes,
        target_weights=targets,
        initial_capital=float(capitale_iniziale),
        commission_bps=float(commission_bps),
        slippage_bps=float(slippage_bps),
        annual_borrow_rate=0.0,
    )

    # Historical A1/A2 output schema, with the same approximate gross-return
    # diagnostic: daily net return plus direct trading cost / previous NAV.
    previous_nav = common["nav"].shift(1)
    previous_nav.iloc[0] = float(capitale_iniziale)

    result = pd.DataFrame(index=common.index)
    result["nav"] = common["nav"]
    result["rendimento_lordo"] = (
        common["daily_return"] + common["transaction_cost"] / previous_nav
    )
    result["rendimento_netto"] = common["daily_return"]
    result["turnover"] = common["turnover"]
    result["costo_transazione"] = common["transaction_cost"]
    # Preserve the pre-refactor A convention even for zero-notional orders.
    result["operazione_eseguita"] = result.index.isin(scheduled.index)
    for ticker in aperture.columns:
        result[f"{ticker}_peso_effettivo"] = common[f"{ticker}_weight"]

    return result

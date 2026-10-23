"""
Allocazione dinamica multi-asset con vincolo di volatilità.

Level A2:
- portafoglio di riferimento Equal Weight;
- covarianza storica su 126 sedute;
- budget di volatilità annualizzata dell'8%;
- ottimizzazione long-only, senza leva;
- risolutore CVXPY + CLARABEL.

Il modulo non legge file e non esegue backtest.
Riceve soltanto i rendimenti storici forniti dal chiamante.
"""

import cvxpy as cp
import numpy as np
import pandas as pd


ASSET = ("SPY", "IEF", "GLD", "SHY")


def calcola_pesi_a2(
    rendimenti: pd.DataFrame,
    finestra: int = 126,
    volatilita_massima: float = 0.08,
) -> tuple[pd.Series, dict]:
    """
    Calcola i pesi target alla data della decisione.

    I rendimenti devono contenere esclusivamente osservazioni
    già disponibili alla data della decisione.

    Restituisce:
    - pesi target per i quattro ETF;
    - diagnostiche dell'ottimizzazione.
    """
    if finestra < 2:
        raise ValueError("La finestra deve contenere almeno due sedute.")

    if not 0 < volatilita_massima:
        raise ValueError("Il limite di volatilità deve essere positivo.")

    if len(rendimenti) < finestra:
        raise ValueError("Osservazioni insufficienti.")

    if not set(ASSET).issubset(rendimenti.columns):
        raise ValueError("Mancano uno o più ETF richiesti.")

    if not isinstance(rendimenti.index, pd.DatetimeIndex):
        raise TypeError("È richiesto un indice temporale.")

    if not rendimenti.index.is_monotonic_increasing:
        raise ValueError("Le date non sono ordinate.")

    if rendimenti.index.has_duplicates:
        raise ValueError("Sono presenti date duplicate.")

    dati = rendimenti.loc[:, list(ASSET)].tail(finestra)

    if not np.isfinite(dati.to_numpy(dtype=float)).all():
        raise ValueError("Rendimenti mancanti o non validi.")

    cov = dati.cov().to_numpy() * 252
    cov = (cov + cov.T) / 2

    riferimento = np.full(len(ASSET), 1.0 / len(ASSET))

    def volatilita(pesi):
        varianza = float(pesi @ cov @ pesi)
        return float(np.sqrt(max(varianza, 0.0)))

    vol_ew = volatilita(riferimento)

    if vol_ew <= volatilita_massima:
        return (
            pd.Series(riferimento, index=ASSET),
            {
                "stato": "equal_weight",
                "volatilita_ew": vol_ew,
                "volatilita_target": vol_ew,
                "limite_rispettato": True,
            },
        )

    # Fattorizzazione PSD della matrice di covarianza.
    autovalori, autovettori = np.linalg.eigh(cov)

    if autovalori.min() < -1e-10:
        raise ValueError("Matrice di covarianza non semidefinita positiva.")

    autovalori = np.maximum(autovalori, 0.0)

    fattore = np.diag(np.sqrt(autovalori)) @ autovettori.T

    w = cp.Variable(len(ASSET))

    vincoli_base = [
        cp.sum(w) == 1,
        w >= 0,
    ]

    # Scala la volatilità in punti percentuali.
    rischio = cp.norm(100 * (fattore @ w), 2)

    problema = cp.Problem(
        cp.Minimize(cp.sum_squares(w - riferimento)),
        vincoli_base + [
            rischio <= 100 * volatilita_massima
        ],
    )

    problema.solve(solver=cp.CLARABEL)

    if problema.status in ("optimal", "optimal_inaccurate"):
        pesi = np.asarray(w.value, dtype=float).flatten()
        stato = "ottimizzato"
    elif problema.status in ("infeasible", "infeasible_inaccurate"):
        # Fallback: portafoglio long-only a varianza minima.
        minimo = cp.Problem(
            cp.Minimize(cp.sum_squares(100 * (fattore @ w))),
            vincoli_base,
        )

        minimo.solve(solver=cp.CLARABEL)

        if minimo.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"Minimum Variance non risolto: {minimo.status}"
            )

        pesi = np.asarray(w.value, dtype=float).flatten()
        stato = "fallback_minimum_variance"
    else:
        raise RuntimeError(
            f"Ottimizzazione A2 non risolta: {problema.status}"
        )

    vol_finale = volatilita(pesi)

    if (
        not np.isfinite(pesi).all()
        or abs(pesi.sum() - 1) > 1e-7
        or np.any(pesi < -1e-7)
        or np.any(pesi > 1 + 1e-7)
    ):
        raise RuntimeError("La soluzione viola i vincoli sui pesi.")

    limite_rispettato = (
        vol_finale <= volatilita_massima + 1e-7
    )

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

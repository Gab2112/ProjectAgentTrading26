"""Calcolo dei pesi del livello A2 della strategia Multi-Asset.

Il livello A2 modifica un portafoglio Equal Weight (25% per ETF) soltanto
quando la volatilità annualizzata stimata supera il limite dell'8%.

Metodo di ricerca, invariato rispetto alla versione originale:
- universo: SPY, IEF, GLD, SHY;
- matrice di covarianza: ultimi 126 rendimenti giornalieri, annualizzati a 252;
- funzione obiettivo: minima distanza quadratica dai pesi Equal Weight;
- vincoli: portafoglio interamente investito, long-only e senza leva;
- risolutore: CVXPY con CLARABEL;
- se il limite è irraggiungibile: portafoglio a varianza minima.

Il modulo non acquisisce dati, non legge il file YAML e non esegue backtest.
Il chiamante deve passare solo rendimenti disponibili entro la chiusura
della data di decisione; l'esecuzione avviene nella seduta successiva.
"""

import cvxpy as cp
import numpy as np
import pandas as pd


# Nomi e ordine sono conservati per compatibilità con portfolio.py.
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

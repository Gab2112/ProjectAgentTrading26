"""Motore giornaliero di simulazione per i portafogli Multi-Asset.

Il modulo applica i pesi target calcolati dalle strategie A1 e A2,
senza conoscere la logica con cui tali pesi sono stati determinati.

Convenzioni operative:
- ogni segnale e' disponibile soltanto dopo la chiusura della sua seduta;
- l'ordine viene eseguito all'apertura della seduta di mercato successiva;
- tra due operazioni le quantita' detenute rimangono costanti;
- commissioni e slippage riducono il capitale effettivamente investibile;
- Open e Close devono essere rettificati su una base coerente.

Il motore non legge o scrive file e non decide quali periodi siano accessibili.
La separazione tra training, validation e OOS spetta a ``backtester.py``.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def _verifica_input(
    aperture: pd.DataFrame,
    chiusure: pd.DataFrame,
    pesi_target: pd.DataFrame,
    capitale_iniziale: float,
    commission_bps: float,
    slippage_bps: float,
) -> tuple[list[str], np.ndarray, np.ndarray, float, float]:
    """Controlla calendario, universo, prezzi, segnali e parametri finanziari.

    I controlli precedono qualsiasi simulazione: dati incompleti o
    disallineati non devono produrre un backtest apparentemente valido.
    """
    for nome, dati in (
        ("aperture", aperture),
        ("chiusure", chiusure),
        ("pesi_target", pesi_target),
    ):
        if not isinstance(dati, pd.DataFrame):
            raise TypeError(f"{nome} deve essere un DataFrame pandas.")
        if dati.empty:
            raise ValueError(f"{nome} non puo' essere vuoto.")
        if not isinstance(dati.index, pd.DatetimeIndex):
            raise TypeError(f"{nome}: e' richiesto un indice temporale.")
        if (
            dati.index.hasnans
            or dati.index.has_duplicates
            or not dati.index.is_monotonic_increasing
            or dati.index.tz is not None
        ):
            raise ValueError(f"{nome}: date mancanti, duplicate o non ordinate.")
        if dati.columns.has_duplicates:
            raise ValueError(f"{nome}: colonne duplicate non ammesse.")

    if not aperture.index.equals(chiusure.index):
        raise ValueError("Le date di Open e Close non coincidono.")
    if not aperture.columns.equals(chiusure.columns):
        raise ValueError("Gli asset di Open e Close non coincidono.")
    if set(pesi_target.columns) != set(aperture.columns):
        raise ValueError("Gli asset dei segnali non coincidono con i prezzi.")
    if not pesi_target.index.isin(aperture.index).all():
        raise ValueError("Un segnale non corrisponde a una seduta di mercato.")

    try:
        capitale = float(capitale_iniziale)
        commissioni = float(commission_bps)
        slippage = float(slippage_bps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("I parametri finanziari devono essere numerici.") from exc

    if (
        not np.isfinite(capitale)
        or capitale <= 0
        or not np.isfinite(commissioni)
        or not np.isfinite(slippage)
        or commissioni < 0
        or slippage < 0
    ):
        raise ValueError("Capitale iniziale o costi di transazione non validi.")

    try:
        prezzi_open = aperture.to_numpy(dtype=float)
        prezzi_close = chiusure.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("I prezzi Open e Close devono essere numerici.") from exc

    if (
        not np.isfinite(prezzi_open).all()
        or not np.isfinite(prezzi_close).all()
        or np.any(prezzi_open <= 0)
        or np.any(prezzi_close <= 0)
    ):
        raise ValueError("Prezzi mancanti, non finiti o non positivi.")

    return (
        list(aperture.columns),
        prezzi_open,
        prezzi_close,
        capitale,
        (commissioni + slippage) / 10000.0,
    )


def _programma_esecuzioni(
    aperture: pd.DataFrame,
    pesi_target: pd.DataFrame,
    asset: list[str],
) -> dict[int, np.ndarray]:
    """Converte le date dei segnali in indici delle aperture successive.

    Una decisione sull'ultima seduta non puo' essere negoziata nel periodo
    fornito e viene scartata. Ogni vettore di pesi deve essere long-only e
    completamente investito, con tolleranza per errori numerici minimi.
    """
    esecuzioni: dict[int, np.ndarray] = {}

    for data, riga in pesi_target.loc[:, asset].iterrows():
        try:
            pesi = riga.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Pesi non numerici alla data {data}.") from exc

        if (
            not np.isfinite(pesi).all()
            or abs(float(pesi.sum()) - 1.0) > 1e-7
            or np.any(pesi < -1e-7)
            or np.any(pesi > 1.0 + 1e-7)
        ):
            raise ValueError(f"Pesi non validi alla data {data}.")

        posizione = aperture.index.get_loc(data)
        if posizione + 1 >= len(aperture):
            continue

        # Corregge soltanto arrotondamenti trascurabili del risolutore.
        pesi = np.clip(pesi, 0.0, 1.0)
        pesi /= pesi.sum()
        esecuzioni[posizione + 1] = pesi

    if not esecuzioni:
        raise ValueError(
            "Nessun segnale eseguibile: serve almeno una seduta "
            "successiva alla data di decisione."
        )

    return esecuzioni


def simula_portafoglio(
    aperture: pd.DataFrame,
    chiusure: pd.DataFrame,
    pesi_target: pd.DataFrame,
    capitale_iniziale: float = 100000.0,
    commission_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> pd.DataFrame:
    """Simula un portafoglio long-only senza leva su base giornaliera.

    Parametri:
        aperture: prezzi di apertura rettificati, un ETF per colonna;
        chiusure: prezzi di chiusura rettificati, stesso calendario e ordine;
        pesi_target: allocazioni deliberate DOPO la chiusura delle date indice;
        capitale_iniziale: capitale disponibile prima della prima operazione;
        commission_bps: costo di commissione per nozionale negoziato;
        slippage_bps: costo di slippage per nozionale negoziato.

    Restituisce un DataFrame con NAV, rendimenti, turnover, costi,
    indicatori di esecuzione e pesi effettivi a ogni chiusura.

    Nota: ``rendimento_lordo`` e' il rendimento netto piu' il costo diretto
    rapportato al NAV precedente. E' una diagnostica di costo approssimata,
    non un secondo backtest senza commissioni.
    """
    asset, op, cl, capitale, tasso_costo = _verifica_input(
        aperture,
        chiusure,
        pesi_target,
        capitale_iniziale,
        commission_bps,
        slippage_bps,
    )
    esecuzioni = _programma_esecuzioni(aperture, pesi_target, asset)

    # Le quantita' cambiano solo alle aperture programmate.
    quantita = np.zeros(len(asset))
    liquidita = float(capitale)
    nav_precedente = float(capitale)
    righe = []

    for i, data in enumerate(aperture.index):
        prezzi_open = op[i]
        prezzi_close = cl[i]
        valore_open = liquidita + quantita @ prezzi_open

        costo = 0.0
        turnover = 0.0
        eseguito = False

        if i in esecuzioni:
            target = esecuzioni[i]
            valori_pretrade = quantita * prezzi_open

            # Il costo dipende dal nozionale scambiato, che a sua volta
            # dipende dal capitale residuo dopo i costi: risolviamo
            # numericamente tale relazione, evitando una stima ex post.
            def equilibrio(capitale_post: float) -> float:
                negoziato = np.abs(
                    capitale_post * target - valori_pretrade
                ).sum()
                return (
                    capitale_post
                    + tasso_costo * negoziato
                    - valore_open
                )

            capitale_post = brentq(
                equilibrio,
                0.0,
                valore_open,
                xtol=1e-10,
            )

            valore_negoziato = np.abs(
                capitale_post * target - valori_pretrade
            ).sum()
            costo = tasso_costo * valore_negoziato
            turnover = valore_negoziato / valore_open

            quantita = capitale_post * target / prezzi_open
            liquidita = (
                valore_open
                - costo
                - quantita @ prezzi_open
            )
            eseguito = True

        # NAV ai prezzi di chiusura, incluse le variazioni intragiornaliere.
        nav = float(liquidita + quantita @ prezzi_close)
        if not np.isfinite(nav) or nav <= 0:
            raise RuntimeError("NAV non valido o non positivo.")

        rendimento_netto = nav / nav_precedente - 1
        costo_relativo = costo / nav_precedente
        rendimento_lordo = rendimento_netto + costo_relativo

        valori_close = quantita * prezzi_close
        riga = {
            "Date": data,
            "nav": nav,
            "rendimento_lordo": rendimento_lordo,
            "rendimento_netto": rendimento_netto,
            "turnover": turnover,
            "costo_transazione": costo,
            "operazione_eseguita": eseguito,
        }
        for j, ticker in enumerate(asset):
            riga[f"{ticker}_peso_effettivo"] = valori_close[j] / nav

        righe.append(riga)
        nav_precedente = nav

    return pd.DataFrame(righe).set_index("Date")

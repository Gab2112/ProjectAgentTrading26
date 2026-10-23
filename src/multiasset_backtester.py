"""
Backtester giornaliero per strategie multi-asset.

Le decisioni vengono prese dopo la chiusura di una seduta.
Gli ordini vengono eseguiti all'apertura della seduta successiva.

Il portafoglio è rappresentato attraverso quantità detenute
e liquidità, evitando ribilanciamenti giornalieri impliciti.

I prezzi Open e Close devono essere rettificati in modo coerente.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def simula_portafoglio(
    aperture: pd.DataFrame,
    chiusure: pd.DataFrame,
    pesi_target: pd.DataFrame,
    capitale_iniziale: float = 100000.0,
    commission_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> pd.DataFrame:
    """
    Simula un portafoglio long-only senza leva.

    Input:
      aperture: prezzi giornalieri di apertura;
      chiusure: prezzi giornalieri di chiusura;
      pesi_target: pesi indicizzati per data effettiva del segnale,
                   disponibile soltanto dopo la chiusura.

    Output:
      DataFrame giornaliero con NAV, rendimenti, turnover,
      costi di transazione e pesi effettivi a fine seduta.

    Le decisioni prive di una seduta successiva vengono escluse.
    """

    if not aperture.index.equals(chiusure.index):
        raise ValueError("Le date di Open e Close non coincidono.")

    if not aperture.columns.equals(chiusure.columns):
        raise ValueError("Gli asset di Open e Close non coincidono.")

    if not isinstance(aperture.index, pd.DatetimeIndex):
        raise TypeError("È richiesto un indice temporale.")

    if (
        not aperture.index.is_monotonic_increasing
        or aperture.index.has_duplicates
    ):
        raise ValueError("Date giornaliere non valide.")

    if (
        not isinstance(pesi_target.index, pd.DatetimeIndex)
        or not pesi_target.index.is_monotonic_increasing
        or pesi_target.index.has_duplicates
    ):
        raise ValueError("Date dei segnali non valide.")

    if set(pesi_target.columns) != set(aperture.columns):
        raise ValueError("Gli asset dei segnali non coincidono.")

    if not pesi_target.index.isin(aperture.index).all():
        raise ValueError("Un segnale non corrisponde a una seduta.")

    if (
        not np.isfinite(capitale_iniziale)
        or capitale_iniziale <= 0
        or not np.isfinite(commission_bps)
        or not np.isfinite(slippage_bps)
        or min(commission_bps, slippage_bps) < 0
    ):
        raise ValueError("Parametri finanziari non validi.")

    asset = list(aperture.columns)

    op = aperture.to_numpy(dtype=float)
    cl = chiusure.to_numpy(dtype=float)

    if (
        not np.isfinite(op).all()
        or not np.isfinite(cl).all()
        or np.any(op <= 0)
        or np.any(cl <= 0)
    ):
        raise ValueError("Prezzi mancanti o non positivi.")

    # Associazione dei segnali alla prima apertura successiva.
    esecuzioni = {}

    for data, riga in pesi_target[asset].iterrows():
        posizione = aperture.index.get_loc(data)

        if posizione + 1 >= len(aperture):
            continue

        w = riga.to_numpy(dtype=float)

        if (
            not np.isfinite(w).all()
            or abs(w.sum() - 1) > 1e-7
            or np.any(w < -1e-7)
            or np.any(w > 1 + 1e-7)
        ):
            raise ValueError(f"Pesi non validi alla data {data}.")

        # Elimina esclusivamente residui numerici trascurabili.
        w = np.clip(w, 0.0, 1.0)
        w = w / w.sum()

        esecuzioni[posizione + 1] = w

    tasso_costo = (commission_bps + slippage_bps) / 10000

    quantita = np.zeros(len(asset))
    liquidita = float(capitale_iniziale)
    nav_precedente = float(capitale_iniziale)

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

            # Il capitale investibile diminuisce per effetto
            # dei costi sostenuti al momento dell'operazione.
            def equilibrio(capitale_post):
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

            quantita = (
                capitale_post * target / prezzi_open
            )

            liquidita = (
                valore_open
                - costo
                - quantita @ prezzi_open
            )

            eseguito = True

        nav = float(
            liquidita + quantita @ prezzi_close
        )

        rendimento_netto = nav / nav_precedente - 1
        costo_relativo = costo / nav_precedente
        rendimento_lordo = (
            rendimento_netto + costo_relativo
        )

        valori_close = quantita * prezzi_close

        if nav <= 0:
            raise RuntimeError("NAV non positivo.")

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
            riga[f"{ticker}_peso_effettivo"] = (
                valori_close[j] / nav
            )

        righe.append(riga)
        nav_precedente = nav

    return pd.DataFrame(righe).set_index("Date")

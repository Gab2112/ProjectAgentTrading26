"""Funzioni di supporto condivise della pipeline Multi-Asset.

Questo modulo centralizza la lettura dell'universo investibile e la gestione
provvisoria dei ritardi di pubblicazione delle serie macroeconomiche.

Non legge file e non modifica dati o configurazioni. Le funzioni pubbliche
mantengono le stesse firme usate dagli altri moduli della pipeline.
"""


# Ritardi indicativi, espressi in giorni di calendario, per alcune serie FRED.
# NON rappresentano un calendario storico delle pubblicazioni né le versioni
# dei dati disponibili in ciascuna data (vintage). Prima del livello B occorre
# verificare le date di rilascio effettive e gli eventuali aggiornamenti.
DEFAULT_RELEASE_LAGS_DAYS = {
    "CPIAUCSL": 45,
    "CPILFESL": 45,
    "FEDFUNDS": 35,
}


def get_tickers(config: dict) -> list[str]:
    """Restituisce i ticker investibili nell'ordine definito in ``config.yaml``.

    Nel nuovo progetto l'universo comprende SPY, IEF, GLD e SHY. La funzione
    non impone tuttavia questi quattro strumenti: la verifica dell'universo
    specifico dei livelli A1/A2 compete ai moduli di strategia.

    Controlla che le voci della configurazione siano valide e che ogni ticker
    sia presente una sola volta, evitando ambiguità tra dati, pesi e ordini.
    """
    if not isinstance(config, dict):
        raise TypeError("La configurazione deve essere un dizionario.")

    universe = config.get("universe")
    if not isinstance(universe, dict):
        raise ValueError("Manca la sezione 'universe' nella configurazione.")

    assets = universe.get("traded_assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("La sezione 'universe.traded_assets' deve contenere asset.")

    tickers: list[str] = []
    for category, asset in assets.items():
        if not isinstance(asset, dict):
            raise ValueError(f"Configurazione non valida per l'asset '{category}'.")

        ticker = asset.get("ticker")
        if not isinstance(ticker, str) or not ticker or ticker != ticker.strip():
            raise ValueError(f"Ticker mancante o non valido per '{category}'.")
        if ticker != ticker.upper():
            raise ValueError(f"Il ticker '{ticker}' deve essere scritto in maiuscolo.")
        if ticker in tickers:
            raise ValueError(f"Ticker duplicato nell'universo investibile: {ticker}.")

        tickers.append(ticker)

    return tickers


def get_release_lag_days(config: dict, series_id: str) -> int:
    """Restituisce il ritardo convenzionale di una serie FRED, in giorni.

    Cerca anzitutto ``data.release_lags_days`` in ``config.yaml``. In assenza
    di un valore specifico usa ``DEFAULT_RELEASE_LAGS_DAYS``; per le altre
    serie conserva il comportamento precedente, restituendo zero.

    Il valore zero NON certifica che una serie fosse disponibile in tempo
    reale. Anche i ritardi fissi di 45 o 35 giorni sono approssimazioni:
    il livello B dovrà utilizzare un protocollo point-in-time verificato.

    Rifiuta ritardi negativi, frazionari o booleani, che altrimenti
    potrebbero generare errori temporali difficili da individuare.
    """
    if not isinstance(config, dict):
        raise TypeError("La configurazione deve essere un dizionario.")
    if not isinstance(series_id, str) or not series_id.strip():
        raise ValueError("L'identificativo della serie FRED non è valido.")

    data = config.get("data", {})
    if not isinstance(data, dict):
        raise TypeError("La sezione 'data' deve essere un dizionario.")

    configured_lags = data.get("release_lags_days")
    if configured_lags is None:
        configured_lags = {}
    if not isinstance(configured_lags, dict):
        raise TypeError("'data.release_lags_days' deve essere un dizionario.")

    lag = configured_lags.get(
        series_id,
        DEFAULT_RELEASE_LAGS_DAYS.get(series_id, 0),
    )
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
        raise ValueError(
            f"Ritardo di pubblicazione non valido per '{series_id}': {lag!r}. "
            "È richiesto un numero intero di giorni maggiore o uguale a zero."
        )

    return lag

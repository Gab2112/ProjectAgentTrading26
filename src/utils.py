"""
utils.py

Funzioni di supporto condivise da più moduli della pipeline.

Raccoglie in un unico punto la logica che prima era duplicata nei singoli
file, così che una modifica (es. un nuovo ETF o un diverso lag di
pubblicazione) vada fatta una sola volta.
"""


def get_tickers(config: dict) -> list[str]:
    """
    Restituisce i ticker dell'universo investibile, nell'ordine del config.

    Esempio con il config attuale:
        ["SHY", "IEF", "TLT"]
    """
    return [
        asset["ticker"]
        for asset in config["universe"]["traded_assets"].values()
    ]


# Fallback provvisori e volutamente prudenti, usati se config.yaml non
# contiene la sezione data.release_lags_days.
DEFAULT_RELEASE_LAGS_DAYS = {
    "CPIAUCSL": 45,
    "CPILFESL": 45,
    "FEDFUNDS": 35,
}


def get_release_lag_days(config: dict, series_id: str) -> int:
    """
    Restituisce il lag prudenziale di pubblicazione, in giorni di calendario.

    Spiegazione intuitiva: dobbiamo evitare il look-ahead bias. Il CPI di
    gennaio descrive l'inflazione di gennaio, ma viene pubblicato solo a metà
    febbraio. Se nel backtest associassimo quel valore alla data di gennaio,
    il modello userebbe un'informazione che in quel momento storico nessun
    trader poteva avere. Il lag sposta quindi in avanti la data del dato fino
    a quando era realmente disponibile.

    I valori 45 e 35 giorni sono fallback prudenti; in una fase successiva
    potranno essere sostituiti dalle date di rilascio effettive.
    """
    configured_lags = (
        config.get("data", {})
        .get("release_lags_days", {})
    )

    if series_id in configured_lags:
        return int(configured_lags[series_id])

    return DEFAULT_RELEASE_LAGS_DAYS.get(series_id, 0)

"""
data_acquisition.py - Modulo per il download e la gestione dei dati grezzi di mercato.
"""

from typing import Union, List
import pandas as pd
import yfinance as yf


def get_data(
    tickers: Union[str, List[str]],
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Scarica i dati storici di mercato tramite l'API di Yahoo Finance.

    Parameters:
        tickers (str | list[str]): Ticker singolo (es. 'AAPL') o lista di ticker (es. ['SPY', 'QQQ']).
        start_date (str): Data di inizio in formato 'YYYY-MM-DD'.
        end_date (str): Data di fine in formato 'YYYY-MM-DD'.
        interval (str): Frequenza dei dati ('1d', '1h', '5m', ecc.). Default: '1d'.

    Returns:
        pd.DataFrame: DataFrame con i dati OHLCV indicizzati per data/ora.
    """
    if isinstance(tickers, list):
        tickers_str = " ".join(tickers)
    else:
        tickers_str = tickers

    print(f"Scaricamento dati per: {tickers_str} da {start_date} a {end_date}...")

    # auto_adjust=True corregge automaticamente Open, High, Low, Close per split e dividendi
    df = yf.download(
        tickers=tickers_str,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"Nessun dato trovato per i ticker '{tickers_str}' nel range specificato.")

    return df
"""Lettura centralizzata e controllata dei dataset del progetto Multi-Asset.

Il modulo gestisce esclusivamente il caricamento dei file e la validazione
strutturale delle date. Non modifica prezzi, rendimenti o informazioni macro.

Ogni formato utilizza un lettore dedicato, derivato da ``AbstractOpener``.
La funzione pubblica ``load_dataset()`` mantiene un'unica interfaccia per
CSV, TSV, TXT, Excel, Parquet, Feather e JSON.

Modalità rigorosa (``strict=True``): date mancanti, non valide o duplicate
producono un errore, senza correzioni implicite. In modalità permissiva
(``strict=False``), le righe con date non valide vengono scartate e, per
date duplicate, viene mantenuta l'ultima occorrenza.

Questo modulo non applica i confini temporali train/validation/OOS:
la responsabilità rimane nel modulo ``time_split.py``.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


# =============================================================================
# Interfaccia comune per i lettori
# =============================================================================

class AbstractOpener(ABC):
    """Definisce i controlli condivisi dai lettori dei diversi formati.

    Le sottoclassi implementano soltanto ``_load_data()``; il metodo
    ``open()`` applica le verifiche comuni e prepara l'indice temporale.
    """

    def open(
        self,
        path: str | Path,
        date_column: str | None = "Date",
        strict: bool = True,
        **read_options,
    ) -> pd.DataFrame:
        """Carica un dataset, preservandone le colonne e l'ordine dei valori.

        Parametri:
            path: percorso del file da leggere.
            date_column: nome della colonna data; con ``None`` non viene
                modificato l'indice del DataFrame.
            strict: se vero, rifiuta date non valide o duplicate; se falso,
                esegue le sole correzioni delle date descritte sopra.
            read_options: opzioni inoltrate al lettore pandas del formato.

        Restituisce:
            DataFrame ordinato per data, quando ``date_column`` è indicata.

        Non riempie valori mancanti, non applica trasformazioni finanziarie
        e non utilizza informazioni di periodi successivi.
        """
        path = Path(path)

        if not isinstance(strict, bool):
            raise TypeError("Il parametro 'strict' deve essere booleano.")

        if date_column is not None and (
            not isinstance(date_column, str) or not date_column
        ):
            raise TypeError("'date_column' deve essere una stringa non vuota o None.")

        if not path.is_file():
            raise FileNotFoundError(f"File non trovato: {path}")

        try:
            df = self._load_data(path, **read_options)
        except Exception as exc:
            raise RuntimeError(
                f"Errore durante la lettura del file {path}: {exc}"
            ) from exc

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Il file {path.name} non contiene un DataFrame.")

        if df.empty:
            raise ValueError(f"Il file {path.name} non contiene osservazioni.")

        if df.columns.has_duplicates:
            raise ValueError(f"Colonne duplicate nel file {path.name}.")

        if date_column is None:
            # Utile per tabelle prive di un indice temporale esplicito.
            return df

        return self._set_date_index(df, path, date_column, strict)

    @abstractmethod
    def _load_data(self, path: Path, **read_options) -> pd.DataFrame:
        """Legge il contenuto del file nel formato specifico della sottoclasse."""
        ...

    @staticmethod
    def _set_date_index(
        df: pd.DataFrame,
        path: Path,
        date_column: str,
        strict: bool,
    ) -> pd.DataFrame:
        """Converte e verifica la colonna data, quindi ordina l'indice.

        Se Parquet o un altro formato conservano la data come indice, questa
        viene riportata a colonna prima di applicare gli stessi controlli
        utilizzati per i file CSV. Non si attribuiscono date mancanti a
        osservazioni che ne sono prive.
        """
        if date_column not in df.columns and df.index.name == date_column:
            df = df.reset_index()

        if date_column not in df.columns:
            raise KeyError(
                f"Colonna '{date_column}' assente nel file {path.name}."
            )

        df = df.copy()
        df[date_column] = pd.to_datetime(
            df[date_column],
            errors="coerce",
        )

        invalid_dates = df[date_column].isna()
        if invalid_dates.any():
            if strict:
                raise ValueError(
                    f"Rilevate {int(invalid_dates.sum())} date non valide "
                    f"nel file {path.name}."
                )
            df = df.loc[~invalid_dates].copy()

        if df.empty:
            raise ValueError(
                f"Nessuna osservazione con data valida nel file {path.name}."
            )

        duplicate_dates = df.duplicated(subset=[date_column], keep=False)
        if duplicate_dates.any():
            if strict:
                raise ValueError(
                    f"Rilevate {int(duplicate_dates.sum())} righe con date "
                    f"duplicate nel file {path.name}."
                )
            # L'ordine del file determina quale osservazione conservare.
            df = df.drop_duplicates(subset=[date_column], keep="last")

        result = df.set_index(date_column).sort_index()
        if not isinstance(result.index, pd.DatetimeIndex):
            raise TypeError(
                f"Impossibile costruire un indice temporale per {path.name}."
            )
        return result


# =============================================================================
# Lettori specifici per formato
# =============================================================================

class CSVOpener(AbstractOpener):
    """Legge file delimitati CSV, TSV e TXT."""

    def _load_data(self, path: Path, **read_options) -> pd.DataFrame:
        if path.suffix.lower() == ".tsv":
            read_options.setdefault("sep", "\t")
        return pd.read_csv(path, **read_options)


class ExcelOpener(AbstractOpener):
    """Legge file Excel (openpyxl per .xlsx, xlrd per .xls)."""

    def _load_data(self, path: Path, **read_options) -> pd.DataFrame:
        return pd.read_excel(path, **read_options)


class ParquetOpener(AbstractOpener):
    """Legge file Parquet; richiede un motore come pyarrow."""

    def _load_data(self, path: Path, **read_options) -> pd.DataFrame:
        return pd.read_parquet(path, **read_options)


class FeatherOpener(AbstractOpener):
    """Legge file Feather; richiede un motore come pyarrow."""

    def _load_data(self, path: Path, **read_options) -> pd.DataFrame:
        return pd.read_feather(path, **read_options)


class JSONOpener(AbstractOpener):
    """Legge dataset JSON compatibili con pandas."""

    def _load_data(self, path: Path, **read_options) -> pd.DataFrame:
        return pd.read_json(path, **read_options)


# =============================================================================
# Registro dei formati e selezione automatica del lettore
# =============================================================================

SUPPORTED_FORMATS: dict[str, type[AbstractOpener]] = {
    ".csv": CSVOpener,
    ".tsv": CSVOpener,
    ".txt": CSVOpener,
    ".xlsx": ExcelOpener,
    ".xls": ExcelOpener,
    ".parquet": ParquetOpener,
    ".feather": FeatherOpener,
    ".json": JSONOpener,
}


def get_opener(path: str | Path) -> AbstractOpener:
    """Seleziona il lettore specializzato in base all'estensione del file."""
    extension = Path(path).suffix.lower()

    if extension not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"Tipo di file non supportato: '{extension}'. "
            f"Formati supportati: {supported}"
        )

    return SUPPORTED_FORMATS[extension]()


# =============================================================================
# Funzione pubblica condivisa tra i moduli della pipeline
# =============================================================================

def load_dataset(
    path: str | Path,
    date_column: str | None = "Date",
    strict: bool = True,
    **read_options,
) -> pd.DataFrame:
    """Punto d'accesso unico ai dataset della ricerca Multi-Asset.

    Esempi:
        ``load_dataset('data/raw/market/SPY.csv')``
        ``load_dataset('data/processed/market_daily.csv')``
        ``load_dataset('data/raw/bloomberg.xlsx', sheet_name='Prices')``

    Il controllo dei prezzi e delle date di negoziazione è gestito da
    ``data_transformation.py``; la separazione temporale è gestita da
    ``time_split.py``.
    """
    opener = get_opener(path)
    return opener.open(
        path,
        date_column=date_column,
        strict=strict,
        **read_options,
    )

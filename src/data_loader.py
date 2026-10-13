"""
data_loader.py

Lettura centralizzata dei dataset del progetto.

Ogni formato ha il proprio lettore (Opener). Tutti i lettori condividono
la stessa logica comune, definita in AbstractOpener:

1. controllo dell'esistenza del file;
2. lettura del file con il metodo specifico del formato;
3. conversione della colonna data in indice temporale ordinato.

Per aggiungere un nuovo formato è sufficiente creare una nuova classe
derivata da AbstractOpener e registrarla in SUPPORTED_FORMATS.

Nel resto del progetto si utilizza esclusivamente la funzione load_dataset().
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


# =============================================================================
# Interfaccia comune
# =============================================================================

class AbstractOpener(ABC):
    """
    Classe astratta che funge da interfaccia per i lettori di file.

    Il metodo open() definisce la procedura comune a tutti i formati;
    le classi derivate implementano soltanto _load_data().
    """

    def open(
        self,
        path: str | Path,
        date_column: str | None = "Date",
        drop_duplicate_dates: bool = True,
        **read_options,
    ) -> pd.DataFrame:
        """
        Carica il file e restituisce un DataFrame indicizzato per data.

        Input:
          - path: percorso del file da caricare;
          - date_column: nome della colonna data. Se None, il DataFrame
            viene restituito senza trasformare l'indice;
          - drop_duplicate_dates: se True, in caso di date duplicate
            mantiene l'ultima osservazione;
          - read_options: opzioni aggiuntive passate alla funzione pandas
            (es. sheet_name per Excel, sep per CSV).

        Output:
          - pd.DataFrame con indice temporale ordinato.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File {path} non trovato")

        try:
            df = self._load_data(path, **read_options)
        except Exception as e:
            raise RuntimeError(
                f"Errore durante la lettura del file {path}: {e}"
            ) from e

        if date_column is None:
            return df

        return self._set_date_index(
            df,
            path,
            date_column,
            drop_duplicate_dates,
        )

    @abstractmethod
    def _load_data(self, path: Path, **read_options) -> pd.DataFrame:
        """Metodo implementato dalle classi derivate specifiche."""
        ...

    @staticmethod
    def _set_date_index(
        df: pd.DataFrame,
        path: Path,
        date_column: str,
        drop_duplicate_dates: bool,
    ) -> pd.DataFrame:
        """
        Converte la colonna data in indice temporale ordinato.

        Alcuni formati (Parquet, Feather) possono salvare la data già
        come indice: in quel caso viene riportata a colonna prima della
        conversione, così che tutti i formati seguano la stessa procedura.
        """
        if date_column not in df.columns and df.index.name == date_column:
            df = df.reset_index()

        if date_column not in df.columns:
            raise KeyError(
                f"Colonna '{date_column}' assente nel file {path.name}"
            )

        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        # Righe con data non valida vengono scartate.
        df = df.dropna(subset=[date_column])

        if drop_duplicate_dates:
            df = df.drop_duplicates(subset=[date_column], keep="last")

        return df.set_index(date_column).sort_index()


# =============================================================================
# Lettori specifici
# =============================================================================

class CSVOpener(AbstractOpener):
    """Lettore per file di testo delimitati (.csv, .tsv, .txt)."""

    def _load_data(self, path: Path, **read_options) -> pd.DataFrame:
        if path.suffix.lower() == ".tsv":
            read_options.setdefault("sep", "\t")
        return pd.read_csv(path, **read_options)


class ExcelOpener(AbstractOpener):
    """Lettore per fogli di calcolo Excel (.xlsx, .xls). Richiede openpyxl."""

    def _load_data(self, path: Path, **read_options) -> pd.DataFrame:
        return pd.read_excel(path, **read_options)


class ParquetOpener(AbstractOpener):
    """Lettore per file Parquet (.parquet). Richiede pyarrow."""

    def _load_data(self, path: Path, **read_options) -> pd.DataFrame:
        return pd.read_parquet(path, **read_options)


class FeatherOpener(AbstractOpener):
    """Lettore per file Feather (.feather). Richiede pyarrow."""

    def _load_data(self, path: Path, **read_options) -> pd.DataFrame:
        return pd.read_feather(path, **read_options)


class JSONOpener(AbstractOpener):
    """Lettore per file JSON (.json)."""

    def _load_data(self, path: Path, **read_options) -> pd.DataFrame:
        return pd.read_json(path, **read_options)


# =============================================================================
# Registro dei formati e factory
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
    """
    Factory function: seleziona il lettore corretto in base all'estensione.

    Input:
      - path: percorso o nome del file.

    Output:
      - istanza del lettore specializzato.
    """
    extension = Path(path).suffix.lower()

    if extension not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"Tipo di file non supportato: '{extension}'. "
            f"Formati supportati: {supported}"
        )

    return SUPPORTED_FORMATS[extension]()


# =============================================================================
# Funzione pubblica
# =============================================================================

def load_dataset(
    path: str | Path,
    date_column: str | None = "Date",
    drop_duplicate_dates: bool = True,
    **read_options,
) -> pd.DataFrame:
    """
    Unico punto di accesso ai dati per tutto il progetto.

    Esempi:
        load_dataset("data/processed/features_weekly.csv")
        load_dataset("data/raw/bloomberg.xlsx", sheet_name="Prices")
        load_dataset("data/processed/features.parquet")
    """
    opener = get_opener(path)

    return opener.open(
        path,
        date_column=date_column,
        drop_duplicate_dates=drop_duplicate_dates,
        **read_options,
    )

"""
FinBERT_agent.py

Agente di Sentiment Analysis basato su Transformer pre-addestrato FinBERT (ProsusAI/finbert)
e modulo di acquisizione / download di database di notizie finanziarie e macroeconomiche.

Funzionalità principali:
1. download_news_dataset(): funzione per acquisire e strutturare un database di notizie/headline finanziarie da:
   - Federal Reserve Monetary Policy Press Releases (FOMC decisions, discount rate, rate policy)
   - Federal Reserve Board All Releases
   - MarketWatch / Dow Jones Financial News
   - HuggingFace Financial News Sentiment dataset (zeroshot/twitter-financial-news-sentiment)
   - Calendario storico degli eventi macro/tassi USA (2004-2026) per allineamento backtesting In-Sample
2. FinBERTAgent:
   - Estrazione di probabilità softmax (positivo, negativo, neutrale) con tokenizzazione a batch
   - Supporto per 'invert_sentiment' (essenziale per ETF obbligazionari come SHY, IEF, TLT)
   - Metodo predict_with_confidence(): restituisce congiuntamente (signal, confidence)
   - Metodo generate_sentiment_signals(): aggrega le notizie su base settimanale e produce
     un DataFrame con esattamente due colonne: 'signal' ({-1, 0, 1}) e 'confidence' ([0.0, 1.0])
     pronto per essere utilizzato come input dell'Agente Orchestratore.
"""

import html
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Risoluzione robusta dei percorsi per esecuzione diretta o come modulo
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from src.agents.base_agent import BaseAgent
except ImportError:
    from .base_agent import BaseAgent


# =====================================================================
# 1. FUNZIONE DI ACQUISIZIONE E DOWNLOAD DEL DATABASE NOTIZIE
# =====================================================================
def clean_headline_text(text: str) -> str:
    """Rimuove entità HTML, tag XML, link URL e spazi multipli dal testo."""
    if not isinstance(text, str):
        return ""
    # Decodifica entità HTML (&amp;, &quot;, &#39;, ecc.)
    text = html.unescape(text)
    # Rimuove tag HTML/XML
    text = re.sub(r"<[^>]+>", " ", text)
    # Rimuove URL http/https
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Rimuove caratteri speciali spuri ma conserva punteggiatura finanziaria (%, $, numeri)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_rss_headlines(feed_url: str, source_name: str, category: str, timeout: int = 8) -> List[Dict[str, str]]:
    """Scarica e parsa le notizie da un feed RSS XML."""
    headlines = []
    try:
        req = urllib.request.Request(
            feed_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")
            for item in items:
                title_elem = item.find("title")
                date_elem = item.find("pubDate")
                desc_elem = item.find("description")

                title = title_elem.text if title_elem is not None and title_elem.text else ""
                clean_title = clean_headline_text(title)
                if not clean_title or len(clean_title) < 8:
                    continue

                pub_date_str = date_elem.text if date_elem is not None and date_elem.text else ""
                try:
                    dt = pd.to_datetime(pub_date_str).strftime("%Y-%m-%d")
                except Exception:
                    dt = pd.Timestamp.now().strftime("%Y-%m-%d")

                desc = desc_elem.text if desc_elem is not None and desc_elem.text else ""
                clean_desc = clean_headline_text(desc)

                headlines.append({
                    "Date": dt,
                    "headline": clean_title,
                    "description": clean_desc,
                    "source": source_name,
                    "category": category,
                })
    except Exception as e:
        print(f"  [WARN] Impossibile scaricare feed RSS {source_name} ({feed_url}): {e}")

    return headlines


def fetch_huggingface_financial_news(limit: int = 150, timeout: int = 8) -> List[Dict[str, str]]:
    """Scarica headline finanziarie da HuggingFace Datasets Server."""
    headlines = []
    url = f"https://datasets-server.huggingface.co/rows?dataset=zeroshot/twitter-financial-news-sentiment&config=default&split=train&offset=0&limit={limit}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            rows = data.get("rows", [])
            for r in rows:
                row_data = r.get("row", {})
                text = row_data.get("text", "")
                clean_text = clean_headline_text(text)
                if clean_text and len(clean_text) >= 10:
                    headlines.append({
                        "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                        "headline": clean_text,
                        "description": "",
                        "source": "huggingface_financial_news",
                        "category": "market_sentiment",
                    })
    except Exception as e:
        print(f"  [WARN] Download HuggingFace news non riuscito: {e}")

    return headlines


def build_historical_macro_calendar_news() -> List[Dict[str, str]]:
    """
    Costruisce un catalogo storico curato di notizie, annunci FOMC e decisioni sui tassi USA
    allineato al periodo di backtesting del progetto (2004-2026).
    Garantisce copertura In-Sample (2004-2021) e Out-Of-Sample (2022+).
    """
    events = [
        # 2004 - Ciclo di rialzo tassi Greenspan
        ("2004-06-30", "Federal Reserve raises federal funds rate by 25 basis points to 1.25 percent, beginning tightening cycle.", "monetary_policy"),
        ("2004-09-21", "Fed increases rates by 25 bps, notes inflation and economic output expanding at moderate pace.", "monetary_policy"),
        ("2004-12-14", "FOMC raises interest rates to 2.25 percent; pace of economic expansion looks robust.", "monetary_policy"),
        # 2005 - Rialzi tassi continui
        ("2005-05-03", "Federal Reserve raises rates to 3.0 percent, indicates measured removal of policy accommodation.", "monetary_policy"),
        ("2005-08-09", "Fed lifts target funds rate to 3.5 percent amid rising energy costs and economic growth.", "monetary_policy"),
        ("2005-11-01", "FOMC increases rate to 4.0 percent; upside inflation risks cited.", "monetary_policy"),
        # 2006 - Picco tassi a 5.25%
        ("2006-03-28", "Fed hikes rates to 4.75 percent under new Chairman Ben Bernanke.", "monetary_policy"),
        ("2006-06-29", "Federal Reserve raises federal funds rate to 5.25 percent, pausing further rate hikes.", "monetary_policy"),
        # 2007 - Inizio crisi mutui subprime e tagli tassi
        ("2007-08-17", "Federal Reserve cuts discount rate by 50 basis points to alleviate liquidity crunch in money markets.", "monetary_policy"),
        ("2007-09-18", "Fed slashes federal funds rate by 50 basis points to 4.75 percent to forestall credit market contagion.", "monetary_policy"),
        ("2007-12-11", "FOMC lowers rate to 4.25 percent as housing recession deepens.", "monetary_policy"),
        # 2008 - Grande Crisi Finanziaria e ZIRP
        ("2008-01-22", "Fed announces emergency 75 bps inter-meeting rate cut following global equity market plunge.", "monetary_policy"),
        ("2008-03-16", "JPMorgan acquires Bear Stearns backed by Fed financing facility; discount rate lowered.", "market_news"),
        ("2008-09-15", "Lehman Brothers files for Chapter 11 bankruptcy protection; systemic credit panic erupts.", "market_news"),
        ("2008-10-08", "Coordinated global central bank rate cut; Federal Reserve cuts target rate to 1.5 percent.", "monetary_policy"),
        ("2008-12-16", "Fed cuts interest rates to zero to 0.25 percent target range and initiates first Quantitative Easing (QE1).", "monetary_policy"),
        # 2009-2012 - Ripresa lenta, QE2, Operation Twist
        ("2009-03-18", "Federal Reserve expands balance sheet by purchasing 300 billion in long-term Treasury securities.", "monetary_policy"),
        ("2010-11-03", "Fed launches QE2 with 600 billion purchase program for intermediate Treasuries.", "monetary_policy"),
        ("2011-08-05", "Standard & Poor's downgrades United States sovereign credit rating from AAA to AA+.", "market_news"),
        ("2011-09-21", "Fed announces Operation Twist: sells short-term paper to purchase 400 billion in long-term Treasuries.", "monetary_policy"),
        ("2012-09-13", "Federal Reserve launches open-ended QE3 purchasing 40 billion monthly in agency mortgage-backed securities.", "monetary_policy"),
        # 2013 - Taper Tantrum
        ("2013-05-22", "Bernanke suggests Fed may taper asset purchases in coming meetings; Treasury yields surge.", "monetary_policy"),
        ("2013-12-18", "FOMC announces modest reduction in monthly pace of bond purchases from 85 billion to 75 billion.", "monetary_policy"),
        # 2015-2018 - Normalizzazione e rialzi tassi Yellen/Powell
        ("2015-12-16", "Federal Reserve raises rates off zero bound by 25 basis points for first time in nearly a decade.", "monetary_policy"),
        ("2016-12-14", "Fed increases federal funds rate to 0.75 percent amid strengthening labor market.", "monetary_policy"),
        ("2017-06-14", "FOMC raises rates to 1.25 percent and unveils balance sheet runoff plan.", "monetary_policy"),
        ("2018-09-26", "Fed raises policy rate to 2.25 percent, removes term accommodative from official statement.", "monetary_policy"),
        ("2018-12-19", "Powell raises interest rates to 2.50 percent despite market volatility; equities tumble.", "monetary_policy"),
        # 2019 - Tagli precauzionali (Insurance cuts)
        ("2019-07-31", "Federal Reserve delivers first rate cut since 2008, lowering rates by 25 bps as trade tensions mount.", "monetary_policy"),
        ("2019-09-18", "FOMC cuts rates another 25 bps; Fed injects liquidity into overnight repo market.", "monetary_policy"),
        # 2020 - Shock pandemico Covid e stimoli massicci
        ("2020-03-03", "Federal Reserve cuts rates by 50 bps in emergency move to combat coronavirus economic fallout.", "monetary_policy"),
        ("2020-03-15", "Fed cuts rates to zero range and pledges 700 billion in emergency asset purchases.", "monetary_policy"),
        ("2020-08-27", "Powell announces average inflation targeting framework, tolerating moderate inflation overshoot.", "monetary_policy"),
        # 2021 - Surriscaldamento inflattivo
        ("2021-06-16", "Fed raises inflation projections, dot plot signals two rate hikes possible by late 2023.", "monetary_policy"),
        ("2021-11-03", "FOMC announces tapering of monthly asset purchases by 15 billion per month.", "monetary_policy"),
        ("2021-12-15", "Fed doubles pace of bond taper and forecasts three rate hikes in 2022 as inflation spikes.", "monetary_policy"),
        # 2022 - Aggressivo ciclo di rialzi tassi (75 bps hikes)
        ("2022-03-16", "Federal Reserve raises target range by 25 bps to combat forty-year high inflation.", "monetary_policy"),
        ("2022-06-15", "Fed hikes rates by 75 basis points, largest single rate increase since 1994.", "monetary_policy"),
        ("2022-09-21", "FOMC implements third consecutive 75 bps hike, reaffirming resolve to tame CPI inflation.", "monetary_policy"),
        ("2022-11-02", "Fed delivers fourth straight 75 bps hike, pushing benchmark rate above 3.75 percent.", "monetary_policy"),
        # 2023 - Crisi banche regionali e picco tassi a 5.50%
        ("2023-03-12", "Silicon Valley Bank and Signature Bank collapse; Fed announces Bank Term Funding Program.", "market_news"),
        ("2023-07-26", "Federal Reserve raises rates by 25 bps to 5.25-5.50 percent, highest level in 22 years.", "monetary_policy"),
        # 2024 - Ciclo di allentamento e tagli tassi
        ("2024-09-18", "Federal Reserve cuts interest rates by jumbo 50 basis points to 4.75-5.00 percent range.", "monetary_policy"),
        ("2024-11-07", "FOMC lowers target range by another 25 bps as inflation cools toward two percent target.", "monetary_policy"),
    ]

    records = []
    for dt, text, cat in events:
        records.append({
            "Date": dt,
            "headline": text,
            "description": text,
            "source": "historical_macro_calendar",
            "category": cat,
        })
    return records


def download_news_dataset(
    save_path: Optional[Union[str, Path]] = None,
    sources: Optional[List[str]] = None,
    limit_per_source: int = 100,
    include_historical_calendar: bool = True,
) -> pd.DataFrame:
    """
    Scarica, unifica e salva il database delle notizie finanziarie e macroeconomiche.

    Parametri:
        save_path: Percorso del file CSV di output (default: 'data/raw/news/financial_news.csv').
        sources: Lista di fonti da scaricare (opzioni: 'fed_monetary', 'fed_all', 'marketwatch', 'huggingface').
        limit_per_source: Numero massimo di notizie per fonte da scaricare via API.
        include_historical_calendar: Se True, include il catalogo storico curato FOMC/tassi (2004-2026).

    Ritorna:
        pd.DataFrame con colonne ['Date', 'headline', 'description', 'source', 'category']
        ordinato cronologicamente e de-duplicato.
    """
    if save_path is None:
        save_path = PROJECT_ROOT / "data" / "raw" / "news" / "financial_news.csv"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if sources is None:
        sources = ["fed_monetary", "fed_all", "marketwatch", "huggingface"]

    all_records: List[Dict[str, str]] = []

    print("\n" + "=" * 80)
    print("### DOWNLOAD DATABASE NOTIZIE FINANZIARIE & MACROECONOMIA ###")
    print(f"Destinazione file: {save_path}")
    print(f"Fonti selezionate: {sources}")
    print("=" * 80)

    # 1. Federal Reserve Monetary Policy Releases
    if "fed_monetary" in sources:
        print("[1/4] Download Federal Reserve Monetary Policy RSS...")
        fed_mon_records = fetch_rss_headlines(
            feed_url="https://www.federalreserve.gov/feeds/press_monetary.xml",
            source_name="federal_reserve_monetary",
            category="monetary_policy",
        )
        print(f"  -> Acquisite {len(fed_mon_records)} notizie FOMC/Politica Monetaria.")
        all_records.extend(fed_mon_records)

    # 2. Federal Reserve Board All Releases
    if "fed_all" in sources:
        print("[2/4] Download Federal Reserve All Press Releases...")
        fed_all_records = fetch_rss_headlines(
            feed_url="https://www.federalreserve.gov/feeds/press_all.xml",
            source_name="federal_reserve_all",
            category="fed_board",
        )
        print(f"  -> Acquisite {len(fed_all_records)} notizie generali Fed.")
        all_records.extend(fed_all_records)

    # 3. MarketWatch Top Stories
    if "marketwatch" in sources:
        print("[3/4] Download MarketWatch / Dow Jones Headlines...")
        mw_records = fetch_rss_headlines(
            feed_url="https://feeds.content.dowjones.io/public/rss/mw_topstories",
            source_name="marketwatch",
            category="financial_markets",
        )
        print(f"  -> Acquisite {len(mw_records)} notizie di mercato.")
        all_records.extend(mw_records)

    # 4. HuggingFace Financial News Dataset
    if "huggingface" in sources:
        print(f"[4/4] Download HuggingFace Financial Sentiment Dataset (limit: {limit_per_source})...")
        hf_records = fetch_huggingface_financial_news(limit=limit_per_source)
        print(f"  -> Acquisite {len(hf_records)} headline da HuggingFace.")
        all_records.extend(hf_records)

    # 5. Calendario Storico Macro (2004-2026)
    if include_historical_calendar:
        print("[5/5] Integrazione calendario storico FOMC & Politica Tassi USA (2004-2026)...")
        hist_records = build_historical_macro_calendar_news()
        print(f"  -> Aggiunti {len(hist_records)} eventi storici macroeconomici curati.")
        all_records.extend(hist_records)

    if not all_records:
        raise RuntimeError("Nessuna notizia scaricata o recuperata dalle fonti configurate.")

    df = pd.DataFrame(all_records)

    # Pulizia, rimozione duplicati e ordinamento per data
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["headline"]).sort_values(by="Date").reset_index(drop=True)

    # Salvataggio CSV
    df.to_csv(save_path, index=False)
    print("\n" + "-" * 80)
    print(f"Download completato: {len(df)} notizie uniche salvate in: {save_path}")
    print(f"Intervallo temporale coperto: {df['Date'].min().strftime('%Y-%m-%d')} -> {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Distribuzione per fonte:\n{df['source'].value_counts().to_string()}")
    print("-" * 80)

    return df


# =====================================================================
# 2. AGENTE FINBERT SENTIMENT ANALYSIS
# =====================================================================
class FinBERTAgent(BaseAgent):
    """
    Agente di Sentiment Analysis basato su Transformer pre-addestrato FinBERT:
    - Elabora testi/headline finanziarie e macroeconomiche.
    - Calcola probabilità softmax su positivo, negativo e neutrale.
    - Genera segnali di BUY (+1), SELL (-1) o HOLD (0) con stima continua della confidenza [0.0, 1.0].
    - Supporta il parametro 'invert_sentiment' (fondamentale per ETF su titoli di stato Treasuries).
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        confidence_threshold: float = 0.50,
        text_col: str = "headline",
        invert_sentiment: Optional[bool] = None,
        target_asset: Optional[str] = "SPY",
        output_format: str = "numeric",
        device: Optional[str] = None,
    ):
        super().__init__(
            name="FinBERTAgent",
            target_asset=target_asset,
            output_format=output_format,
        )
        self.model_name = model_name
        self.confidence_threshold = float(confidence_threshold)
        self.text_col = text_col
        self.target_asset = target_asset.upper() if target_asset else "SPY"
        # Per ETF obbligazionari (TLT, IEF, SHY) default invert_sentiment=True; per azionario (SPY) e oro (GLD) False
        if invert_sentiment is not None:
            self.invert_sentiment = bool(invert_sentiment)
        else:
            self.invert_sentiment = self.target_asset in ("TLT", "IEF", "SHY")
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.params = {
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "text_col": self.text_col,
            "invert_sentiment": self.invert_sentiment,
        }

        print(f"[FinBERTAgent] Caricamento modello '{self.model_name}' su device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        config_label2id = getattr(self.model.config, "label2id", {})
        norm_label2id = {str(k).lower(): v for k, v in config_label2id.items()}
        self.pos_idx = norm_label2id.get("positive", 0)
        self.neg_idx = norm_label2id.get("negative", 1)
        self.neu_idx = norm_label2id.get("neutral", 2)

    def _apply_params(self, params: Dict[str, Any]) -> None:
        """Aggiorna i parametri dell'agente dopo il caricamento da JSON."""
        if "confidence_threshold" in params:
            self.confidence_threshold = float(params["confidence_threshold"])
        if "text_col" in params:
            self.text_col = str(params["text_col"])
        if "invert_sentiment" in params:
            self.invert_sentiment = bool(params["invert_sentiment"])

    def download_news(
        self,
        save_path: Optional[Union[str, Path]] = None,
        sources: Optional[List[str]] = None,
        limit_per_source: int = 100,
        include_historical_calendar: bool = True,
    ) -> pd.DataFrame:
        """Metodo di istanza per richiamare download_news_dataset()."""
        return download_news_dataset(
            save_path=save_path,
            sources=sources,
            limit_per_source=limit_per_source,
            include_historical_calendar=include_historical_calendar,
        )

    def predict_with_confidence(
        self, df: pd.DataFrame, batch_size: int = 32
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Elabora il DataFrame di testi ed estrae la coppia (segnale, confidenza).

        Ritorna:
            Tupla (signals, confidence):
            - signals: pd.Series con valori in {-1, 0, 1}
            - confidence: pd.Series con valori in [0.0, 1.0]
        """
        if self.text_col not in df.columns:
            signals = pd.Series(0, index=df.index, dtype=int)
            conf = pd.Series(0.0, index=df.index, dtype=float)
            return self._format_signals(signals), conf

        signals = pd.Series(0, index=df.index, dtype=int)
        confidence = pd.Series(0.0, index=df.index, dtype=float)

        has_text_mask = df[self.text_col].notna() & (
            df[self.text_col].astype(str).str.strip() != ""
        )
        valid_indices = df.index[has_text_mask]
        valid_texts = df.loc[has_text_mask, self.text_col].astype(str).tolist()

        if not valid_texts:
            return self._format_signals(signals), confidence

        computed_signals: List[int] = []
        computed_confs: List[float] = []

        with torch.no_grad():
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self.device)

                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                for p in probs:
                    pos_p = float(p[self.pos_idx])
                    neg_p = float(p[self.neg_idx])
                    neu_p = (
                        float(p[self.neu_idx])
                        if self.neu_idx < len(p)
                        else 0.0
                    )

                    # Logica sentiment direzionale
                    if pos_p > neg_p and pos_p > neu_p and pos_p >= self.confidence_threshold:
                        sig = -1 if self.invert_sentiment else 1
                        conf = pos_p
                    elif neg_p > pos_p and neg_p > neu_p and neg_p >= self.confidence_threshold:
                        sig = 1 if self.invert_sentiment else -1
                        conf = neg_p
                    else:
                        sig = 0
                        conf = 0.0

                    computed_signals.append(sig)
                    computed_confs.append(round(conf, 4))

        signals.loc[valid_indices] = computed_signals
        confidence.loc[valid_indices] = computed_confs

        return self._format_signals(signals), confidence

    def predict(self, df: pd.DataFrame, batch_size: int = 32) -> pd.Series:
        """Calcola i soli segnali operativi discreti {-1, 0, 1}."""
        signals, _ = self.predict_with_confidence(df, batch_size=batch_size)
        return signals

    def generate_sentiment_signals(
        self,
        news_df: pd.DataFrame,
        calendar_index: Optional[pd.DatetimeIndex] = None,
        target_asset: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Analizza l'intero database di notizie con FinBERT, aggrega i segnali e le confidenze
        su base settimanale e li allinea al calendario del progetto.

        Ritorna:
            pd.DataFrame con esattamente 2 colonne:
                1. 'signal': {-1, 0, 1}
                2. 'confidence': [0.0, 1.0]
            indicizzato per data settimanale, pronto come input per l'Agente Orchestratore.
        """
        asset = (target_asset or self.target_asset or "SPY").upper()
        if calendar_index is None:
            try:
                calendar_index = self.load_market_data().index
            except Exception:
                pass

        print(f"\n[FinBERTAgent] Generazione segnali sentiment per asset {asset} su {len(news_df)} notizie...")

        # 1. Inferenza a livello di singola notizia
        sig_news, conf_news = self.predict_with_confidence(news_df)
        df_scored = news_df.copy()
        df_scored["signal"] = sig_news.values
        df_scored["confidence"] = conf_news.values

        # Convinzione direzionale per singola notizia
        df_scored["conviction"] = df_scored["signal"] * df_scored["confidence"]
        df_scored["Date"] = pd.to_datetime(df_scored["Date"])

        # 2. Aggregazione settimanale (W-FRI per coerenza con il resto della pipeline)
        # Se in una settimana ci sono più notizie, calcola il punteggio medio ponderato
        weekly_groups = df_scored.set_index("Date").resample("W-FRI")

        weekly_signals = []
        weekly_dates = []
        weekly_conf = []

        for week_end, group in weekly_groups:
            if len(group) == 0:
                continue

            active_conv = group["conviction"]
            # Media delle convinzioni nella settimana
            mean_conv = float(active_conv.mean())

            if mean_conv >= 0.20:
                sig = 1
                c = float(group.loc[group["signal"] == 1, "confidence"].mean()) if (group["signal"] == 1).any() else 0.60
            elif mean_conv <= -0.20:
                sig = -1
                c = float(group.loc[group["signal"] == -1, "confidence"].mean()) if (group["signal"] == -1).any() else 0.60
            else:
                sig = 0
                c = 0.0

            weekly_dates.append(week_end)
            weekly_signals.append(sig)
            weekly_conf.append(round(c, 4))

        df_weekly = pd.DataFrame(
            {"signal": weekly_signals, "confidence": weekly_conf},
            index=pd.DatetimeIndex(weekly_dates),
        )

        # 3. Allineamento al calendario comune della pipeline (se fornito)
        if calendar_index is not None:
            df_aligned = df_weekly.reindex(calendar_index)
            # Dove non ci sono notizie specifiche nella settimana, il segnale è FLAT (0, conf: 0.0)
            df_aligned["signal"] = df_aligned["signal"].fillna(0).astype(int)
            df_aligned["confidence"] = df_aligned["confidence"].fillna(0.0).astype(float)
            df_weekly = df_aligned

        assert list(df_weekly.columns) == ["signal", "confidence"]
        print(f"[FinBERTAgent] Segnali settimanali generati: {len(df_weekly)} settimane.")
        print(f"Distribuzione segnali: {dict(df_weekly['signal'].value_counts())}")

        return df_weekly

    def get_signals(self, df: pd.DataFrame, batch_size: int = 32) -> pd.Series:
        """Alias per retrocompatibilità."""
        return self.predict(df, batch_size=batch_size)

    def run(
        self,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        filepath: Optional[Union[str, Path]] = None,
        force_retrain: bool = False,
    ) -> pd.Series:
        """Metodo di interfaccia unificata per retrocompatibilità."""
        if test_df is None:
            raise ValueError("È necessario passare test_df per estrarre i segnali.")
        return self.predict(test_df)


# =====================================================================
# 3. ENTRY POINT Esecuzione Standalone dell'Agente FinBERT
# =====================================================================
if __name__ == "__main__":
    # 1. Scarica il database delle notizie
    news_csv_path = PROJECT_ROOT / "data" / "raw" / "news" / "financial_news.csv"
    news_df = download_news_dataset(
        save_path=news_csv_path,
        sources=["fed_monetary", "fed_all", "marketwatch", "huggingface"],
        limit_per_source=100,
        include_historical_calendar=True,
    )

    # 2. Inizializza FinBERT con invert_sentiment=True per bond/Treasuries
    agent = FinBERTAgent(invert_sentiment=True)

    # 3. Carica il calendario settimanale della pipeline da features_weekly.csv
    features_csv = PROJECT_ROOT / "data" / "processed" / "features_weekly.csv"
    if features_csv.exists():
        df_feat = pd.read_csv(features_csv, index_col=0, parse_dates=True)
        common_cal = df_feat.index
    else:
        common_cal = None

    # 4. Genera il DataFrame dei segnali sentiment con esattamente 2 colonne ['signal', 'confidence']
    df_sentiment = agent.generate_sentiment_signals(
        news_df=news_df,
        calendar_index=common_cal,
        target_asset="TLT",
    )

    # 5. Salva il file dei segnali sentiment in data/signals/
    signals_output_dir = PROJECT_ROOT / "data" / "signals"
    signals_output_dir.mkdir(parents=True, exist_ok=True)
    sentiment_csv_file = signals_output_dir / "signals_sentiment_tlt.csv"
    df_sentiment.to_csv(sentiment_csv_file)
    print(f"\n[OK] Segnali sentiment FinBERT salvati in: {sentiment_csv_file}")
    print(df_sentiment.head(15))

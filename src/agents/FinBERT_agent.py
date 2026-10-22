"""
FinBERT_agent.py

Agente di Sentiment Analysis basato su Transformer pre-addestrato FinBERT (ProsusAI/finbert)
e modulo di acquisizione / download di database di notizie finanziarie e macroeconomiche.

Funzionalità principali:
1. download_news_dataset(): acquisisce e struttura un database di notizie finanziarie e macroeconomiche da:
   - Federal Reserve Monetary Policy Press Releases (FOMC decisions, discount rate, rate policy)
   - Federal Reserve Board All Releases
   - MarketWatch / Dow Jones Financial Headlines
   - HuggingFace Financial News Sentiment dataset (zeroshot/twitter-financial-news-sentiment)
   - Calendario storico degli eventi macro/tassi USA (2004-2026) per allineamento In-Sample
2. FinBERTAgent:
   - Estrazione probabilità softmax (positivo, negativo, neutrale) con tokenizzazione a batch
   - Mappatura multi-asset differenziata per SHY, IEF, TLT:
     * Dovish / Tagli Fed / Rallentamento -> TLT: +1 (Massima Duration), SHY: 0, IEF: 0
     * Hawkish / Rialzi Fed / Inflazione alta -> SHY: +1 (Rifugio Difensivo / Breve Termine), TLT: -1 (Short Duration), IEF: 0
     * Neutro / Misto -> IEF: +1 (Duration Intermedia), SHY: 0, TLT: 0
   - Generazione di 3 DataFrame separati con esattamente 2 colonne: 'signal' e 'confidence'
     salvati in signals_sentiment_shy.csv, signals_sentiment_ief.csv, signals_sentiment_tlt.csv.
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
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
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

    if "fed_monetary" in sources:
        print("[1/4] Download Federal Reserve Monetary Policy RSS...")
        fed_mon_records = fetch_rss_headlines(
            feed_url="https://www.federalreserve.gov/feeds/press_monetary.xml",
            source_name="federal_reserve_monetary",
            category="monetary_policy",
        )
        print(f"  -> Acquisite {len(fed_mon_records)} notizie FOMC/Politica Monetaria.")
        all_records.extend(fed_mon_records)

    if "fed_all" in sources:
        print("[2/4] Download Federal Reserve All Press Releases...")
        fed_all_records = fetch_rss_headlines(
            feed_url="https://www.federalreserve.gov/feeds/press_all.xml",
            source_name="federal_reserve_all",
            category="fed_board",
        )
        print(f"  -> Acquisite {len(fed_all_records)} notizie generali Fed.")
        all_records.extend(fed_all_records)

    if "marketwatch" in sources:
        print("[3/4] Download MarketWatch / Dow Jones Headlines...")
        mw_records = fetch_rss_headlines(
            feed_url="https://feeds.content.dowjones.io/public/rss/mw_topstories",
            source_name="marketwatch",
            category="financial_markets",
        )
        print(f"  -> Acquisite {len(mw_records)} notizie di mercato.")
        all_records.extend(mw_records)

    if "huggingface" in sources:
        print(f"[4/4] Download HuggingFace Financial Sentiment Dataset (limit: {limit_per_source})...")
        hf_records = fetch_huggingface_financial_news(limit=limit_per_source)
        print(f"  -> Acquisite {len(hf_records)} headline da HuggingFace.")
        all_records.extend(hf_records)

    if include_historical_calendar:
        print("[5/5] Integrazione calendario storico FOMC & Politica Tassi USA (2004-2026)...")
        hist_records = build_historical_macro_calendar_news()
        print(f"  -> Aggiunti {len(hist_records)} eventi storici macroeconomici curati.")
        all_records.extend(hist_records)

    if not all_records:
        raise RuntimeError("Nessuna notizia scaricata o recuperata dalle fonti configurate.")

    df = pd.DataFrame(all_records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["headline"]).sort_values(by="Date").reset_index(drop=True)

    df.to_csv(save_path, index=False)
    print("\n" + "-" * 80)
    print(f"Download completato: {len(df)} notizie uniche salvate in: {save_path}")
    print(f"Intervallo temporale: {df['Date'].min().strftime('%Y-%m-%d')} -> {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Distribuzione per fonte:\n{df['source'].value_counts().to_string()}")
    print("-" * 80)

    return df


# =====================================================================
# 2. AGENTE FINBERT SENTIMENT ANALYSIS CON LOGICA MULTI-ASSET
# =====================================================================
class FinBERTAgent(BaseAgent):
    """
    Agente di Sentiment Analysis basato su Transformer pre-addestrato FinBERT:
    - Elabora testi/headline finanziarie e macroeconomiche.
    - Calcola probabilità softmax su positivo, negativo e neutrale.
    - Genera segnali differenziati e specifici per ciascuno dei 3 ETF dell'universo:
      * SHY: difensivo (+1) quando i tassi salgono e la duration soffre;
      * IEF: preferito (+1) nei regimi neutri o di transizione;
      * TLT: speculativo long (+1) sui tagli tassi e crisi; short (-1) sui rialzi tassi aggressivi.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        confidence_threshold: float = 0.50,
        text_col: str = "headline",
        invert_sentiment: bool = True,
        output_format: str = "numeric",
        device: Optional[str] = None,
    ):
        super().__init__(name="FinBERTAgent", output_format=output_format)
        self.model_name = model_name
        self.confidence_threshold = float(confidence_threshold)
        self.text_col = text_col
        self.invert_sentiment = bool(invert_sentiment)
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
        """Richiama il downloader di notizie."""
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

    def predict_multi_asset(
        self,
        news_df: pd.DataFrame,
        calendar_index: Optional[pd.DatetimeIndex] = None,
        threshold_duration: float = 0.20,
    ) -> Dict[str, pd.DataFrame]:
        """
        Mappa il sentiment economico in segnali differenziati e specifici per SHY, IEF e TLT:

        Logica Economica di Fixed Income:
        1. Sentiment Favorevole alla Duration (Dovish, Tagli Tassi, Rallentamento Economico):
           - Score >= +threshold_duration
           - TLT: +1 (beneficia al massimo del calo dei tassi a lunga scadenza)
           - IEF:  0
           - SHY:  0
        2. Sentiment Ostile alla Duration (Hawkish, Rialzi Tassi, Surriscaldamento Inflattivo):
           - Score <= -threshold_duration
           - TLT: -1 (protezione / short duration per evitare perdite di capitale)
           - SHY: +1 (rifugio difensivo su scadenze brevi / cash yield)
           - IEF:  0
        3. Sentiment Neutro o Misto:
           - -threshold_duration < Score < +threshold_duration
           - IEF: +1 (allocazione intermedia bilanciata)
           - SHY:  0
           - TLT:  0

        Ritorna:
            Dizionario {'SHY': df_shy, 'IEF': df_ief, 'TLT': df_tlt}, ciascuno con colonne ['signal', 'confidence'].
        """
        print(f"\n[FinBERTAgent] Calcolo sentiment multi-asset differenziato (SHY, IEF, TLT) su {len(news_df)} notizie...")

        # 1. Inferenza a livello di singola notizia
        sig_news, conf_news = self.predict_with_confidence(news_df)
        df_scored = news_df.copy()
        df_scored["signal"] = sig_news.values
        df_scored["confidence"] = conf_news.values
        df_scored["conviction"] = df_scored["signal"] * df_scored["confidence"]
        df_scored["Date"] = pd.to_datetime(df_scored["Date"])

        # 2. Aggregazione settimanale
        weekly_groups = df_scored.set_index("Date").resample("W-FRI")

        dates = []
        scores = []
        mean_confs = []

        for week_end, group in weekly_groups:
            if len(group) == 0:
                continue
            mean_c = float(group["conviction"].mean())
            mean_conf = float(group["confidence"].mean()) if (group["signal"] != 0).any() else 0.50

            dates.append(week_end)
            scores.append(mean_c)
            mean_confs.append(mean_conf)

        weekly_idx = pd.DatetimeIndex(dates)
        score_series = pd.Series(scores, index=weekly_idx)
        conf_series = pd.Series(mean_confs, index=weekly_idx)

        # Allineamento al calendario comune della pipeline
        if calendar_index is not None:
            score_series = score_series.reindex(calendar_index).fillna(0.0)
            conf_series = conf_series.reindex(calendar_index).fillna(0.0)
            target_idx = calendar_index
        else:
            target_idx = weekly_idx

        # 3. Mappatura differenziata per i 3 asset
        th = threshold_duration

        # TLT (Lunga Duration): +1 su tagli/dovish, -1 su rialzi/hawkish
        sig_tlt = np.where(score_series >= th, 1, np.where(score_series <= -th, -1, 0))
        conf_tlt = np.where(sig_tlt != 0, np.maximum(0.50, conf_series), 0.0)

        # SHY (Breve Duration / Cash): +1 solo come rifugio quando i tassi salgono (score <= -th)
        sig_shy = np.where(score_series <= -th, 1, 0)
        conf_shy = np.where(sig_shy != 0, np.maximum(0.50, conf_series), 0.0)

        # IEF (Duration Intermedia): +1 quando il sentiment è moderato/neutro
        sig_ief = np.where((score_series > -th) & (score_series < th) & (conf_series > 0.0), 1, 0)
        conf_ief = np.where(sig_ief != 0, np.maximum(0.50, conf_series), 0.0)

        df_tlt = pd.DataFrame({"signal": sig_tlt, "confidence": np.round(conf_tlt, 4)}, index=target_idx)
        df_shy = pd.DataFrame({"signal": sig_shy, "confidence": np.round(conf_shy, 4)}, index=target_idx)
        df_ief = pd.DataFrame({"signal": sig_ief, "confidence": np.round(conf_ief, 4)}, index=target_idx)

        for name, df_a in [("SHY", df_shy), ("IEF", df_ief), ("TLT", df_tlt)]:
            assert list(df_a.columns) == ["signal", "confidence"]
            print(f"  [{name}] Segnali generati: {dict(df_a['signal'].value_counts())}")

        return {
            "SHY": df_shy,
            "IEF": df_ief,
            "TLT": df_tlt,
        }

    def generate_sentiment_signals(
        self,
        news_df: pd.DataFrame,
        calendar_index: Optional[pd.DatetimeIndex] = None,
        target_asset: str = "TLT",
    ) -> pd.DataFrame:
        """Restituisce il DataFrame a due colonne specifico per il target_asset richiesto."""
        multi_dict = self.predict_multi_asset(news_df, calendar_index=calendar_index)
        asset_key = target_asset.upper()
        if asset_key in multi_dict:
            return multi_dict[asset_key]
        else:
            return multi_dict["TLT"]


# =====================================================================
# 3. ENTRY POINT Standalone: Genera e Salva i Segnali per Tutti i 3 Asset
# =====================================================================
def run_finbert_pipeline() -> Dict[str, pd.DataFrame]:
    """Scarica le notizie, calcola il sentiment FinBERT e salva i segnali per SHY, IEF e TLT."""
    news_csv_path = PROJECT_ROOT / "data" / "raw" / "news" / "financial_news.csv"
    if not news_csv_path.exists():
        news_df = download_news_dataset(save_path=news_csv_path)
    else:
        news_df = pd.read_csv(news_csv_path)

    agent = FinBERTAgent(invert_sentiment=True)

    features_csv = PROJECT_ROOT / "data" / "processed" / "features_weekly.csv"
    if features_csv.exists():
        df_feat = pd.read_csv(features_csv, index_col=0, parse_dates=True)
        common_cal = df_feat.index
    else:
        common_cal = None

    multi_signals = agent.predict_multi_asset(news_df=news_df, calendar_index=common_cal)

    signals_dir = PROJECT_ROOT / "data" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    for ticker, df_sig in multi_signals.items():
        out_file = signals_dir / f"signals_sentiment_{ticker.lower()}.csv"
        df_sig.to_csv(out_file)
        print(f"[OK] Segnale FinBERT salvato per {ticker}: {out_file}")

    return multi_signals


if __name__ == "__main__":
    run_finbert_pipeline()

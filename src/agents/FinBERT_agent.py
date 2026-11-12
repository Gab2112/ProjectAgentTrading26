"""Sentiment agent based on the FinBERT transformer (ProsusAI/finbert).

The module downloads financial headlines (public RSS feeds, an open-source
HuggingFace dataset and an asset-specific historical calendar), classifies
them with FinBERT and aggregates the results into weekly signals {-1, 0, 1}
with a confidence in [0, 1]. Sentiment is inverted for the defensive assets
(IEF, SHY, GLD). The agent is optional and requires torch and transformers.
"""
import html
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Robust path resolution.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from src.agents.base_agent import BaseAgent
from src.data_loader import load_agent_market_data


# =====================================================================
# 1. AUTOMATIC NEWS ACQUISITION BY ASSET
# =====================================================================
def clean_headline_text(text: str) -> str:
    """Remove HTML entities, XML tags, URLs and repeated whitespace from the text."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_rss_headlines(feed_url: str, source_name: str, category: str, timeout: int = 10) -> List[Dict[str, str]]:
    """Download and parse the headlines of a public XML RSS feed."""
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


def fetch_huggingface_financial_news(limit: int = 200, timeout: int = 10) -> List[Dict[str, str]]:
    """Download an open-source financial sentiment dataset through the HuggingFace Datasets API.
    """
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
                        "source": "huggingface_twitter_financial",
                        "category": "market_sentiment",
                    })
    except Exception as e:
        print(f"  [WARN] Download HuggingFace dataset non riuscito: {e}")

    return headlines


def build_asset_specific_historical_calendar(target_asset: str) -> List[Dict[str, str]]:
    """Return a curated historical catalogue of headlines for the requested asset.

    Covers SPY, GLD, IEF and SHY so that backtests have adequate coverage.
    """
    asset = target_asset.upper()

    macro_events = [
        ("2004-06-30", "Federal Reserve raises federal funds rate by 25 basis points to 1.25 percent.",
         "monetary_policy"),
        ("2007-09-18",
         "Fed slashes federal funds rate by 50 basis points to 4.75 percent to forestall credit contagion.",
         "monetary_policy"),
        ("2008-09-15", "Lehman Brothers files for bankruptcy protection; systemic credit panic erupts globally.",
         "market_news"),
        ("2008-12-16", "Fed cuts interest rates to zero and initiates first Quantitative Easing (QE1).",
         "monetary_policy"),
        ("2011-08-05", "Standard & Poor's downgrades United States sovereign credit rating from AAA to AA+.",
         "market_news"),
        ("2015-12-16", "Federal Reserve raises rates off zero bound for first time in nearly a decade.",
         "monetary_policy"),
        ("2020-03-15", "Fed cuts rates to zero and pledges massive emergency asset purchases.", "monetary_policy"),
        ("2022-06-15", "Fed hikes rates by 75 basis points, largest single rate increase since 1994.",
         "monetary_policy"),
        ("2024-09-18", "Federal Reserve cuts interest rates by jumbo 50 basis points to 4.75-5.00 percent range.",
         "monetary_policy"),
    ]

    gold_events = [
        ("2006-05-11",
         "Gold prices surge to multi-year highs above 700 dollars per ounce amid soaring inflation fears.",
         "gold_commodity"),
        ("2008-10-10", "Gold rallies sharply as investors flee equities seeking ultimate safe-haven bullion assets.",
         "gold_commodity"),
        ("2011-09-06",
         "Spot gold hits historic record high above 1,900 dollars per ounce driven by sovereign debt crisis.",
         "gold_commodity"),
        ("2020-08-07", "Gold breaks above 2,000 dollars per ounce amid pandemic stimulus and plunging real yields.",
         "gold_commodity"),
        ("2024-03-01",
         "Gold prices rally to record highs on central bank accumulation and escalating geopolitical risks.",
         "gold_commodity"),
    ]

    equity_events = [
        ("2007-10-09", "S&P 500 closes at all-time high before onset of global financial crisis.", "equity_market"),
        ("2009-03-09", "S&P 500 bottoms out at bear market low, marking beginning of historic bull market.",
         "equity_market"),
        ("2013-03-28", "S&P 500 fully recovers all financial crisis losses, reaching new record highs.",
         "equity_market"),
        ("2020-03-23", "Federal Reserve announces unlimited QE, sparking massive equity market rebound.",
         "equity_market"),
        ("2022-01-03", "S&P 500 hits record peak before entering bear market due to aggressive monetary tightening.",
         "equity_market"),
        ("2024-01-19",
         "S&P 500 confirms new bull market, closing at record high driven by artificial intelligence boom.",
         "equity_market"),
    ]

    selected_events = macro_events.copy()
    if asset == "GLD":
        selected_events.extend(gold_events)
    elif asset == "SPY":
        selected_events.extend(equity_events)

    records = []
    for dt, text, cat in selected_events:
        records.append({
            "Date": dt,
            "headline": text,
            "description": text,
            "source": f"historical_calendar_{asset.lower()}",
            "category": cat,
        })
    return records


def download_news_dataset(
        save_path: Optional[Union[str, Path]] = None,
        target_asset: str = "SPY",
        limit_per_source: int = 150,
) -> pd.DataFrame:
    """Build the news dataset for one asset from all sources and save it to CSV."""
    if save_path is None:
        save_path = PROJECT_ROOT / "data" / "raw" / "news" / f"financial_news_{target_asset.lower()}.csv"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n==============================================================")
    print(f" DOWNLOAD E COSTRUZIONE DATASET NOTIZIE PER ASSET: {target_asset.upper()}")
    print(f"==============================================================")

    all_records = []
    all_records.extend(fetch_rss_headlines("https://www.federalreserve.gov/feeds/press_monetary.xml", "federal_reserve",
                                           "monetary_policy"))
    all_records.extend(
        fetch_rss_headlines("https://feeds.content.dowjones.io/public/rss/mw_topstories", "marketwatch", "market_news"))
    all_records.extend(fetch_huggingface_financial_news(limit=limit_per_source))
    all_records.extend(build_asset_specific_historical_calendar(target_asset=target_asset))

    df = pd.DataFrame(all_records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["headline"]).sort_values(by="Date").reset_index(drop=True)

    df.to_csv(save_path, index=False)
    print(f"[OK] Dataset salvato in: {save_path} ({len(df)} record)")
    return df


# =====================================================================
# 2. FINBERT SENTIMENT AGENT
# =====================================================================
class FinBERTAgent(BaseAgent):
    """Sentiment agent that turns financial headlines into trading signals with FinBERT.
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
        super().__init__(name="FinBERTAgent", target_asset=target_asset, output_format=output_format)
        self.model_name = model_name
        self.confidence_threshold = float(confidence_threshold)
        self.text_col = text_col
        self.target_asset = target_asset.upper() if target_asset else "SPY"

        # Sentiment inversion: True for GLD, IEF and SHY; False for SPY.
        if invert_sentiment is not None:
            self.invert_sentiment = bool(invert_sentiment)
        else:
            self.invert_sentiment = self.target_asset in ("IEF", "SHY", "GLD")

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        print(
            f"[FinBERTAgent] Caricamento '{self.model_name}' per {self.target_asset} (invert={self.invert_sentiment})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        config_label2id = getattr(self.model.config, "label2id", {})
        norm_label2id = {str(k).lower(): v for k, v in config_label2id.items()}
        self.pos_idx = norm_label2id.get("positive", 0)
        self.neg_idx = norm_label2id.get("negative", 1)
        self.neu_idx = norm_label2id.get("neutral", 2)

    def download_news(self, save_path: Optional[Union[str, Path]] = None, limit_per_source: int = 150) -> pd.DataFrame:
        """Download the news dataset for the agent's target asset."""
        return download_news_dataset(save_path=save_path, target_asset=self.target_asset,
                                     limit_per_source=limit_per_source)

    def predict(self, df: pd.DataFrame, batch_size: int = 32) -> pd.Series:
        """Return the discrete signals (implementation of the BaseAgent abstract method).
        """
        signals, _ = self.predict_with_confidence(df, batch_size=batch_size)
        return signals

    def predict_with_confidence(self, df: pd.DataFrame, batch_size: int = 32) -> Tuple[pd.Series, pd.Series]:
        """Classify each headline and return its signal and confidence.

        A headline produces a signal only when the winning class (positive or
        negative) has a probability of at least ``confidence_threshold``; the sign
        is inverted for the defensive assets.
        """
        if self.text_col not in df.columns:
            signals = pd.Series(0, index=df.index, dtype=int)
            conf = pd.Series(0.0, index=df.index, dtype=float)
            return self._format_signals(signals), conf

        signals = pd.Series(0, index=df.index, dtype=int)
        confidence = pd.Series(0.0, index=df.index, dtype=float)

        has_text_mask = df[self.text_col].notna() & (df[self.text_col].astype(str).str.strip() != "")
        valid_indices = df.index[has_text_mask]
        valid_texts = df.loc[has_text_mask, self.text_col].astype(str).tolist()

        if not valid_texts:
            return self._format_signals(signals), confidence

        computed_signals, computed_confs = [], []

        with torch.no_grad():
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i: i + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(
                    self.device)
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                for p in probs:
                    pos_p, neg_p = float(p[self.pos_idx]), float(p[self.neg_idx])
                    neu_p = float(p[self.neu_idx]) if self.neu_idx < len(p) else 0.0

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

    def generate_sentiment_signals(self, news_df: pd.DataFrame,
                                   calendar_index: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """Aggregate headline sentiment into weekly (W-FRI) signals.

        A week is long when its mean conviction is at least 0.20 and short when it
        is at most -0.20; otherwise it is flat. If a calendar is supplied, the
        weekly signals are reindexed to it, with missing weeks set to flat.
        """
        if calendar_index is None:
            try:
                calendar_index = load_agent_market_data().index
            except Exception:
                pass

        sig_news, conf_news = self.predict_with_confidence(news_df)
        df_scored = news_df.copy()
        df_scored["signal"] = sig_news.values
        df_scored["confidence"] = conf_news.values
        df_scored["conviction"] = df_scored["signal"] * df_scored["confidence"]
        df_scored["Date"] = pd.to_datetime(df_scored["Date"])

        weekly_groups = df_scored.set_index("Date").resample("W-FRI")
        weekly_signals, weekly_dates, weekly_conf = [], [], []

        for week_end, group in weekly_groups:
            if len(group) == 0:
                continue
            mean_conv = float(group["conviction"].mean())
            if mean_conv >= 0.20:
                sig, c = 1, float(group.loc[group["signal"] == 1, "confidence"].mean()) if (
                            group["signal"] == 1).any() else 0.60
            elif mean_conv <= -0.20:
                sig, c = -1, float(group.loc[group["signal"] == -1, "confidence"].mean()) if (
                            group["signal"] == -1).any() else 0.60
            else:
                sig, c = 0, 0.0

            weekly_dates.append(week_end)
            weekly_signals.append(sig)
            weekly_conf.append(round(c, 4))

        df_weekly = pd.DataFrame({"signal": weekly_signals, "confidence": weekly_conf},
                                 index=pd.DatetimeIndex(weekly_dates))

        if calendar_index is not None:
            df_aligned = df_weekly.reindex(calendar_index)
            df_aligned["signal"] = df_aligned["signal"].fillna(0).astype(int)
            df_aligned["confidence"] = df_aligned["confidence"].fillna(0.0).astype(float)
            df_weekly = df_aligned

        return df_weekly


# =====================================================================
# 3. MULTI-ASSET EXECUTION (SPY, GLD, IEF, SHY)
# =====================================================================
if __name__ == "__main__":
    assets_to_process = ["SPY", "GLD", "IEF", "SHY"]

    for asset in assets_to_process:
        print(f"\n>>> Elaborazione pipeline dati e sentiment per: {asset}")
        agent = FinBERTAgent(target_asset=asset)
        news_df = agent.download_news(limit_per_source=150)
        df_signals = agent.generate_sentiment_signals(news_df=news_df)

        output_dir = PROJECT_ROOT / "data" / "signals"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"signals_sentiment_{asset.lower()}.csv"
        df_signals.to_csv(out_file)

        print(f"[OK] Segnali sentiment salvati in: {out_file}")
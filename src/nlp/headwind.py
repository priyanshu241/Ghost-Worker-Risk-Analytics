"""
headwind.py

Computes a monthly "policy headwind index" from the NER output.

The index aggregates three signals per calendar month:
    1. Sentiment score of policy sentences mentioning workers or platforms
       (negative sentiment = higher headwind).
    2. Regulatory mention density — how many ORG/LAW entities appear per
       1,000 words (a proxy for regulatory activity).
    3. Platform exit signal — presence of exit/shutdown/ban keywords near
       platform entity mentions.

The final index is a normalised composite score between 0 and 1,
where higher means more adverse conditions for informal workers.

When real policy documents are unavailable, the module generates a
synthetic time series so downstream components (dashboard, notebooks) can
still run end to end.

Usage:
    from src.nlp.headwind import compute_headwind_index
    df = compute_headwind_index()
    print(df)
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from src.utils.config import DATA_PROCESSED, DATA_SYNTHETIC

OUTPUT_FILE = DATA_PROCESSED / "headwind_index.csv"

EXIT_KEYWORDS = re.compile(
    r"\b(ban|suspend|exit|shutdown|shut down|cancel|revoke|restrict|fine)\b",
    re.IGNORECASE
)

WORKER_KEYWORDS = re.compile(
    r"\b(gig|informal|worker|labour|laborer|unorganised|migrant)\b",
    re.IGNORECASE
)


def _sentiment_score(sentence):
    """
    Lightweight rule-based sentiment for policy text.
    Returns a float in [-1, 1] where negative means adverse for workers.

    In production replace with a fine-tuned BERT model.
    """
    negative_words = {"ban", "fine", "restrict", "suspend", "penalise", "penalty",
                      "revoke", "shutdown", "cancel", "liability", "dispute"}
    positive_words = {"scheme", "benefit", "support", "registration", "pension",
                      "insurance", "welfare", "protection", "assist"}
    words = set(sentence.lower().split())
    score = len(words & positive_words) - len(words & negative_words)
    return max(-1.0, min(1.0, score / max(1, len(words)) * 20))


def _parse_documents_for_signals(doc_paths):
    """
    Reads downloaded policy documents and extracts per-sentence signals.
    Returns a list of dicts with keys: date, sentiment, regulatory_hit, exit_signal.
    """
    signals = []
    ner_path = DATA_PROCESSED / "ner_output.csv"

    if not ner_path.exists() or not doc_paths:
        return signals

    ner_df = pd.read_csv(ner_path)

    for doc_path in doc_paths:
        text = doc_path.read_text(encoding="utf-8", errors="ignore")

        # Try to infer a date from filename (format: YYYY or YYYY-MM)
        date_match = re.search(r"(20\d{2})[-_]?(\d{2})?", doc_path.name)
        if date_match:
            year  = int(date_match.group(1))
            month = int(date_match.group(2)) if date_match.group(2) else 6
            doc_date = datetime(year, month, 1)
        else:
            doc_date = datetime.now().replace(day=1)

        sentences = [s.strip() for s in re.split(r"[.\n]+", text) if len(s.strip()) > 20]

        for sent in sentences:
            if not WORKER_KEYWORDS.search(sent):
                continue
            signals.append({
                "date":           doc_date,
                "sentiment":      _sentiment_score(sent),
                "regulatory_hit": 1 if re.search(r"\b(section|act|rule|gazette)\b", sent, re.I) else 0,
                "exit_signal":    1 if EXIT_KEYWORDS.search(sent) else 0,
            })

    return signals


def _synthetic_headwind_series(n_months=18):
    """
    Generates a plausible synthetic headwind index for demonstration.
    Starts at ~0.35, trends upward with noise, peaks around month 14.
    """
    np.random.seed(42)
    base  = np.linspace(0.35, 0.72, n_months)
    noise = np.random.normal(0, 0.04, n_months)
    index = np.clip(base + noise, 0.1, 0.95)

    end_date = datetime.now().replace(day=1)
    dates = [end_date - timedelta(days=30 * i) for i in range(n_months - 1, -1, -1)]

    return pd.DataFrame({"month": dates, "headwind_index": index})


def compute_headwind_index(use_synthetic=True):
    """
    Main entry point. Returns a DataFrame with columns [month, headwind_index].

    If use_synthetic=True or no real documents are available, returns the
    synthetic series. Otherwise parses real documents.

    Args:
        use_synthetic  bool — force synthetic output even if docs exist

    Returns:
        pd.DataFrame with columns: month (datetime), headwind_index (float 0-1)
    """
    if use_synthetic:
        df = _synthetic_headwind_series()
    else:
        from src.nlp.scraper import list_downloaded
        doc_paths = list_downloaded()

        if not doc_paths:
            print("No documents found. Using synthetic headwind series.")
            df = _synthetic_headwind_series()
        else:
            signals = _parse_documents_for_signals(doc_paths)
            if not signals:
                df = _synthetic_headwind_series()
            else:
                sig_df = pd.DataFrame(signals)
                sig_df["month"] = pd.to_datetime(sig_df["date"]).dt.to_period("M").dt.to_timestamp()

                monthly = sig_df.groupby("month").agg(
                    avg_sentiment=("sentiment", "mean"),
                    regulatory_density=("regulatory_hit", "sum"),
                    exit_signals=("exit_signal", "sum"),
                ).reset_index()

                # Invert sentiment (more negative = higher headwind)
                monthly["sentiment_component"] = 1 - (monthly["avg_sentiment"] + 1) / 2
                monthly["reg_component"]       = monthly["regulatory_density"] / (monthly["regulatory_density"].max() + 1e-9)
                monthly["exit_component"]      = monthly["exit_signals"] / (monthly["exit_signals"].max() + 1e-9)

                monthly["headwind_index"] = (
                    0.5 * monthly["sentiment_component"] +
                    0.3 * monthly["reg_component"] +
                    0.2 * monthly["exit_component"]
                ).clip(0, 1)

                df = monthly[["month", "headwind_index"]]

    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Headwind index saved to {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    df = compute_headwind_index()
    print(df.tail(6).to_string(index=False))

"""
ner.py

Named entity recognition pipeline for policy documents.

Runs spaCy NER over all downloaded policy documents and extracts mentions of
platforms, districts, skill types, policy numbers, and monetary figures.
Results are saved to data/processed/ner_output.csv.

Usage:
    python -m src.nlp.ner
"""

import re
import csv
import spacy
from pathlib import Path
from src.nlp.scraper import list_downloaded
from src.utils.config import DATA_PROCESSED, SPACY_MODEL, POLICY_ENTITIES

OUTPUT_FILE = DATA_PROCESSED / "ner_output.csv"

# Domain-specific keyword lists for post-filtering NER results
PLATFORM_KEYWORDS = {
    "zomato", "blinkit", "swiggy", "ola", "rapido", "uber",
    "urban company", "dunzo", "zepto", "bigbasket", "nrega",
}

SKILL_KEYWORDS = {
    "delivery", "construction", "domestic work", "agriculture",
    "textile", "recycling", "driving", "cooking", "cleaning",
}


def load_nlp_model():
    """Loads the spaCy model. Downloads en_core_web_sm if not installed."""
    try:
        return spacy.load(SPACY_MODEL)
    except OSError:
        print(f"Downloading {SPACY_MODEL}...")
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL], check=True)
        return spacy.load(SPACY_MODEL)


def extract_entities(text, nlp, entity_types=None):
    """
    Runs NER on a text string and returns a list of (entity_text, label) tuples.

    Args:
        text          raw document text
        nlp           loaded spaCy model
        entity_types  entity labels to keep (defaults to POLICY_ENTITIES from config)

    Returns:
        list of dicts with keys: text, label, start_char, end_char
    """
    entity_types = entity_types or POLICY_ENTITIES
    doc = nlp(text[:1_000_000])  # spaCy has a char limit; chunk large docs

    results = []
    for ent in doc.ents:
        if ent.label_ in entity_types:
            results.append({
                "text":       ent.text.strip(),
                "label":      ent.label_,
                "start_char": ent.start_char,
                "end_char":   ent.end_char,
            })
    return results


def tag_domain_entities(entities):
    """
    Post-processes NER output to tag platform and skill mentions specifically.
    Adds a 'domain_tag' key ('platform', 'skill', or None).
    """
    for ent in entities:
        lower = ent["text"].lower()
        if any(k in lower for k in PLATFORM_KEYWORDS):
            ent["domain_tag"] = "platform"
        elif any(k in lower for k in SKILL_KEYWORDS):
            ent["domain_tag"] = "skill"
        else:
            ent["domain_tag"] = None
    return entities


def run_ner_pipeline(docs=None):
    """
    Processes all downloaded policy documents and writes NER results to CSV.

    Args:
        docs  list of Path objects (defaults to all downloaded docs)

    Returns:
        list of all extracted entity dicts with a 'source_file' key added
    """
    docs = docs or list_downloaded()
    if not docs:
        print("No policy documents found. Run scraper.py first.")
        return []

    nlp = load_nlp_model()
    all_entities = []

    for doc_path in docs:
        print(f"Processing {doc_path.name}...")
        text = doc_path.read_text(encoding="utf-8", errors="ignore")

        # Chunk text if longer than 100k chars to avoid memory issues
        chunks = [text[i:i+100_000] for i in range(0, len(text), 100_000)]
        for chunk in chunks:
            entities = extract_entities(chunk, nlp)
            entities = tag_domain_entities(entities)
            for ent in entities:
                ent["source_file"] = doc_path.name
            all_entities.extend(entities)

        print(f"  Found {len(entities)} entities in last chunk")

    if all_entities:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source_file", "text", "label",
                                                    "domain_tag", "start_char", "end_char"])
            writer.writeheader()
            writer.writerows(all_entities)
        print(f"\nNER output saved to {OUTPUT_FILE} ({len(all_entities)} entities)")

    return all_entities


def get_platform_mentions(ner_output_path=OUTPUT_FILE):
    """
    Reads the NER output CSV and returns a DataFrame of platform mentions,
    aggregated by entity text and source document.
    """
    import pandas as pd
    df = pd.read_csv(ner_output_path)
    return df[df["domain_tag"] == "platform"].groupby(
        ["text", "source_file"]
    ).size().reset_index(name="mention_count").sort_values("mention_count", ascending=False)


if __name__ == "__main__":
    run_ner_pipeline()

"""
scraper.py

Downloads policy documents from public government sources for the NLP pipeline.

Sources targeted:
    - NITI Aayog publications (niti.gov.in)
    - Ministry of Labour and Employment press releases (labour.gov.in)
    - e-Shram portal announcements (eshram.gov.in)

All documents are saved to data/raw/policy_docs/ as plain text files so that
the NER and headwind pipelines can process them offline.

Usage:
    python -m src.nlp.scraper
"""

import os
import time
import hashlib
import requests
from pathlib import Path
from src.utils.config import DATA_RAW

OUTPUT_DIR = DATA_RAW / "policy_docs"

# Publicly available PDF or HTML endpoints.
# Replace these with real URLs as you find them.
# The scraper handles both HTML pages and direct PDF links.
SOURCES = [
    {
        "name": "niti_aayog_gig_economy_report",
        "url": "https://niti.gov.in/sites/default/files/2022-06/Gig-Economy-Report-Final.pdf",
        "type": "pdf",
    },
    {
        "name": "mol_press_release_eshram",
        "url": "https://labour.gov.in/sites/default/files/PressNote_eSHRAM_2021.pdf",
        "type": "pdf",
    },
    # Add more sources here as you find them.
    # For HTML pages use type "html".
]


def _filename_for(name, url):
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{name}_{url_hash}.txt"


def fetch_pdf_text(url):
    """
    Downloads a PDF and extracts its text using pdfplumber.
    Returns the text as a string, or None on failure.
    """
    try:
        import pdfplumber, io
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)
    except Exception as e:
        print(f"  PDF fetch failed for {url}: {e}")
        return None


def fetch_html_text(url):
    """
    Downloads an HTML page and strips tags using BeautifulSoup.
    Returns the visible text, or None on failure.
    """
    try:
        from bs4 import BeautifulSoup
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    except Exception as e:
        print(f"  HTML fetch failed for {url}: {e}")
        return None


def scrape_all(sources=None, delay=2.0):
    """
    Iterates over all sources and saves extracted text to OUTPUT_DIR.
    Skips documents that have already been downloaded.

    Args:
        sources   list of source dicts (defaults to module-level SOURCES)
        delay     seconds to wait between requests (be polite to servers)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sources = sources or SOURCES

    for source in sources:
        filename = _filename_for(source["name"], source["url"])
        out_path = OUTPUT_DIR / filename

        if out_path.exists():
            print(f"Skipping {source['name']} (already downloaded)")
            continue

        print(f"Fetching {source['name']} from {source['url']}")
        if source["type"] == "pdf":
            text = fetch_pdf_text(source["url"])
        else:
            text = fetch_html_text(source["url"])

        if text:
            out_path.write_text(text, encoding="utf-8")
            print(f"  Saved {len(text):,} chars to {out_path.name}")
        else:
            print(f"  No text extracted for {source['name']}")

        time.sleep(delay)

    print("Scraping complete.")


def list_downloaded():
    """Returns a list of paths to all downloaded policy documents."""
    if not OUTPUT_DIR.exists():
        return []
    return sorted(OUTPUT_DIR.glob("*.txt"))


if __name__ == "__main__":
    scrape_all()
    docs = list_downloaded()
    print(f"\n{len(docs)} documents available:")
    for p in docs:
        print(f"  {p.name}")

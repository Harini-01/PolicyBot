# src/crawler.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import yaml
import tldextract
from requests.adapters import HTTPAdapter, Retry
import urllib3
from pathlib import Path

# ----------------------------
# Disable SSL warnings
# ----------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------------
# Paths
# ----------------------------
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
DOC_LINKS_FILE = RAW_DIR / "doc_links.yaml"
DOWNLOADED_FILE = RAW_DIR / "already_downloaded.yaml"

# ----------------------------
# Crawl a single site recursively
# ----------------------------
def crawl_site(base_url, max_depth=2):
    print(f"Crawling {base_url} ...")
    visited = set()
    docs = set()

    # Requests session with retries
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    domain = tldextract.extract(base_url).top_domain_under_public_suffix

    def crawl(url, depth):
        if depth > max_depth or url in visited:
            return
        visited.add(url)

        try:
            response = session.get(url, verify=False, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to crawl {url}: {e}")
            return

        soup = BeautifulSoup(response.text, "lxml")

        for a_tag in soup.find_all("a", href=True):
            link = urljoin(url, a_tag['href'])
            parsed_link = urlparse(link)

            # Skip external domains
            if tldextract.extract(parsed_link.netloc).top_domain_under_public_suffix != domain:
                continue

            # Only PDFs
            if link.lower().endswith(".pdf"):
                docs.add(link)
            else:
                # Recurse into internal pages
                crawl(link, depth + 1)

    crawl(base_url, 0)
    print(f"Found {len(docs)} PDFs on {base_url}")
    return list(docs)

# ----------------------------
# Main function
# ----------------------------
def main():
    config_path = "config/sites.yaml"
    if not Path(config_path).exists():
        print(f"{config_path} not found!")
        return

    with open(config_path, "r") as f:
        sites = yaml.safe_load(f).get("sites", [])

    if not sites:
        print("No sites found in sites.yaml")
        return

    # Load already downloaded PDFs to avoid duplicates
    if DOWNLOADED_FILE.exists():
        with open(DOWNLOADED_FILE) as f:
            downloaded = yaml.safe_load(f)
        downloaded_urls = {doc["url"] for doc in downloaded} if downloaded else set()
    else:
        downloaded_urls = set()

    all_docs = []
    for site in sites:
        try:
            docs = crawl_site(site)
            # Keep only new PDFs
            new_docs = [doc for doc in docs if doc not in downloaded_urls]
            all_docs.extend(new_docs)
        except Exception as e:
            print(f"Error crawling {site}: {e}")

    # Save new doc links
    with open(DOC_LINKS_FILE, "w") as f:
        yaml.dump(all_docs, f, sort_keys=False)

    print(f"Crawling completed. {len(all_docs)} new PDFs saved to {DOC_LINKS_FILE}")

# ----------------------------
# Run as script
# ----------------------------
if __name__ == "__main__":
    main()

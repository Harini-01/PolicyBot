# src/downloader.py

import os
import yaml
import requests
from urllib.parse import urlparse
from pathlib import Path

RAW_DIR = Path("data/raw")
DOC_LINKS_FILE = RAW_DIR / "doc_links.yaml"
DOWNLOADED_FILE = RAW_DIR / "already_downloaded.yaml"

# ----------------------------
# Load document links
# ----------------------------
def load_doc_links(path=DOC_LINKS_FILE):
    if not path.exists():
        return []
    with open(path, "r") as f:
        return yaml.safe_load(f) or []

# ----------------------------
# Load already downloaded
# ----------------------------
def load_downloaded():
    if DOWNLOADED_FILE.exists():
        with open(DOWNLOADED_FILE) as f:
            return yaml.safe_load(f) or []
    return []

# ----------------------------
# Save already downloaded
# ----------------------------
def save_downloaded(downloaded_list):
    with open(DOWNLOADED_FILE, "w") as f:
        yaml.dump(downloaded_list, f, sort_keys=False)

# ----------------------------
# Sanitize filenames
# ----------------------------
def sanitize_filename(url):
    filename = os.path.basename(urlparse(url).path)
    if not filename:
        filename = "unknown_file"
    return filename

# ----------------------------
# Download a single file
# ----------------------------
def download_file(url, save_path):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

# ----------------------------
# Main function
# ----------------------------
def main():
    doc_links = load_doc_links()
    downloaded = load_downloaded()
    downloaded_urls = {doc["url"] for doc in downloaded}

    for link in doc_links:
        if link in downloaded_urls:
            print(f"Already downloaded: {link}")
            continue

        # Create site folder
        site_folder_name = urlparse(link).netloc.replace(".", "_")
        site_folder = RAW_DIR / site_folder_name
        site_folder.mkdir(exist_ok=True, parents=True)

        filename = sanitize_filename(link)
        save_path = site_folder / filename

        if download_file(link, save_path):
            # Track downloaded PDFs
            downloaded.append({"url": link, "path": str(save_path)})

    # Save updated downloaded list
    save_downloaded(downloaded)
    print("All downloads completed.")

# ----------------------------
# Run as script
# ----------------------------
if __name__ == "__main__":
    main()

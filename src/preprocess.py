# src/preprocess.py

import os
from pathlib import Path
import pdfplumber
from docx import Document
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Extract text from PDF
# ----------------------------
def extract_pdf(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Failed to extract PDF {path}: {e}")
    return text

# ----------------------------
# Extract text from DOCX
# ----------------------------
def extract_docx(path):
    text = ""
    try:
        doc = Document(path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    except Exception as e:
        print(f"Failed to extract DOCX {path}: {e}")
    return text

# ----------------------------
# Extract text from HTML
# ----------------------------
def extract_html(path):
    text = ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "lxml")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator="\n")
    except Exception as e:
        print(f"Failed to extract HTML {path}: {e}")
    return text

# ----------------------------
# Clean extracted text
# ----------------------------
def clean_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

# ----------------------------
# Main preprocessing function
# ----------------------------
def main():
    for site_folder in RAW_DIR.iterdir():
        if not site_folder.is_dir():
            continue
        clean_site_folder = CLEAN_DIR / site_folder.name
        clean_site_folder.mkdir(exist_ok=True, parents=True)

        for file_path in site_folder.iterdir():
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()

            # Only process PDFs for govt circulars, skip others if you want
            if ext != ".pdf":
                print(f"Skipping non-PDF file {file_path}")
                continue

            text = extract_pdf(file_path)
            text = clean_text(text)

            if text:
                out_file = clean_site_folder / (file_path.stem + ".txt")
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Processed {file_path} → {out_file}")

if __name__ == "__main__":
    main()

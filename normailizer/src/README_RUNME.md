# UIT Law Pipeline — PDF → PNG → TEXT → DB

This is a **standalone pipeline** that converts **PDF (scan or digital)** → **PNG images** → **text** → **SQLite DB**
with hierarchical structure (**Chương → Điều → Khoản → Điểm**) and **FTS5**.

## 0) Setup

1) Install Python 3.10+
2) Create venv & install requirements
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
3) Install **Tesseract OCR** (required for scanned PDFs):
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`
   - If binary is not on PATH, set env `TESSERACT_CMD` or edit `config.py`.

## 1) Quick start

```bash
# initialize DB (creates schema & FTS)
python build_db.py --db uit_law.db

# run pipeline on a folder of PDFs (scan or digital)
python run_pipeline.py --pdf_dir ./pdfs --db uit_law.db

# or run on a single PDF
python run_pipeline.py --pdf ./pdfs/790-qd-dhcntt_28-9-22_quy_che_dao_tao.pdf --db uit_law.db
```

## 2) What it does

- Renders **each PDF page → PNG** (folder `outputs/png/<pdfname>/<page>.png`)
- Extracts **text layer** (for digital PDFs) and **OCR fallback** (for scan pages)
- Parses headings to **Chương / Điều / Khoản / Điểm**
- Writes to **SQLite (`uit_law.db`)** with **FTS5** and triggers

## 3) Structure

- `config.py` — Tesseract path/config, image DPI, OCR languages.
- `schema.sql` — DB schema, FTS5, triggers, views.
- `build_db.py` — create DB schema.
- `pdf2png.py` — render PDF pages to PNG using PyMuPDF.
- `ocr_text.py` — hybrid text extractor (text layer + OCR fallback).
- `parse_structure.py` — split into Điều → Khoản → Điểm.
- `import_to_db.py` — insert documents/items into DB.
- `run_pipeline.py` — orchestrates the whole process.

## 4) Notes

- The parser is regex-based and robust for common UIT formats:
  - **Chương**: `^\s*Chương\s+([IVXLCDM]+|\d+)`
  - **Điều**: `^\s*Điều\s+(\d+)`
  - **Khoản**: `^\s*(\d+)[\.\)]`
  - **Điểm**: `^\s*([a-zA-Z])\)`
- For pure scanned PDFs, ensure good OCR accuracy: correct language `vie` and DPI.
- You can re-run `run_pipeline.py` anytime; it will **skip duplicates** by SHA256 checksum.

## 5) Regenerate Mermaid / CSV

This minimal pipeline focuses on **PDF → DB**. For Mermaid/CSV exports you can use the earlier tools,
or add your own export scripts.


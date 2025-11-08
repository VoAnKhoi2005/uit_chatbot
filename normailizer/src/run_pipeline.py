import argparse, re
from pathlib import Path
from import_to_db import ensure_schema, insert_document
from ocr_text import pdf_to_text
from parse_structure import parse_chapters_and_articles

def infer_meta_from_filename(path: Path):
    base = path.stem
    token = base.split("_")[0]
    so_hieu = token.replace("-", "/").upper()
    title = base.replace("_", " ").strip().title()
    return so_hieu, title

def process_pdf(db: Path, pdf: Path, so_hieu=None, title=None):
    txt = pdf_to_text(pdf)
    chapters = parse_chapters_and_articles(txt)
    if not so_hieu or not title:
        so, ti = infer_meta_from_filename(pdf)
        so_hieu = so_hieu or so
        title = title or ti
    res = insert_document(db, pdf, so_hieu, title, chapters)
    return res

def main():
    ap = argparse.ArgumentParser(description="PDF → PNG → TEXT → DB (UIT Law)")
    ap.add_argument("--db", required=True, help="Path to output DB (SQLite)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf", help="Single PDF path")
    g.add_argument("--pdf_dir", help="Directory of PDFs (recursive)")
    ap.add_argument("--so_hieu", help="Override so_hieu (for single file)")
    ap.add_argument("--title", help="Override title (for single file)")
    args = ap.parse_args()

    db = Path(args.db)
    ensure_schema(db)

    stats = []
    if args.pdf:
        stats.append(process_pdf(db, Path(args.pdf), args.so_hieu, args.title))
    else:
        root = Path(args.pdf_dir)
        for p in root.rglob("*.pdf"):
            stats.append(process_pdf(db, p))
    for s in stats:
        print(s)

if __name__ == "__main__":
    main()

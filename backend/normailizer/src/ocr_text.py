from pathlib import Path
import pytesseract, os
from PIL import Image
import fitz

from config import TESSERACT_CMD, OCR_LANG, PNG_OUT_ROOT

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_text_layer(pdf_path: Path) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        return "\n".join([doc.load_page(i).get_text("text") for i in range(len(doc))])
    except Exception:
        return ""

def ocr_folder(png_dir: Path) -> str:
    texts = []
    for img_path in sorted(png_dir.glob("*.png")):
        try:
            img = Image.open(img_path)
            txt = pytesseract.image_to_string(img, lang=OCR_LANG)
        except Exception as e:
            txt = ""
        texts.append(txt or "")
    return "\n".join(texts)

def pdf_to_text(pdf_path: Path, png_dir: Path=None) -> str:
    # Try text layer first
    t = extract_text_layer(pdf_path) or ""
    if t.strip():
        return t
    # fallback: OCR images
    from pdf2png import pdf_to_pngs
    png_dir = png_dir or pdf_to_pngs(pdf_path)
    return ocr_folder(png_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--png_dir", default=None)
    args = ap.parse_args()
    print(pdf_to_text(Path(args.pdf), Path(args.png_dir) if args.png_dir else None))

import fitz, os
from pathlib import Path
from config import RENDER_DPI, PNG_OUT_ROOT

def pdf_to_pngs(pdf_path: Path, out_root: Path=None):
    out_root = Path(out_root or PNG_OUT_ROOT)
    out_dir = out_root / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=RENDER_DPI)
        out_file = out_dir / f"page_{i+1:04d}.png"
        pix.save(out_file.as_posix())
    return out_dir

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()
    d = pdf_to_pngs(Path(args.pdf), args.out_root)
    print("PNG saved to:", d)

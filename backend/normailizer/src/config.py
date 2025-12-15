import os

# If Tesseract isn't on PATH, set absolute path here, e.g.:
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", None)

# OCR languages (install traineddata in your Tesseract)
# 'vie' for Vietnamese, add 'eng' if mixed text
OCR_LANG = os.environ.get("OCR_LANG", "vie+eng")

# Render resolution for PDF->PNG (DPI). 200-300 is a good trade-off.
RENDER_DPI = int(os.environ.get("RENDER_DPI", "220"))

# PNG output root
PNG_OUT_ROOT = os.environ.get("PNG_OUT_ROOT", "outputs/png")

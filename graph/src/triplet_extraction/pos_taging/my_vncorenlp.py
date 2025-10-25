from . import *

vncorenlp_pos_map = {
    "N":   "Noun (Danh từ)",
    "Np":  "Proper noun (Danh từ riêng)",
    "Nc":  "Classifier noun (Danh từ giống loại)",
    "Nu":  "Unit noun (Danh từ đơn vị)",
    "V":   "Verb (Động từ)",
    "Vb":  "Verb (base) (Động từ gốc)",
    "A":   "Adjective (Tính từ)",
    "Ai":  "Adjective (predicative) (Tính từ vị ngữ)",
    "P":   "Pronoun (Đại từ)",
    "R":   "Adverb (Trạng từ)",
    "M":   "Numeral / number (Số từ)",
    "E":   "Preposition / particle (Giới từ / trợ từ)",
    "C":   "Coordinating conjunction (Liên từ phối hợp)",
    "CC":  "Subordinating conjunction / complementizer (Liên từ phụ thuộc)",
    "L":   "Determiner / article (Từ hạn định)",
    "D":   "Adverbial marker / degree marker (Từ chỉ mức độ)",
    "X":   "Other (Khác)",
    "CH":  "Punctuation (Dấu câu)"
}

def init_vncorenlp(vncorenlp_dir, annotators=None):
    if annotators is None:
        annotators = ["wseg"] #["wseg", "pos", "ner", "parse"]
    if "rdrsegmenter" not in globals():
        import py_vncorenlp
        rdrsegmenter = py_vncorenlp.VnCoreNLP(
            annotators=annotators,
            save_dir=vncorenlp_dir
        )
    return rdrsegmenter
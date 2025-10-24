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


def process_sentence(text: str, rdrsegmenter, verbose: bool = True) -> dict:
    results = {
        "original": text,
        "cleaned": "",
        "segmented": [],
        "pos_annotation": [],
        "concepts": []
    }

    if verbose:
        print("1. Original text:\n", text, "\n")

    # 2. Clean
    text = clean_text(text)
    results["cleaned"] = text
    if verbose:
        print("2. Cleaned text:\n", text, "\n")

    # 3. Segmentation
    segmented = rdrsegmenter.word_segment(text)
    results["segmented"] = segmented
    if verbose:
        print("3. Segmented text (tokens):\n", segmented, "\n")

    # 4. POS tagging
    output = rdrsegmenter.annotate_text(text)
    sents = output.values() if isinstance(output, dict) else output

    pos_annot = []
    tokens = []

    if verbose:
        print("4. POS & Head & DepRel annotation:")
        print(f"{'Idx':<5} {'Token':<15} {'Head':<5} {'DepRel':<15} {'POS':<25}")
        print("-" * 75)

    for sent in sents:
        if not isinstance(sent, list):
            continue
        for token in sent:
            if not isinstance(token, dict):
                continue
            word = token.get("wordForm", "")
            pos = token.get("posTag", "")
            head = token.get("head", "")
            dep = token.get("depLabel", "")
            pos_full = vncorenlp_pos_map.get(pos, pos)

            pos_data = {
                "index": token.get("index", ""),
                "token": word,
                "pos": pos_full,
                "head": head,
                "dep": dep
            }
            pos_annot.append(pos_data)

            if verbose:
                print(f"{pos_data['index']:<5} {pos_data['token']:<15} {pos_data['head']:<5} {pos_data['dep']:<15} {pos_data['pos']:<25}")

            if pos.startswith(("N", "V", "A", "M")):
                tokens.append((word.replace("_", " "), pos))

    results["pos_annotation"] = pos_annot

    # 5. Triplet extraction
    triplets = []
    concept1_tokens = []
    concept2_tokens = []
    relation_tokens = []

    for token, pos in tokens:
        if pos.startswith("V"):
            if concept2_tokens:
                triplet = (
                    " ".join(concept1_tokens),
                    " ".join(relation_tokens),
                    " ".join(concept2_tokens)
                )
                if triplet not in triplets:  # chỉ thêm nếu chưa tồn tại
                    triplets.append(triplet)

                concept1_tokens = concept2_tokens
                concept2_tokens = []
                relation_tokens = []
            relation_tokens.append(token)
        else:
            if relation_tokens:
                concept2_tokens.append(token)
            else:
                concept1_tokens.append(token)

    # Append last triplet if complete
    if concept1_tokens and relation_tokens and concept2_tokens:
        triplets.append((
            " ".join(concept1_tokens),
            " ".join(relation_tokens),
            " ".join(concept2_tokens)
        ))

    results["concepts"] = triplets

    if verbose:
        print("\n5. Extracted triplets:")
        for c in triplets:
            print(c)

    return results
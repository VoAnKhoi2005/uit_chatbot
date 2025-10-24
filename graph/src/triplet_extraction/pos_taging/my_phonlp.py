from itertools import product
import pandas as pd

def parsing_result(annotation):
    words = annotation[0][0]
    pos_tags_nested = annotation[1][0]
    ner_tags = annotation[2][0]
    dep_tags_nested = annotation[3][0]
    # Create the 'id' list
    ids = list(range(1, len(words) + 1))

    # Flatten the POS tags
    pos_tags = [tag[0] for tag in pos_tags_nested]

    # Split the dependency tags
    heads = [dep[0] for dep in dep_tags_nested]
    deprels = [dep[1] for dep in dep_tags_nested]

    data = {
        'id': ids,
        'word': words,
        'pos': pos_tags,
        # 'ner': ner_tags,
        'head': heads,
        'deprel': deprels
    }

    return pd.DataFrame(data)

def process_sentence(df):
    data_dict = df.set_index('id').to_dict(orient='index')

    if data_dict[1]['deprel'] == 'root':
        return None

    # === 1. Tiền xử lý ===
    subjects_of_verbs = {}
    for token_id, info in data_dict.items():
        if info['deprel'] == 'sub':
            head_id = info['head']
            subjects_of_verbs.setdefault(head_id, []).append(token_id)

    # === 2. Khởi tạo ===
    triplets = []
    concept1_groups = [[]]
    concept2_groups = [[]]
    relation_groups = [[]]

    verb_deprel_tag = ['root', 'vmod', 'nmod', 'x', 'conj', 'prd', 'tpc', 'dep']
    remove_POS_tag = ["R", "CH", "E", "L"]
    coord_POS_tag = ["C"]
    coord_words = {"và", "hoặc", ","}

    def ids_to_string(id_list):
        id_list = sorted(set(id_list))
        return " ".join(data_dict[tid]['word'] for tid in id_list if tid in data_dict)

    def is_coord_word(info):
        return (info["word"].lower() in coord_words or
                any(info['pos'].startswith(prefix) for prefix in coord_POS_tag))

    token_ids = sorted(list(data_dict.keys()))
    token_set = set(token_ids)

    # === 3. Duyệt tokens ===
    last_was_coord = False

    for token_id in token_ids:
        info = data_dict[token_id]

        # Logic bỏ qua token
        if any(info['pos'].startswith(prefix) for prefix in remove_POS_tag) and info['deprel'] != 'root':
            continue

        # Logic từ nối
        if is_coord_word(info):
            last_was_coord = True
            continue

        is_relation = False

        # --- Logic 'vmod' (bắt buộc phải là 'V') ---
        is_vmod_continuation = (
            info['pos'].startswith("V")
            and info['deprel'] == "vmod"
            and any(g for g in relation_groups if g)
        )

        # --- Logic 'relation' khác (bắt buộc phải là 'V') ---
        prev_is_n = (token_id - 1 in token_set and data_dict[token_id - 1]['pos'].startswith("N"))
        next_is_n = (token_id + 1 in token_set and data_dict[token_id + 1]['pos'].startswith("N"))

        is_valid_relation_type = (
            info['pos'].startswith("V")
            and info['deprel'] in verb_deprel_tag
            and info['deprel'] not in ("vmod", "root")
            and (info['deprel'] != "nmod" or (prev_is_n and next_is_n))
        )

        # --- Logic 'root' (Có thể là 'V' hoặc 'R') ---
        is_valid_root = False
        if info['deprel'] == "root":
            is_valid_root = True

        # Quyết định cuối cùng
        if is_vmod_continuation or is_valid_relation_type or is_valid_root:
            is_relation = True
        target_groups = None

        if is_relation:
            # print(token_id) # Bỏ comment để debug
            if any(g for g in concept2_groups if g):
                # Hoàn thành triplet cũ
                for c1g, rg, c2g in product(concept1_groups, relation_groups, concept2_groups):
                    triplet = (ids_to_string(c1g), ids_to_string(rg), ids_to_string(c2g))
                    if triplet not in triplets and c1g and rg and c2g:
                        triplets.append(triplet)

                concept1_groups = concept2_groups[:]
                concept2_groups = [[]]
                relation_groups = [[]]

            target_groups = relation_groups
        else:
            # Nó là một Concept
            if any(g for g in relation_groups if g):
                target_groups = concept2_groups
            else:
                target_groups = concept1_groups

        if last_was_coord:
            # FORK: Tạo nhóm mới
            if target_groups is not None:
                target_groups.append([token_id])
            last_was_coord = False
        else:
            # APPEND: Thêm vào nhóm cuối
            if target_groups is not None:
                if not target_groups or not target_groups[-1]:
                     target_groups.append([token_id])
                else:
                     target_groups[-1].append(token_id)

    # === 6. Kết thúc (Giữ nguyên) ===
    for c1g, rg, c2g in product(concept1_groups, relation_groups, concept2_groups):
        if c1g and rg and c2g:
            triplet = (
                ids_to_string(c1g),
                ids_to_string(rg),
                ids_to_string(c2g)
            )
            if triplet not in triplets:
                triplets.append(triplet)
    return triplets
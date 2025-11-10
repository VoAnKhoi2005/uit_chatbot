from itertools import product
from graph.src.triplet_extraction import load_stopwords, is_valid_term

def process_sentence(df):
    data_dict = df.set_index('id').to_dict(orient='index')

    if data_dict[1]['deprel'] == 'root':
        return None

    # Khởi tạo
    triplets = []
    concept1_groups = [[]]
    concept2_groups = [[]]
    relation_groups = [[]]

    verb_deprel_tag = ['root', 'vmod', 'nmod', 'x', 'conj', 'prd', 'tpc', 'dep']
    remove_POS_tag = ["R", "CH", "E", "L", "M"]
    coord_POS_tag = ["C"]
    coord_words = {"và", "hoặc", ",", ";", "-"}

    # Tiền xử lý
    token_ids = sorted(list(data_dict.keys()))
    filtered_token_ids = []
    for token_id in token_ids:
        info = data_dict[token_id]

        # Skip token if POS matches remove_POS_tag and not root
        if any(info['pos'].startswith(prefix) for prefix in remove_POS_tag) and info['deprel'] != 'root':
            continue

        filtered_token_ids.append(token_id)

    token_set = set(filtered_token_ids)
    subjects_of_verbs = {}
    for token_id in filtered_token_ids:
        info = data_dict[token_id]
        if info['deprel'] == 'sub':
            head_id = info['head']
            if head_id in token_set:
                subjects_of_verbs.setdefault(head_id, []).append(token_id)

    def ids_to_string(id_list):
        id_list = sorted(set(id_list))
        return " ".join(data_dict[tid]['word'] for tid in id_list if tid in data_dict)

    def is_coord_word(info):
        return info["deprel"] != "root" and (info["word"].lower() in coord_words or any(info['pos'].startswith(prefix) for prefix in coord_POS_tag))

    last_was_coord = False
    for token_id in filtered_token_ids:
        info = data_dict[token_id]

        # Logic từ nối
        if is_coord_word(info):
            last_was_coord = True
            continue

        is_relation = False

        # 'vmod' (bắt buộc phải là 'V')
        is_vmod_continuation = (
            info['pos'].startswith("V")
            and info['deprel'] == "vmod"
            and any(g for g in relation_groups if g)
        )

        # 'relation' khác (bắt buộc phải là 'V')
        prev_is_n = (token_id - 1 in token_set and data_dict[token_id - 1]['pos'].startswith("N"))
        next_is_n = (token_id + 1 in token_set and data_dict[token_id + 1]['pos'].startswith("N"))

        is_valid_relation_type = (
            info['pos'].startswith("V")
            and info['deprel'] in verb_deprel_tag
            and info['deprel'] not in ("vmod", "root")
            and (info['deprel'] != "nmod" or (prev_is_n and next_is_n))
        )

        # 'root' (Có thể là 'V' hoặc 'R')
        is_valid_root = False
        if info['deprel'] == "root":
            is_valid_root = True

        if is_vmod_continuation or is_valid_relation_type or is_valid_root:
            is_relation = True
        target_groups = None

        if is_relation:
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
            # Tạo nhóm mới
            if target_groups is not None:
                target_groups.append([token_id])
            last_was_coord = False
        else:
            # Thêm vào nhóm cuối
            if target_groups is not None:
                if not target_groups or not target_groups[-1]:
                     target_groups.append([token_id])
                else:
                     target_groups[-1].append(token_id)

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
from graph.src.triplet_extraction.utils import is_valid_term, clean_text
from graph.src.triplet_extraction import parsing_result

def parse_dataframe_to_tokens(df):
    """Convert DataFrame to a list of token dicts"""
    tokens = []
    for _, row in df.iterrows():
        token = {
            'id': int(row['id']),
            'word': str(row['word']),
            'pos': str(row['pos']),
            'head': int(row['head']),
            'deprel': str(row['deprel'])
        }
        tokens.append(token)
    return tokens


def split_sentence_np_vp(tokens):
    if not tokens:
        return [], []

    root_index = -1
    root_id = None

    # Find the main verb (root or first valid verb)
    for i, token in enumerate(tokens):
        if token['pos'] == 'V':
            if token['deprel'] == 'root' and token['head'] == 0:
                # Avoid picking verb at start (index 0)
                if i == 0:
                    continue
                root_index = i
                root_id = token['id']
                break
            elif root_index == -1 and token['deprel'] != 'nmod':
                # Avoid first word if it's a verb
                if i == 0:
                    continue
                root_index = i
                root_id = token['id']

    # Fallback – pick next verb if root not found
    if root_index == -1:
        for i, token in enumerate(tokens):
            if token['pos'] == 'V' and i > 0:  # skip first position
                root_index = i
                root_id = token['id']
                break

    # Final split
    if root_index != -1 and root_id is not None:
        np_tokens = tokens[:root_index]
        vp_tokens = tokens[root_index:]
        return np_tokens, vp_tokens

    return [], []

def collect_dependents(tokens, head_id):
    """Return set of token ids: head_id + all recursive dependents"""
    subtree = {head_id}
    result = []
    added = True
    while added:
        added = False
        for token in tokens:
            if token['head'] in subtree and token['id'] not in subtree:
                subtree.add(token['id'])
                result.append(token)
                added = True
    return result


def collect_direct_dependents(tokens, head_id):
    """Return list of token dicts that directly depend on head_id"""
    return [t for t in tokens if t['head'] == head_id]


def rebuild_phrase(tokens):
    """Sort tokens by their original position in the sentence and join them together"""
    tokens_sorted = sorted(tokens, key=lambda x: x['id'])
    phrase = " ".join(t['word'] for t in tokens_sorted)
    return phrase

def extract_main_subjects(np_tokens):
    if not np_tokens:
        return []

    sub_tokens = [t for t in np_tokens if t['deprel'] == 'sub']
    if not sub_tokens:
        sub_tokens = [t for t in np_tokens if t['deprel'] == 'root']
    if not sub_tokens:
        return []

    main_subjects = [sub_tokens[0]]
    main_subjects.extend(collect_direct_dependents(np_tokens, sub_tokens[0]['id']))
    if main_subjects and main_subjects[-1]['pos'] in ['Cc', 'CH']:
        main_subjects.pop()
    if len(main_subjects) == len(np_tokens):
        return [rebuild_phrase(np_tokens)]

    # Find Coordination Word (Cc, CH)
    coord_tokens = [t for t in np_tokens if (t['pos'] in ['Cc', 'CH'] and t not in main_subjects)]
    non_main_tokens = set()
    if len(coord_tokens) > 0:
        phrases = []

        for coord in coord_tokens:
            coord_index = next((i for i, t in enumerate(np_tokens) if t['id'] == coord['id']), None)

            left_tokens = []
            main_subjects_id = [obj['id'] for obj in main_subjects]
            for i in range(coord_index - 1, -1, -1):
                token = np_tokens[i]
                if token['pos'] not in ['CH', 'Cc'] and token['id'] not in main_subjects_id:
                    left_tokens.append(token)
                    non_main_tokens.add(token['id'])
                else:
                    break

            right_tokens = []
            for i in range(coord_index + 1, len(np_tokens)):
                token = np_tokens[i]
                if token['pos'] not in ['CH', 'Cc']:
                    right_tokens.append(token)
                    non_main_tokens.add(token['id'])
                else:
                    break

            if left_tokens:
                phrases.append(rebuild_phrase(left_tokens))
            if right_tokens:
                phrases.append(rebuild_phrase(right_tokens))

        # Remove duplicates while preserving order
        phrases = list(dict.fromkeys(phrases))

        for sub in main_subjects:
            if sub['id'] in non_main_tokens:
                main_subjects.remove(sub)
        main_subject_phrase = rebuild_phrase(main_subjects)

        # Properly combine main subject with each phrase
        combined_phrases = []
        for phrase in phrases:
            combined_phrases.append(main_subject_phrase + " " + phrase)

        if not combined_phrases:
            return [main_subject_phrase]

        return combined_phrases
    else:
        return [rebuild_phrase(np_tokens)]

def extract_verbs(vp_tokens):
    if not vp_tokens:
        return [], []

    # Find the root verb first
    root_verb = None
    for t in vp_tokens:
        if t['deprel'] == 'root' and t['head'] == 0 and t['pos'] == 'V':
            root_verb = t
            break

    # Fallback to the first verb that not nmod or aux
    if not root_verb:
        for t in vp_tokens:
            if t['pos'] == 'V' and t['deprel'] not in ['nmod', 'aux']:
                root_verb = t
                break

    # Fallback to any first verb in vp_tokens
    if not root_verb:
        for t in vp_tokens:
            if t['pos'] == 'V':
                root_verb = t
                break

    if not root_verb:
        return [vp_tokens[0]['word']], [vp_tokens[0]]

    # Check for coordination markers (CH, Cc) that are direct dependents of root
    coord_markers = [t for t in vp_tokens if t['pos'] in ['Cc', 'CH'] and t['head'] == root_verb['id']]

    if coord_markers:
        coordinated_verbs = [root_verb]
        for t in vp_tokens:
            if t['pos'] == 'V' and t['head'] == root_verb['id'] and t['deprel'] in ['vmod', 'conj']:
                coordinated_verbs.append(t)

        # Sort by ID to maintain order
        coordinated_verbs.sort(key=lambda x: x['id'])

        verb_phrases = []
        all_tokens = []
        coordinated_verbs_id = [v['id'] for v in coordinated_verbs]
        for verb in coordinated_verbs:
            phrase_tokens = [verb]
            dependents = collect_direct_dependents(vp_tokens, verb['id'])

            # Keep only dependents that are not other coordinated verbs or coordination markers
            dependents = [d for d in dependents if
                          d['id'] not in coordinated_verbs_id
                          and d['pos'] not in ['CH', 'Cc']
                          and d['deprel'] == 'vmod']

            phrase_tokens.extend(dependents)
            all_tokens.extend(phrase_tokens)

            verb_phrases.append({
                'text': rebuild_phrase(phrase_tokens),
                'tokens': phrase_tokens
            })

        return verb_phrases, all_tokens

    # Single verb: return it with its dependents
    verb_tokens = [root_verb]
    verb_tokens.extend(collect_direct_dependents(vp_tokens, root_verb['id']))
    sorted_verb_tokens = sorted(verb_tokens, key=lambda x: x['id'])

    # Filter out tokens after the first noun
    filtered_tokens = []
    for token in sorted_verb_tokens:
        if token['pos'].startswith('N'):
            break
        filtered_tokens.append(token)

    # If we filtered out everything, at least return the root verb
    if not filtered_tokens:
        filtered_tokens = [root_verb]

    return [{
        'text': rebuild_phrase(filtered_tokens),
        'tokens': filtered_tokens
    }], filtered_tokens

def extract_objects(vp_tokens, verb_token):
    # Remove verb tokens from vp_tokens
    for v in verb_token:
        for t in vp_tokens:
            if t['id'] == v['id']:
                vp_tokens.remove(t)
                break

    if not vp_tokens:
        return []

    # Find the first object token (dob, iob, pob)
    obj_token = next((t for t in vp_tokens if t['deprel'] in ['dob', 'iob', 'pob']), None)

    # Fallback to the first noun in vp_tokens
    if obj_token is None:
        obj_token = next((t for t in vp_tokens if t['pos'] == 'N'), None)

    if obj_token is None:
        return []

    # Collect main object and its dependents
    main_objects = [obj_token]
    main_objects.extend(collect_direct_dependents(vp_tokens, obj_token['id']))

    if len(main_objects) == len(vp_tokens):
        return [{
            'text': rebuild_phrase(vp_tokens),
            'tokens': vp_tokens
        }]

    # Find coordination tokens
    coord_tokens = [t for t in vp_tokens if t['pos'] in ['Cc', 'CH']]
    for obj in main_objects:
        if obj in coord_tokens:
            main_objects = []
            break

    if coord_tokens:
        combined_phrases = []

        for coord in coord_tokens:
            coord_index = next((i for i, t in enumerate(vp_tokens) if t['id'] == coord['id']), None)

            # LEFT TOKENS
            left_tokens = []
            for i in range(coord_index - 1, -1, -1):
                token = vp_tokens[i]
                if token['pos'] not in ['CH', 'Cc'] and token['id'] not in [obj['id'] for obj in main_objects]:
                    left_tokens.append(token)
                else:
                    break
            left_tokens = left_tokens[::-1]

            # RIGHT TOKENS
            right_tokens = []
            for i in range(coord_index + 1, len(vp_tokens)):
                token = vp_tokens[i]
                if token['pos'] not in ['CH', 'Cc']:
                    right_tokens.append(token)
                else:
                    break

            # Combine main object with left and right tokens
            for tokens_side in [left_tokens, right_tokens]:
                if tokens_side:
                    combined_phrases.append({
                        'text': rebuild_phrase(main_objects) + " " + rebuild_phrase(tokens_side),
                        'tokens': main_objects + tokens_side
                    })

        # Remove duplicates while preserving order
        seen = set()
        final_phrases = []
        for item in combined_phrases:
            if item['text'] not in seen:
                final_phrases.append(item)
                seen.add(item['text'])

        return final_phrases

    else:
        return [{
            'text': rebuild_phrase(vp_tokens),
            'tokens': vp_tokens
        }]


def process_sentence(df, logger):
    tokens = parse_dataframe_to_tokens(df)
    np_tokens, vp_tokens = split_sentence_np_vp(tokens)
    logger.debug("-----------------NP-----------------")
    logger.debug(np_tokens)
    logger.debug("-----------------VP-----------------")
    logger.debug(vp_tokens)

    # Extract subjects, verbs, objects
    subjects = extract_main_subjects(np_tokens)
    verbs, verbs_token = extract_verbs(vp_tokens)
    objects = extract_objects(vp_tokens, verbs_token)

    logger.debug("-----------------subjects----------------")
    logger.debug(subjects)
    logger.debug("-----------------verbs----------------")
    for verb in verbs:
        logger.debug(verb['text'])
    logger.debug("-----------------objects----------------")
    for obj in objects:
        logger.debug(obj['text'])

    verbs_position = {}
    for verb in verbs:
        verb_last_id = verb['tokens'][0]['id']
        verbs_position[verb['text']] = verb_last_id

    verbs_sorted = sorted(verbs, key=lambda v: v['tokens'][0]['id'])
    objects_sorted = sorted(objects, key=lambda o: o['tokens'][0]['id'])

    # Start combine them into triplets
    triplets = []
    for subj in subjects:
        for i, verb in enumerate(verbs_sorted):
            verb_last_id = verb['tokens'][-1]['id']

            # Determine the next verb's first ID (or infinity if this is the last verb)
            next_verb_first_id = verbs_sorted[i + 1]['tokens'][0]['id'] if i + 1 < len(verbs_sorted) else float('inf')

            # Objects that come after this verb but before the next verb
            obj_candidates = []
            for obj in objects_sorted:
                obj_id = obj['tokens'][-1]['id']
                if verb_last_id < obj_id < next_verb_first_id:
                    obj_candidates.append(obj)

            for obj in obj_candidates:
                triplets.append((subj, verb['text'], obj['text']))

    return triplets

def triplet_extraction(text, vncorenlp_client, phoNLP_model, stopwords, logger, max_depth=2, depth=0):
    """Recursively extract triplets from text, including nested subjects/objects"""
    if depth > max_depth or not text.strip():
        return []

    sentence = clean_text(text)
    segmented_text = vncorenlp_client.word_segment(sentence)

    # Annotate text
    annotation = phoNLP_model.annotate(text=segmented_text[0])
    df = parsing_result(annotation)

    triplets = process_sentence(df, logger)
    all_triplets = []

    for subj, verb, obj in triplets:
        # Refine subject
        subj_annotation = phoNLP_model.annotate(text=subj)
        df_subj = parsing_result(subj_annotation)
        refined_subj_triplets = process_sentence(df_subj, logger)
        if refined_subj_triplets:
            # Replace subject with first refined subject
            subj_refined = refined_subj_triplets[0][0]
        else:
            subj_refined = subj

        # Refine object
        obj_annotation = phoNLP_model.annotate(text=obj)
        df_obj = parsing_result(obj_annotation)
        refined_obj_triplets = process_sentence(df_obj, logger)
        if refined_obj_triplets:
            # Replace object with first refined object
            obj_refined = refined_obj_triplets[0][0]
        else:
            obj_refined = obj

        all_triplets.append((subj_refined, verb, obj_refined))
        all_triplets.extend(refined_subj_triplets)
        all_triplets.extend(refined_obj_triplets)

    filtered_triplets = []

    for triplet in all_triplets:
        subj, verb, obj = triplet

        # Remove stopwords
        subj_filtered = ' '.join([w for w in subj.split() if w.lower() not in stopwords]).strip()
        verb_filtered = ' '.join([w for w in verb.split() if w.lower() not in stopwords]).strip()
        obj_filtered = ' '.join([w for w in obj.split() if w.lower() not in stopwords]).strip()

        # Skip remove stopwords if any element becomes empty
        if not subj_filtered:
            subj_filtered = subj
        if not verb_filtered:
            verb_filtered = verb
        if not obj_filtered:
            obj_filtered = obj

        subj_filtered = subj_filtered.replace('_', ' ').strip().lower()
        verb_filtered = verb_filtered.replace('_', ' ').strip().lower()
        obj_filtered = obj_filtered.replace('_', ' ').strip().lower()
        filtered_triplets.append((subj_filtered, verb_filtered, obj_filtered))

    return filtered_triplets
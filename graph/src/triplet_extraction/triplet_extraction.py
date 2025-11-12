import pandas as pd


def parse_dataframe_to_tokens(df):
    """
    Convert DataFrame to list of token tuples.

    Args:
        df: DataFrame with columns [id, word, pos, head, deprel]

    Returns:
        List of tuples: [(id, word, pos, head, deprel), ...]
    """
    tokens = []
    for _, row in df.iterrows():
        token_id = int(row['id'])
        word = str(row['word'])
        pos = str(row['pos'])
        head = int(row['head'])
        deprel = str(row['deprel'])
        tokens.append((token_id, word, pos, head, deprel))

    return tokens


def split_sentence_np_vp(tokens):
    """
    Split Vietnamese sentence into NP (Noun Phrase) and VP (Verb Phrase) using head and deprel.

    Args:
        tokens: List of tuples [(id, word, pos, head, deprel), ...]

    Returns:
        tuple: (np_tokens, vp_tokens) where each is a list of (id, word, pos, head, deprel)
    """
    if not tokens:
        return [], []

    # Find the root verb (head = 0 and deprel = 'root')
    root_index = -1
    root_id = None

    for i, (token_id, word, pos, head, deprel) in enumerate(tokens):
        if deprel == 'root' and head == 0:
            root_index = i
            root_id = token_id
            break

    if root_index == -1:
        # Fallback: find first verb
        for i, (token_id, word, pos, head, deprel) in enumerate(tokens):
            if pos == 'V':
                root_index = i
                root_id = token_id
                break

    if root_index != -1 and root_id is not None:
        np_tokens = []
        vp_tokens = []

        # Build dependency tree to find all children
        def get_all_descendants(node_id, tokens):
            """Get all tokens that depend on node_id (recursively)"""
            descendants = []
            for token in tokens:
                if token[3] == node_id:  # head == node_id
                    descendants.append(token)
                    descendants.extend(get_all_descendants(token[0], tokens))
            return descendants

        for i, token in enumerate(tokens):
            token_id, word, pos, head, deprel = token

            # NP: tokens before root that don't depend on root
            # Typically subject and its modifiers
            if i < root_index:
                # Check if this token or its head eventually leads to root
                current_head = head
                leads_to_root = False
                visited = set()

                while current_head != 0 and current_head not in visited:
                    visited.add(current_head)
                    if current_head == root_id:
                        leads_to_root = True
                        break
                    # Find the head's head
                    for t in tokens:
                        if t[0] == current_head:
                            current_head = t[3]
                            break
                    else:
                        break

                # If it doesn't directly depend on root verb, it's likely part of NP
                if not leads_to_root or deprel in ['sub', 'nsubj']:
                    np_tokens.append(token)
                else:
                    vp_tokens.append(token)
            else:
                # VP: root and everything at or after root
                vp_tokens.append(token)

        return np_tokens, vp_tokens

    # Fallback: split at midpoint
    mid = len(tokens) // 2
    return tokens[:mid], tokens[mid:]


def extract_main_subject(np_tokens):
    """
    Extract main subject from Vietnamese Noun Phrase using POS and deprel tags.

    Args:
        np_tokens: List of tuples [(id, word, pos, head, deprel), ...]

    Returns:
        str: Main subject of the sentence (or None if not found)
    """
    if not np_tokens:
        return None

    # Priority 1: Find token with deprel 'sub' or 'nsubj' (subject)
    for token_id, word, pos, head, deprel in np_tokens:
        if deprel in ['sub', 'nsubj', 'nsubj:pass']:
            return word

    # Priority 2: Find the head noun (noun that other nouns modify)
    noun_heads = set()
    for token_id, word, pos, head, deprel in np_tokens:
        if pos == 'N' and deprel in ['nmod', 'compound']:
            noun_heads.add(head)

    for token_id, word, pos, head, deprel in np_tokens:
        if token_id in noun_heads and pos == 'N':
            return word

    # Priority 3: Find the first noun
    for token_id, word, pos, head, deprel in np_tokens:
        if pos == 'N':
            return word

    # Priority 4: Look for pronouns
    for token_id, word, pos, head, deprel in np_tokens:
        if pos == 'P':
            return word

    # Fallback: return first token
    return np_tokens[0][1] if np_tokens else None


def extract_main_verb(vp_tokens):
    """
    Extract main verb from Vietnamese Verb Phrase using POS and deprel tags.

    Args:
        vp_tokens: List of tuples [(id, word, pos, head, deprel), ...]

    Returns:
        str: Main verb of the sentence (or None if not found)
    """
    if not vp_tokens:
        return None

    # Priority 1: Find the root verb (head = 0, deprel = 'root')
    for token_id, word, pos, head, deprel in vp_tokens:
        if deprel == 'root' and head == 0 and pos == 'V':
            return word

    # Priority 2: Find any root
    for token_id, word, pos, head, deprel in vp_tokens:
        if deprel == 'root' and head == 0:
            return word

    # Priority 3: Find the first verb that is not a modifier
    for token_id, word, pos, head, deprel in vp_tokens:
        if pos == 'V' and deprel not in ['vmod', 'aux']:
            return word

    # Priority 4: Find any verb
    for token_id, word, pos, head, deprel in vp_tokens:
        if pos == 'V':
            return word

    # Fallback: return first token
    return vp_tokens[0][1] if vp_tokens else None


def extract_object(vp_tokens):
    """
    Extract object from Vietnamese Verb Phrase using POS and deprel tags.

    Args:
        vp_tokens: List of tuples [(id, word, pos, head, deprel), ...]

    Returns:
        str: Object of the sentence (or None if not found)
    """
    if not vp_tokens:
        return None

    # Find the root verb first
    root_id = None
    for token_id, word, pos, head, deprel in vp_tokens:
        if deprel == 'root' and head == 0:
            root_id = token_id
            break

    if root_id is None and vp_tokens:
        root_id = vp_tokens[0][0]

    # Priority 1: Find direct object (dobj, obj) that depends on root
    for token_id, word, pos, head, deprel in vp_tokens:
        if head == root_id and deprel in ['dobj', 'obj']:
            return word

    # Priority 2: Find noun in prepositional phrase (pob - prepositional object)
    # that is close to the root
    for token_id, word, pos, head, deprel in vp_tokens:
        if deprel == 'pob' and pos == 'N':
            # Check if this pob's head is close to root
            for t_id, t_word, t_pos, t_head, t_deprel in vp_tokens:
                if t_id == head and t_head == root_id:
                    return word

    # Priority 3: Find verb modifier (vmod) that is a noun directly under root
    for token_id, word, pos, head, deprel in vp_tokens:
        if head == root_id and deprel == 'vmod' and pos == 'N':
            return word

    # Priority 4: Find any noun that directly depends on the root verb
    for token_id, word, pos, head, deprel in vp_tokens:
        if head == root_id and pos == 'N':
            return word

    # Priority 5: Find any prepositional object (pob) that is a noun
    for token_id, word, pos, head, deprel in vp_tokens:
        if deprel == 'pob' and pos == 'N':
            return word

    # Priority 6: Find any noun in VP
    for token_id, word, pos, head, deprel in vp_tokens:
        if pos == 'N':
            return word

    return None


def process_sentence(df):
    # Convert DataFrame to token list
    tokens = parse_dataframe_to_tokens(df)

    # Split into NP and VP
    np_tokens, vp_tokens = split_sentence_np_vp(tokens)

    # Extract components
    subject = extract_main_subject(np_tokens)
    verb = extract_main_verb(vp_tokens)
    obj = extract_object(vp_tokens)

    result = {
        'subject': subject,
        'verb': verb,
        'object': obj
    }

    return result
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
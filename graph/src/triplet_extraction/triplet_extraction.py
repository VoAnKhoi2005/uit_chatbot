from . import process_sentence
from ..db import insert_triplet


def extract_triplet_and_store(input_text: str, rdrsegmenter, driver, db_name, document_id, document_number):
    with driver.session(database=db_name) as session:
        split_text = input_text.split("\n")
        for t in split_text:
            res = process_sentence(t, rdrsegmenter, verbose=False)
            for concept1, relation, concept2 in res["concepts"]:
                session.execute_write(
                    insert_triplet,
                    concept1,
                    relation,
                    concept2,
                    c1_metadata={"document_number": document_number, "document_id": document_id},
                    c2_metadata={"document_number": document_number, "document_id": document_id},
                    rel_metadata={"document_number": document_number, "document_id": document_id},
                )

def extract_triplet(input_text: str, rdrsegmenter, verbose=False):
    result = []
    split_text = input_text.split("\n")
    for t in split_text:
        if not t.strip():
            continue
        res = process_sentence(t, rdrsegmenter, verbose=verbose)
        result.append(res["concepts"])
    return result
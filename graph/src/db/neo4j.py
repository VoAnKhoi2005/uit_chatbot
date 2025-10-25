from neo4j import GraphDatabase

def init_neo4j(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    return driver

def insert_triplet(tx, concept1, relation, concept2, c1_metadata=None, c2_metadata=None, rel_metadata=None):
    query = """
    MERGE (c1:Concept {name: $concept1})
    ON CREATE SET c1 = $c1_metadata
    ON MATCH SET
        c1.document_number = apoc.coll.toSet(coalesce(c1.document_number, []) + [$c1_metadata.document_number]),
        c1.document_id     = apoc.coll.toSet(coalesce(c1.document_id, []) + [$c1_metadata.document_id])

    MERGE (c2:Concept {name: $concept2})
    ON CREATE SET c2 = $c2_metadata
    ON MATCH SET
        c2.document_number = apoc.coll.toSet(coalesce(c2.document_number, []) + [$c2_metadata.document_number]),
        c2.document_id     = apoc.coll.toSet(coalesce(c2.document_id, []) + [$c2_metadata.document_id])

    MERGE (c1)-[r:RELATION {name: $relation}]->(c2)
    ON CREATE SET r = $rel_metadata
    ON MATCH SET  r += $rel_metadata

    RETURN c1, r, c2
    """
    tx.run(
        query,
        concept1=concept1,
        concept2=concept2,
        relation=relation,
        c1_metadata=c1_metadata or {},
        c2_metadata=c2_metadata or {},
        rel_metadata=rel_metadata or {}
    )

def insert_triplet_batch(tx, triplets_list, metadata):
    query = """
    UNWIND $triplets as triplet
    WITH triplet WHERE triplet.c1 IS NOT NULL AND triplet.r IS NOT NULL AND triplet.c2 IS NOT NULL

    MERGE (c1:Concept {name: triplet.c1})
    ON CREATE SET 
        c1.name = triplet.c1,
        c1.document_id = [$metadata.document_id],
        c1.document_number = [$metadata.document_number]
    ON MATCH SET
        c1.document_number = apoc.coll.toSet(coalesce(c1.document_number, []) + [$metadata.document_number]),
        c1.document_id     = apoc.coll.toSet(coalesce(c1.document_id, []) + [$metadata.document_id])

    MERGE (c2:Concept {name: triplet.c2})
    ON CREATE SET 
        c2.name = triplet.c2,
        c2.document_id = [$metadata.document_id],
        c2.document_number = [$metadata.document_number]
    ON MATCH SET
        c2.document_number = apoc.coll.toSet(coalesce(c2.document_number, []) + [$metadata.document_number]),
        c2.document_id     = apoc.coll.toSet(coalesce(c2.document_id, []) + [$metadata.document_id])

    MERGE (c1)-[r:RELATION {name: triplet.r}]->(c2)
    ON CREATE SET r.source_document_ids = [$metadata.document_id]
    ON MATCH SET
        r.source_document_ids = apoc.coll.toSet(coalesce(r.source_document_ids, []) + [$metadata.document_id])
    """
    tx.run(query, triplets=triplets_list, metadata=metadata)

def delete_all(session):
    session.run("MATCH (n) DETACH DELETE n")

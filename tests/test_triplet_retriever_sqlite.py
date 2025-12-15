def test_triplet_retriever_sqlite_import():
    from retrieval.src.retrieval.triplet_retriever import TripletRetriever
    retriever = TripletRetriever()
    assert hasattr(retriever.vector_db, 'get_doc_ids_for_concept')
    assert hasattr(retriever.vector_db, 'get_doc_ids_for_relation')

def test_import_triplet_retriever():
    from backend.retrieval.src.retrieval.triplet_retriever import TripletRetriever
    retriever = TripletRetriever()
    assert retriever.vector_db is not None

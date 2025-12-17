import argparse
import sys
from retrieval.src.ingestion.change_detector import ChangeDetector
from retrieval.src.registry.metadata_registry import MetadataRegistry
from retrieval.text_rag.chunker import iter_all_chunks
from retrieval.text_rag.vector_store import ChunkVectorStore
from retrieval.src.retrieval.triplet_retriever import TripletRetriever
from pathlib import Path
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs_dir', type=str, default='data/docs')
    parser.add_argument('--ttl_path', type=str, default='ontology/uit_regulations.ttl')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--registry_db', type=str, default='metadata_registry.db')
    parser.add_argument('--change_db', type=str, default='change_state.db')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    detector = ChangeDetector(args.change_db)
    registry = MetadataRegistry(args.registry_db)

    # 1. Scan docs
    doc_files = [str(p) for p in Path(args.docs_dir).rglob("*.txt")]
    doc_changes = detector.scan([args.docs_dir], group="docs")
    ttl_changes = detector.scan([args.ttl_path], group="ttl")
    logging.info(f"Doc changes: {len(doc_changes)} | TTL changes: {len(ttl_changes)}")

    # 2. Re-chunk/index text if needed
    if doc_changes or args.force:
        logging.info("Re-chunking and indexing text docs...")
        vector_store = ChunkVectorStore()
        chunk_count = 0
        for doc_path in doc_files:
            for chunk in iter_all_chunks(doc_path):
                vector_store.index_chunks([chunk])
                registry.upsert_chunk({
                    **chunk,
                    "source_path": doc_path
                })
                chunk_count += 1
        logging.info(f"Indexed {chunk_count} text chunks.")

    # 3. Re-extract/index triples if needed
    if ttl_changes or args.force:
        logging.info("Re-extracting and indexing triples from TTL...")
        triplet_retriever = TripletRetriever()
        # Giả định TripletRetriever có hàm extract_triples_from_ttl
        triples = triplet_retriever.extract_triples_from_ttl(args.ttl_path)
        triple_count = 0
        for triple in triples:
            triple_id = MetadataRegistry.make_triple_id(triple['subject'], triple['predicate'], triple['object'], triple.get('article_id'), triple.get('ttl_uri'))
            registry.upsert_triple({
                **triple,
                "triple_id": triple_id,
                "source_path": args.ttl_path
            })
            triple_count += 1
        logging.info(f"Indexed {triple_count} triples.")

    # 4. Commit scan state
    detector.commit_scan([args.docs_dir], group="docs")
    detector.commit_scan([args.ttl_path], group="ttl")
    logging.info("Ingestion complete.")

if __name__ == "__main__":
    main()

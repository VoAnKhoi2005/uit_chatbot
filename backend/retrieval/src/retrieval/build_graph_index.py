"""Pre-warm the knowledge-graph embedding cache (node names + triples) so a
later app start doesn't need to call the embedder at all for unchanged data.

Mirrors ``retrieval.text_rag.build_index`` (the equivalent offline step for
the text chunk index): run this once after the exported triplets change, and
GraphRetriever's construction on every subsequent process start becomes a
pure cache read, with zero embedding calls, until the triplets change again.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from retrieval.text_rag.embeddings import TextEmbedder
from retrieval.src.retrieval.graph_retriever import GraphRetriever, DEFAULT_TRIPLETS_PATH


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    triplets_path = Path(os.getenv("UIT_TRIPLETS_PATH", str(DEFAULT_TRIPLETS_PATH)))

    embedder = TextEmbedder()
    retriever = GraphRetriever(triplets_path=triplets_path, embedder=embedder)

    logging.info(
        "Warmed graph embedding cache: %d nodes, %d triples from %s",
        len(retriever.nodes),
        len(retriever.triples),
        triplets_path,
    )


if __name__ == "__main__":
    main()

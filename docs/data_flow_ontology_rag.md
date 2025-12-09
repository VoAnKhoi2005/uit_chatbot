# Data flow: graph vs. content sources

Rule enforced:
- Ontology (RDF/OWL) is built primarily from the graph export `graph/mongo_export_uit/KB_UIT.triplets.json`.
- RAG text index is built from the content/hierarchy export `graph/mongo_export_uit/KB_UIT.items.json`.

## Ontology (ontology/from_jsonl.py)
- Primary triples come from `KB_UIT.triplets.json`:
  - Nodes: subject/object URIs are derived from `subject_id` / `object_id` (Mongo `$oid` or strings) as `uit:Entity_<id>`.
  - Predicates: derived from `relation_name` (mapped via `RELATION_PROPERTY_MAP`, else `uit:relatedTo`).
  - Relation triples are written directly from the triplet data.
- Items file `KB_UIT.items.json` is optional and only used to enrich nodes:
  - Adds labels/text (`uit:title`, `uit:heading`, `uit:fullText`, etc.) and doc linkage (`uit:docId`).
  - Adds hierarchy edges (`uit:hasParent`, `uit:hasClause`, `uit:hasArticle`) based on `parent_id` and `level`.
  - Creates document URIs (`uit:Document_<doc_id>`) and attaches article/clause ids (`uit:articleId`, `uit:clauseId`).
- Output: serialized to `ontology/uit_regulations.ttl`.

## RAG (retrieval/text_rag)
- Loader `iter_raw_docs` reads only `KB_UIT.items.json`:
  - Extracts article/clause `_id` as `article_id` / `clause_id`, with titles/headings/content.
  - These IDs align with ontology URIs (`uit:Article_<id>`, `uit:Clause_<id>`) for cross-linking.
- Chunker splits each doc’s `content` into ≤800-char chunks with deterministic `chunk_id`.
- Embedding + vector store index built from these chunks (no dependency on triplets) via `build_index.py`.

## ID alignment
- RAG uses `_id` from items as text-unit identifiers.
- Ontology:
  - Entities from triplets use `subject_id/object_id` → `uit:Entity_<id>`.
  - Articles/clauses from items use `_id` → `uit:Article_<id>` / `uit:Clause_<id>`.
  - Document linkage uses `doc_id` to `uit:Document_<doc_id>`.


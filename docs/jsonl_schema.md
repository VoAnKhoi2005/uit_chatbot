# UIT regulation data schemas

Summary of the two JSON sources under `graph/mongo_export_uit` that act as
graph and content inputs.

## `KB_UIT.triplets.json` (graph)

- Structure: JSON array, each element is a relation triple.
- Fields observed:
  - `_id`: MongoDB object id (stored as `{"$oid": ...}`).
  - `subject_id`, `relation_id`, `object_id`: MongoDB ids for the triple
    endpoints and the relation (also in `{"$oid": ...}` form).
  - `subject_name`: surface string of the subject concept/entity.
  - `relation_name`: surface string of the relation/predicate.
  - `object_name`: surface string of the object concept/entity.
  - `document_id`: UUID of the source regulation document (matches `_id`
    entries in `KB_UIT.items.json` for the root document node).
  - `document_number`: optional document number; `null` in sampled rows.

## `KB_UIT.items.json` (content + hierarchy)

- Structure: JSON array, each element is a hierarchical unit of a regulation
  document.
- Fields observed:
  - `_id`: UUID for the unit (used as stable article/clause ids).
  - `doc_id`: UUID for the regulation document this unit belongs to.
  - `parent_id`: UUID of the parent unit (or `null` for top level).
  - `level`: hierarchy label such as `chuong` (chapter), `muc` (section),
    `dieu` (article), `khoan` (clause).
  - `title`: heading label (e.g., “Điều 33”).
  - `heading`: short heading text; often mirrors `title`.
  - `content`: full textual content for the unit (article or clause text when
    present).
  - `ordinal`: integer ordering key used in the original document.
  - `path`: slash-delimited hierarchy path within the document.
  - `created_at`, `updated_at`: timestamps from the source export.

These field names are used directly in the ontology and RAG loaders—no
assumed/guessed names are hardcoded beyond what appears in the files.


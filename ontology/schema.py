"""
Ontology schema/vocabulary for UIT regulations.

Fields map to the exported JSON:
- Triplets file (`KB_UIT.triplets.json`): subject_id/object_id -> UIT.Entity,
  subject_name/object_name -> UIT.label literals, relation_name -> predicates.
- Items file (`KB_UIT.items.json`): `_id` -> UIT.articleId / UIT.clauseId,
  `level` drives class selection (Article, Clause, Chapter, Section),
  `doc_id` -> UIT.docId, `content` -> UIT.fullText, `title`/`heading` -> UIT.title.
"""

from rdflib import Namespace, URIRef

UIT = Namespace("http://uit.vn/regulations#")

# Core classes.
Regulation: URIRef = UIT.Regulation
Document: URIRef = UIT.Document
Chapter: URIRef = UIT.Chapter
Section: URIRef = UIT.Section
Article: URIRef = UIT.Article
Clause: URIRef = UIT.Clause
Action: URIRef = UIT.Action
Condition: URIRef = UIT.Condition
Sanction: URIRef = UIT.Sanction
Program: URIRef = UIT.Program
Entity: URIRef = UIT.Entity  # generic fallback when no better class exists

# Object properties.
hasArticle: URIRef = UIT.hasArticle
hasClause: URIRef = UIT.hasClause
hasParent: URIRef = UIT.hasParent
appliesToProgram: URIRef = UIT.appliesToProgram
hasCondition: URIRef = UIT.hasCondition
hasSanction: URIRef = UIT.hasSanction
hasAction: URIRef = UIT.hasAction
refersToArticle: URIRef = UIT.refersToArticle
relatedTo: URIRef = UIT.relatedTo  # generic relation when no mapping exists
inDocument: URIRef = UIT.inDocument

# Datatype properties.
articleId: URIRef = UIT.articleId
clauseId: URIRef = UIT.clauseId
docId: URIRef = UIT.docId
docTitle: URIRef = UIT.docTitle
soHieu: URIRef = UIT.soHieu
title: URIRef = UIT.title
heading: URIRef = UIT.heading
fullText: URIRef = UIT.fullText
level: URIRef = UIT.level
path: URIRef = UIT.path
ordinal: URIRef = UIT.ordinal
relationName: URIRef = UIT.relationName
subjectName: URIRef = UIT.subjectName
objectName: URIRef = UIT.objectName

# Common relation-name → property mappings inferred from the triplet export.
# Keys should match `relation_name` values in `KB_UIT.triplets.json`.
RELATION_PROPERTY_MAP: dict[str, URIRef] = {
    "có": hasCondition,
    "kể từ": relatedTo,
    "giảng dạy": hasAction,
    "bao gồm": relatedTo,
    "liên quan": relatedTo,
}


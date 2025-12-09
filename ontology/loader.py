"""Helpers to load the UIT ontology and run common SPARQL queries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from rdflib import Graph

import ontology.schema as SC


def load_ontology(path: str | Path = "ontology/uit_regulations.ttl") -> Graph:
    """Load the serialized TTL graph."""
    graph = Graph()
    graph.parse(str(path), format="turtle")
    return graph


def run_sparql(graph: Graph, query: str) -> List[Dict[str, str]]:
    """Run a SPARQL query and normalize bindings to plain strings."""
    results: list[dict[str, str]] = []
    for row in graph.query(query):
        row_dict: dict[str, str] = {}
        for var, val in row.asdict().items():
            row_dict[str(var)] = str(val)
        results.append(row_dict)
    return results


def get_article_by_id(graph: Graph, article_id: str) -> List[Dict[str, str]]:
    query = f"""
    PREFIX uit: <{SC.UIT}>
    SELECT ?article ?title ?text
    WHERE {{
        ?article a uit:Article ;
                 uit:articleId "{article_id}" .
        OPTIONAL {{ ?article uit:title ?title. }}
        OPTIONAL {{ ?article uit:fullText ?text. }}
    }}
    """
    return run_sparql(graph, query)


def get_clauses_for_article(graph: Graph, article_id: str) -> List[Dict[str, str]]:
    query = f"""
    PREFIX uit: <{SC.UIT}>
    SELECT ?clause ?title ?text
    WHERE {{
        ?article a uit:Article ;
                 uit:articleId "{article_id}" ;
                 (uit:hasClause|^uit:hasParent)+ ?clause .
        ?clause a uit:Clause .
        OPTIONAL {{ ?clause uit:title ?title. }}
        OPTIONAL {{ ?clause uit:fullText ?text. }}
    }}
    """
    return run_sparql(graph, query)


def get_conditions_for_action(graph: Graph, action_keyword: str) -> List[Dict[str, str]]:
    """Find clauses whose text mentions a keyword; works even if ontology is sparse."""
    keyword = action_keyword.lower()
    query = f"""
    PREFIX uit: <{SC.UIT}>
    SELECT ?clause ?text
    WHERE {{
        ?clause a uit:Clause ;
                uit:fullText ?text .
        FILTER (CONTAINS(LCASE(STR(?text)), "{keyword}"))
    }}
    LIMIT 20
    """
    return run_sparql(graph, query)


from rdflib import Graph, Literal, RDF

import ontology.schema as SC
from ontology.loader import get_conditions_for_action


def test_keyword_query_finds_clause() -> None:
    graph = Graph()
    clause = SC.UIT["Clause_Test"]
    graph.add((clause, RDF.type, SC.Clause))
    graph.add((clause, SC.fullText, Literal("Sinh viên phải nộp học phí đúng hạn.")))

    rows = get_conditions_for_action(graph, "học phí")
    assert rows


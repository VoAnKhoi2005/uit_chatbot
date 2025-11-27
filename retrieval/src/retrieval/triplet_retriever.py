from graph.src.db import *
import sys
from retrieval.src.db.vector_db import *
from collections import defaultdict


class TripletRetriever():
    def __init__(self, db_name: str = "KB_UIT", concepts_col_name: str = "concepts",
                 relations_col_name: str = "relations"):
        self.mongo_client = init_mongo()
        if not self.mongo_client:
            print("Failed to connect to MongoDB. Exiting.")
            sys.exit(1)

        self.mongo_db = self.mongo_client[db_name]
        self.concepts_collection = self.mongo_db[concepts_col_name]
        self.relations_collection = self.mongo_db[relations_col_name]
        self.vector_db = ConceptRelationDB()

    def _search_document_ids(self, text: str, collection, is_concept: bool = True) -> List[str]:
        if is_concept:
            search_results = self.vector_db.search_concepts(text, top_k=1)
        else:
            search_results = self.vector_db.search_relations(text, top_k=1)

        if not search_results:
            return []
        r = search_results[0]

        item = collection.find_one({"_id": to_object_id(r["parent_id"])})

        if not item or "documents" not in item:
            return []

        ids = [doc["document_id"] for doc in item["documents"]]
        return ids

    def search_concepts_document_ids(self, text: str) -> List[str]:
        return self._search_document_ids(text, self.concepts_collection, is_concept=True)

    def search_relations_document_ids(self, text: str) -> List[str]:
        return self._search_document_ids(text, self.relations_collection, is_concept=False)

    def search_triplet(self, triplet: Dict[str, str]) -> List[Dict[str, Any]]:

        c1 = triplet["c1"]
        c2 = triplet["c2"]
        r = triplet["r"]

        ds1 = self.search_concepts_document_ids(c1)
        ds2 = self.search_concepts_document_ids(c2)
        ds3 = self.search_relations_document_ids(r)

        set1 = set(ds1)
        set2 = set(ds2)
        set3 = set(ds3)

        # --- 1. Score 3 (Ưu tiên cao nhất): Giao của cả 3 ---
        priority_1_ids_set = set1.intersection(set2).intersection(set3)
        priority_1_ids = list(priority_1_ids_set)

        # --- 2. Score 2 (Ưu tiên trung bình): Giao từng cặp 2 (Loại trừ Score 3) ---
        pair_c1_c2 = set1.intersection(set2)
        pair_c1_r = set1.intersection(set3)
        pair_c2_r = set2.intersection(set3)
        priority_2_ids_raw = pair_c1_c2.union(pair_c1_r).union(pair_c2_r)
        priority_2_ids_set = priority_2_ids_raw.difference(priority_1_ids_set)
        priority_2_ids = list(priority_2_ids_set)

        # --- 3. Score 1 (Ưu tiên thấp): Gộp đơn lẻ (Loại trừ Score 3 và 2) ---
        all_found_ids = set1.union(set2).union(set3)
        classified_ids = priority_1_ids_set.union(priority_2_ids_set)
        priority_3_ids_set = all_found_ids.difference(classified_ids)
        priority_3_ids = list(priority_3_ids_set)

        # --- Kết quả trả về ---
        results = [
            {"score": 3, "ids": priority_1_ids},
            {"score": 2, "ids": priority_2_ids},
            {"score": 1, "ids": priority_3_ids}
        ]
        return results

    def search_triplets(self, triplets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Thực hiện tìm kiếm cho một danh sách các triplets và tổng hợp điểm số.
        Kết quả được sắp xếp theo tổng điểm giảm dần.
        """
        doc_priority_scores = defaultdict(list)

        # 1. Thực hiện tìm kiếm và tích lũy điểm cho từng triplet
        for triplet in triplets:
            triplet_results = self.search_triplet(triplet)

            for result in triplet_results:
                score = result["score"]
                ids = result["ids"]

                for doc_id in ids:
                    doc_priority_scores[doc_id].append(score)

        # 2. Tính tổng điểm và tạo danh sách kết quả cuối cùng
        final_results = []
        for doc_id, scores in doc_priority_scores.items():
            total_score = sum(scores)

            final_results.append({
                "doc_id": doc_id,
                "total_score": total_score,
            })

        # 3. Sắp xếp kết quả: Điểm tổng (total_score) giảm dần
        final_results.sort(key=lambda x: x["total_score"], reverse=True)

        return final_results

    def get_document(self, doc_id) -> str:
        return f"Hiện đang chưa có DB document trong Mongo (ID: {doc_id})"
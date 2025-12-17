import sys

from graph.src.db import init_mongo
from vector_db import *


def main():
    # Kết nối MongoDB
    mongo_client = init_mongo("")
    if not mongo_client:
        print("❌ Failed to connect to MongoDB. Exiting.")
        sys.exit(1)

    db = mongo_client["KB_UIT"]
    vector_db = ConceptRelationDB()
    concepts_collection = db["concepts"]
    relations_collection = db["relations"]

    # ==================== CONCEPTS ====================
    print("=" * 50)
    print("PROCESSING CONCEPTS")
    print("=" * 50)

    all_concepts = list(concepts_collection.find())
    print(f"📊 Tổng số concepts: {len(all_concepts)}")

    concept_items = []
    for doc in all_concepts:
        # Thêm concept chính
        concept_items.append({
            "name": doc["name"],
            "_id": str(doc["_id"]),
        })

        # Thêm synonyms
        if "synonym" in doc and doc["synonym"]:  # ✓ Kiểm tra trước
            for synonym in doc["synonym"]:
                concept_items.append({
                    "name": synonym,
                    "_id": str(doc["_id"]),
                })

    print(f"📝 Tổng items (concept + synonyms): {len(concept_items)}")
    print(f"   Ví dụ: {concept_items[0]}")

    # Thêm vào database
    print("\n⏳ Đang thêm concepts...")
    success = 0
    failed = 0
    for item in concept_items:
        result = vector_db.add_concept(item["name"], item["_id"])
        if result:
            success += 1
        else:
            failed += 1

    print(f"✓ Thêm thành công: {success}")
    print(f"❌ Lỗi: {failed}")

    # ==================== RELATIONS ====================
    print("\n" + "=" * 50)
    print("PROCESSING RELATIONS")
    print("=" * 50)

    all_relations = list(relations_collection.find())
    print(f"📊 Tổng số relations: {len(all_relations)}")

    relation_items = []
    for doc in all_relations:
        # Thêm relation chính
        relation_items.append({
            "name": doc["name"],
            "_id": str(doc["_id"]),
        })

        # Thêm synonyms
        if "synonym" in doc and doc["synonym"]:  # ✓ Kiểm tra trước
            for synonym in doc["synonym"]:
                relation_items.append({
                    "name": synonym,
                    "_id": str(doc["_id"]),
                })

    print(f"📝 Tổng items (relation + synonyms): {len(relation_items)}")
    print(f"   Ví dụ: {relation_items[0]}")

    # Thêm vào database
    print("\n⏳ Đang thêm relations...")
    success = 0
    failed = 0
    for item in relation_items:
        result = vector_db.add_relation(item["name"], item["_id"])  # ✓ add_relation
        if result:
            success += 1
        else:
            failed += 1

    print(f"✓ Thêm thành công: {success}")
    print(f"❌ Lỗi: {failed}")

    # ==================== STATS ====================
    print("\n" + "=" * 50)
    print("FINAL STATS")
    print("=" * 50)
    stats = vector_db.get_stats()
    print(f"📊 Concepts: {stats['concepts_count']}")
    print(f"📊 Relations: {stats['relations_count']}")

    # ==================== CLOSE ====================
    vector_db.close()  # ✓ Đóng SQLite
    mongo_client.close()  # ✓ Đóng MongoDB
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
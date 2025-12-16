from __future__ import annotations

import os
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer, util

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "vector.db"
db_path: str = os.getenv("VECTOR_DB_PATH", str(DEFAULT_DB_PATH))


def to_object_id(value) -> str:
    return "" if value is None else str(value)


def to_string_id(value) -> str:
    return "" if value is None else str(value)

class ConceptRelationDB:
    def __init__(self, db_path=db_path, model_name='keepitreal/vietnamese-sbert'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.model = SentenceTransformer(model_name)
        self.setup()
        self._setup_doc_tables()

    def _setup_doc_tables(self):
        # Table for mapping concepts to doc_ids
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS concept_docs (
                concept_id TEXT,
                doc_id TEXT
            )
        ''')
        # Table for mapping relations to doc_ids
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS relation_docs (
                relation_id TEXT,
                doc_id TEXT
            )
        ''')
        self.conn.commit()

    def get_doc_ids_for_concept(self, parent_id: str) -> list:
        self.cursor.execute('SELECT doc_id FROM concept_docs WHERE concept_id = ?', (parent_id,))
        return [row[0] for row in self.cursor.fetchall()]

    def get_doc_ids_for_relation(self, parent_id: str) -> list:
        self.cursor.execute('SELECT doc_id FROM relation_docs WHERE relation_id = ?', (parent_id,))
        return [row[0] for row in self.cursor.fetchall()]

    def setup(self):
        """Tạo hai bảng: concepts và relations"""

        # Bảng Concepts (Khái niệm)
        self.cursor.execute('''
                            CREATE TABLE IF NOT EXISTS concepts
                            (
                                id
                                INTEGER
                                PRIMARY
                                KEY
                                AUTOINCREMENT,
                                name
                                TEXT
                                NOT
                                NULL
                                UNIQUE,
                                parent_id
                                TEXT
                                NOT
                                NULL,
                                vector
                                BLOB
                                NOT
                                NULL,
                                created_at
                                TIMESTAMP
                                DEFAULT
                                CURRENT_TIMESTAMP
                            )
                            ''')

        # Bảng Relations (Quan hệ)
        self.cursor.execute('''
                            CREATE TABLE IF NOT EXISTS relations
                            (
                                id
                                INTEGER
                                PRIMARY
                                KEY
                                AUTOINCREMENT,
                                name
                                TEXT
                                NOT
                                NULL
                                UNIQUE,
                                parent_id
                                TEXT
                                NOT
                                NULL,
                                vector
                                BLOB
                                NOT
                                NULL,
                                created_at
                                TIMESTAMP
                                DEFAULT
                                CURRENT_TIMESTAMP
                            )
                            ''')

        self.conn.commit()
        print("✓ Database setup thành công!")

    # ==================== CONCEPTS ====================

    def add_concept(self, name: str, parent_id: str) -> int | None:
        """Thêm concept mới"""
        try:
            vector = self.model.encode(name)
            vector_bytes = vector.astype(np.float32).tobytes()


            self.cursor.execute('''
                                INSERT INTO concepts (name, parent_id, vector)
                                VALUES (?, ?, ?)
                                ''', (name, parent_id, vector_bytes))

            self.conn.commit()
            concept_id = self.cursor.lastrowid
            print(f"✓ Thêm concept: {name} (ID: {concept_id})")
            return concept_id

        except sqlite3.IntegrityError:
            print(f"❌ Concept '{name}' đã tồn tại!")

    def get_concept(self, concept_id: int) -> dict | None:
        """Lấy concept theo ID"""
        self.cursor.execute('''
                            SELECT id, name, parent_id, vector
                            FROM concepts
                            WHERE id = ?
                            ''', (concept_id,))

        row = self.cursor.fetchone()
        if row:
            return {
                'id': row[0],
                'name': row[1],
                'parent_id': row[2],
                'vector': np.frombuffer(row[3], dtype=np.float32)
            }
        return None

    def get_all_concepts(self) -> list[dict]:
        """Lấy tất cả concepts"""
        self.cursor.execute('SELECT id, name, parent_id, vector FROM concepts')
        rows = self.cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'name': row[1],
                'parent_id': row[2],
                'vector': np.frombuffer(row[3], dtype=np.float32)
            })
        return results

    def get_concepts_by_parent(self, parent_id: str) -> list[dict]:
        """Lấy concepts theo parent_id"""
        self.cursor.execute('''
                            SELECT id, name, parent_id, vector
                            FROM concepts
                            WHERE parent_id = ?
                            ''', (parent_id,))

        rows = self.cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'name': row[1],
                'parent_id': row[2],
                'vector': np.frombuffer(row[3], dtype=np.float32)
            })
        return results

    def update_concept(self, concept_id: int, name: str = None, parent_id: str = None) -> bool:
        """Cập nhật concept"""
        try:
            if name:
                vector = self.model.encode(name)
                vector_bytes = vector.astype(np.float32).tobytes()
                self.cursor.execute('''
                                    UPDATE concepts
                                    SET name   = ?,
                                        vector = ?
                                    WHERE id = ?
                                    ''', (name, vector_bytes, concept_id))

            if parent_id:
                self.cursor.execute('''
                                    UPDATE concepts
                                    SET parent_id = ?
                                    WHERE id = ?
                                    ''', (parent_id, concept_id))

            self.conn.commit()
            print(f"✓ Cập nhật concept ID {concept_id}")
            return True

        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return False

    def delete_concept(self, concept_id: int) -> bool:
        """Xóa concept"""
        try:
            self.cursor.execute('DELETE FROM concepts WHERE id = ?', (concept_id,))
            self.conn.commit()
            print(f"✓ Xóa concept ID {concept_id}")
            return True
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return False

    def search_concepts(self, query: str, top_k: int = 5, threshold: float = 0.0) -> list[dict]:
        """Tìm kiếm concepts theo ngữ nghĩa"""
        all_concepts = self.get_all_concepts()

        if not all_concepts:
            return []

        query_vector = self.model.encode(query)
        # Convert list of vectors to numpy array to avoid warning
        concept_vectors = np.array([concept['vector'] for concept in all_concepts])
        scores = util.cos_sim(query_vector, concept_vectors)[0]
        
        results = []
        for i, concept in enumerate(all_concepts):
            score_val = float(scores[i])

            if score_val >= threshold:
                results.append({
                    'id': concept['id'],
                    'name': concept['name'],
                    'parent_id': to_object_id(concept['parent_id']),
                    'score': round(score_val, 4)
                })

        results.sort(key=lambda x: x['score'], reverse=True) 
        return results[:top_k]

    def count_concepts(self) -> int:
        """Đếm số concepts"""
        self.cursor.execute('SELECT COUNT(*) FROM concepts')
        return self.cursor.fetchone()[0]

    # ==================== RELATIONS ====================

    def add_relation(self, name: str, parent_id: str) -> int | None:
        """Thêm relation mới"""
        try:
            vector = self.model.encode(name)
            vector_bytes = vector.astype(np.float32).tobytes()

            self.cursor.execute('''
                                INSERT INTO relations (name, parent_id, vector)
                                VALUES (?, ?, ?)
                                ''', (name, parent_id, vector_bytes))

            self.conn.commit()
            relation_id = self.cursor.lastrowid
            print(f"✓ Thêm relation: {name} (ID: {relation_id})")
            return relation_id

        except sqlite3.IntegrityError:
            print(f"❌ Relation '{name}' đã tồn tại!")
            

    def get_relation(self, relation_id: int) -> dict | None:
        """Lấy relation theo ID"""
        self.cursor.execute('''
                            SELECT id, name, parent_id, vector
                            FROM relations
                            WHERE id = ?
                            ''', (relation_id,))

        row = self.cursor.fetchone()
        if row:
            return {
                'id': row[0],
                'name': row[1],
                'parent_id': row[2],
                'vector': np.frombuffer(row[3], dtype=np.float32)
            }
        return None

    def get_all_relations(self) -> list[dict]:
        """Lấy tất cả relations"""
        self.cursor.execute('SELECT id, name, parent_id, vector FROM relations')
        rows = self.cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'name': row[1],
                'parent_id': row[2],
                'vector': np.frombuffer(row[3], dtype=np.float32)
            })
        return results

    def get_relations_by_parent(self, parent_id: str) -> list[dict]:
        """Lấy relations theo parent_id"""
        self.cursor.execute('''
                            SELECT id, name, parent_id, vector
                            FROM relations
                            WHERE parent_id = ?
                            ''', (parent_id,))

        rows = self.cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'name': row[1],
                'parent_id': row[2],
                'vector': np.frombuffer(row[3], dtype=np.float32)
            })
        return results

    def update_relation(self, relation_id: int, name: str = None, parent_id: str = None) -> bool:
        """Cập nhật relation"""
        try:
            if name:
                vector = self.model.encode(name)
                vector_bytes = vector.astype(np.float32).tobytes()
                self.cursor.execute('''
                                    UPDATE relations
                                    SET name   = ?,
                                        vector = ?
                                    WHERE id = ?
                                    ''', (name, vector_bytes, relation_id))

            if parent_id:
                self.cursor.execute('''
                                    UPDATE relations
                                    SET parent_id = ?
                                    WHERE id = ?
                                    ''', (parent_id, relation_id))

            self.conn.commit()
            print(f"✓ Cập nhật relation ID {relation_id}")
            return True

        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return False

    def delete_relation(self, relation_id: int) -> bool:
        """Xóa relation"""
        try:
            self.cursor.execute('DELETE FROM relations WHERE id = ?', (relation_id,))
            self.conn.commit()
            print(f"✓ Xóa relation ID {relation_id}")
            return True
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return False

    def search_relations(self, query: str, top_k: int = 5, threshold: float = 0.0) -> list[dict]:
        """Tìm kiếm relations theo ngữ nghĩa"""
        all_relations = self.get_all_relations()

        if not all_relations:
            return []

        query_vector = self.model.encode(query)
        # Convert list of vectors to numpy array to avoid warning
        relation_vectors = np.array([relation['vector'] for relation in all_relations])
        scores = util.cos_sim(query_vector, relation_vectors)[0]
        
        results = []
        for i, relation in enumerate(all_relations):
            score_val = float(scores[i])

            if score_val >= threshold:
                results.append({
                    'id': relation['id'],
                    'name': relation['name'],
                    'parent_id': relation['parent_id'],
                    'score': round(score_val, 4)
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def count_relations(self) -> int:
        """Đếm số relations"""
        self.cursor.execute('SELECT COUNT(*) FROM relations')
        return self.cursor.fetchone()[0]

    # ==================== COMBINED SEARCH ====================

    def search_all(self, query: str, top_k: int = 5) -> dict:
        """Tìm kiếm cả concepts và relations"""
        return {
            'concepts': self.search_concepts(query, top_k=top_k),
            'relations': self.search_relations(query, top_k=top_k)
        }

    # ==================== UTILITY ====================

    def get_stats(self) -> dict:
        """Lấy thống kê database"""
        return {
            'concepts_count': self.count_concepts(),
            'relations_count': self.count_relations()
        }

    def export_data(self, output_file='data.json') -> bool:
        """Export dữ liệu ra JSON"""
        try:
            data = {
                'concepts': self.get_all_concepts(),
                'relations': self.get_all_relations()
            }

            # Convert vectors thành list
            data['concepts'] = [
                {**c, 'vector': c['vector'].tolist()}
                for c in data['concepts']
            ]
            data['relations'] = [
                {**r, 'vector': r['vector'].tolist()}
                for r in data['relations']
            ]

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"✓ Export dữ liệu vào {output_file}")
            return True
        except Exception as e:
            print(f"❌ Lỗi export: {e}")
            return False

    def close(self):
        """Đóng kết nối"""
        self.conn.close()
        print("✓ Đóng kết nối database")
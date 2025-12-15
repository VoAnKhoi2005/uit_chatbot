import sqlite3
import hashlib
import json
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

class MetadataRegistry:
    def __init__(self, db_path: str = "metadata_registry.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._setup()

    def _setup(self):
        self.conn.execute('''CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            doc_title TEXT,
            so_hieu TEXT,
            article_id TEXT,
            clause_id TEXT,
            source_path TEXT,
            updated_at TEXT
        )''')
        self.conn.execute('''CREATE TABLE IF NOT EXISTS triples (
            triple_id TEXT PRIMARY KEY,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            ttl_uri TEXT,
            article_id TEXT,
            doc_id TEXT,
            source_path TEXT,
            updated_at TEXT
        )''')
        self.conn.commit()

    def upsert_chunk(self, meta: Dict[str, Any]):
        meta = meta.copy()
        meta.setdefault("updated_at", datetime.utcnow().isoformat())
        keys = ["chunk_id","doc_id","doc_title","so_hieu","article_id","clause_id","source_path","updated_at"]
        vals = [meta.get(k) for k in keys]
        self.conn.execute(f"""
            INSERT OR REPLACE INTO chunks ({','.join(keys)})
            VALUES ({','.join(['?']*len(keys))})
        """, vals)
        self.conn.commit()

    def upsert_triple(self, meta: Dict[str, Any]):
        meta = meta.copy()
        meta.setdefault("updated_at", datetime.utcnow().isoformat())
        keys = ["triple_id","subject","predicate","object","ttl_uri","article_id","doc_id","source_path","updated_at"]
        vals = [meta.get(k) for k in keys]
        self.conn.execute(f"""
            INSERT OR REPLACE INTO triples ({','.join(keys)})
            VALUES ({','.join(['?']*len(keys))})
        """, vals)
        self.conn.commit()

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,))
        row = cur.fetchone()
        if not row:
            return None
        keys = [d[0] for d in cur.description]
        return dict(zip(keys, row))

    def get_triple(self, triple_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM triples WHERE triple_id=?", (triple_id,))
        row = cur.fetchone()
        if not row:
            return None
        keys = [d[0] for d in cur.description]
        return dict(zip(keys, row))

    def get_citation_by_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT doc_id, doc_title, so_hieu FROM chunks WHERE article_id=? LIMIT 1", (article_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {"doc_id": row[0], "doc_title": row[1], "so_hieu": row[2]}

    def export_snapshot(self, path: str):
        data = {"chunks": [], "triples": []}
        for table in ["chunks", "triples"]:
            cur = self.conn.execute(f"SELECT * FROM {table}")
            keys = [d[0] for d in cur.description]
            for row in cur.fetchall():
                data[table].append(dict(zip(keys, row)))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def make_triple_id(subject, predicate, object, article_id=None, ttl_uri=None):
        s = f"{subject}|{predicate}|{object}|{article_id or ''}|{ttl_uri or ''}"
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

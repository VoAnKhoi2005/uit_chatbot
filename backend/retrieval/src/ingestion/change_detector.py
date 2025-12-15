import os
import hashlib
import sqlite3
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

@dataclass
class FileChange:
    path: str
    change_type: str  # 'ADDED', 'MODIFIED', 'DELETED'
    old: Optional[dict] = None
    new: Optional[dict] = None

class ChangeDetector:
    def __init__(self, db_path: str = "change_state.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._setup()
        self._scan_result = []

    def _setup(self):
        self.conn.execute('''CREATE TABLE IF NOT EXISTS file_state (
            path TEXT PRIMARY KEY,
            sha256 TEXT,
            size INTEGER,
            mtime REAL,
            group_name TEXT,
            updated_at TEXT
        )''')
        self.conn.commit()

    def _file_fingerprint(self, path: str) -> Dict:
        p = Path(path)
        if not p.exists():
            return {}
        with open(path, "rb") as f:
            content = f.read()
        return {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": p.stat().st_size,
            "mtime": p.stat().st_mtime,
        }

    def scan(self, paths: List[str], group: str = "docs") -> List[FileChange]:
        # Load old state
        cur = self.conn.execute("SELECT path, sha256, size, mtime FROM file_state WHERE group_name=?", (group,))
        old_state = {row[0]: {"sha256": row[1], "size": row[2], "mtime": row[3]} for row in cur.fetchall()}
        new_state = {}
        changes = []
        for path in paths:
            if not os.path.exists(path):
                continue
            if os.path.isdir(path):
                files = [str(p) for p in Path(path).rglob("*") if p.is_file()]
            else:
                files = [path]
            for f in files:
                fp = self._file_fingerprint(f)
                new_state[f] = fp
                if f not in old_state:
                    changes.append(FileChange(f, "ADDED", None, fp))
                elif fp["sha256"] != old_state[f]["sha256"]:
                    changes.append(FileChange(f, "MODIFIED", old_state[f], fp))
        for f in old_state:
            if f not in new_state:
                changes.append(FileChange(f, "DELETED", old_state[f], None))
        self._scan_result = changes
        return changes

    def has_changes(self, group: str = "docs") -> bool:
        return bool(self._scan_result)

    def commit_scan(self, paths: List[str], group: str = "docs"):
        # Save new state
        cur = self.conn.execute("SELECT path FROM file_state WHERE group_name=?", (group,))
        old_paths = set(row[0] for row in cur.fetchall())
        new_paths = set()
        for path in paths:
            if not os.path.exists(path):
                continue
            if os.path.isdir(path):
                files = [str(p) for p in Path(path).rglob("*") if p.is_file()]
            else:
                files = [path]
            for f in files:
                fp = self._file_fingerprint(f)
                self.conn.execute("""
                    INSERT OR REPLACE INTO file_state (path, sha256, size, mtime, group_name, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (f, fp.get("sha256"), fp.get("size"), fp.get("mtime"), group, datetime.utcnow().isoformat()))
                new_paths.add(f)
        # Remove deleted
        for f in old_paths - new_paths:
            self.conn.execute("DELETE FROM file_state WHERE path=? AND group_name=?", (f, group))
        self.conn.commit()

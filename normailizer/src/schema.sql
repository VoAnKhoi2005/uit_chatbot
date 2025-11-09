PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
  id            TEXT PRIMARY KEY,
  so_hieu       TEXT NOT NULL,
  title         TEXT NOT NULL,
  issued_date   TEXT,
  effective_date TEXT,
  unit          TEXT,
  status        TEXT DEFAULT 'effective',
  checksum      TEXT,
  created_at    TEXT,
  updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS items (
  id          TEXT PRIMARY KEY,
  doc_id      TEXT NOT NULL,
  parent_id   TEXT,
  level       TEXT NOT NULL,
  title       TEXT,
  heading     TEXT,
  content     TEXT,
  ordinal     REAL,
  path        TEXT,
  created_at  TEXT,
  updated_at  TEXT,
  FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE,
  FOREIGN KEY(parent_id) REFERENCES items(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sources (
  id          TEXT PRIMARY KEY,
  doc_id      TEXT NOT NULL,
  file_name   TEXT NOT NULL,
  file_path   TEXT,
  pages       INTEGER,
  imported_at TEXT,
  FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
  title, heading, content, doc_id, path, level, item_id UNINDEXED,
  tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
  INSERT INTO items_fts(rowid, title, heading, content, doc_id, path, level, item_id)
  VALUES (new.rowid, new.title, new.heading, new.content, new.doc_id, new.path, new.level, new.id);
END;
CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
  INSERT INTO items_fts(items_fts, rowid, title) VALUES('delete', old.rowid, old.title);
END;
CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE ON items BEGIN
  INSERT INTO items_fts(items_fts, rowid, title, heading, content, doc_id, path, level, item_id)
  VALUES('delete', old.rowid, old.title, old.heading, old.content, old.doc_id, old.path, old.level, old.id);
  INSERT INTO items_fts(rowid, title, heading, content, doc_id, path, level, item_id)
  VALUES (new.rowid, new.title, new.heading, new.content, new.doc_id, new.path, new.level, new.id);
END;

CREATE VIEW IF NOT EXISTS v_docs AS
SELECT d.*, COUNT(i.id) AS n_items
FROM documents d LEFT JOIN items i ON i.doc_id = d.id
GROUP BY d.id;

CREATE VIEW IF NOT EXISTS v_toc AS
SELECT d.so_hieu, d.title AS doc_title, i.id AS item_id, i.level, i.title, i.heading, i.ordinal, i.path, i.parent_id
FROM items i JOIN documents d ON d.id = i.doc_id
ORDER BY d.so_hieu, i.path, i.ordinal;

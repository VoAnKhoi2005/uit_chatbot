import sqlite3, uuid, hashlib, datetime
from pathlib import Path

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_schema(db: Path):
    con = sqlite3.connect(str(db), timeout=60.0, isolation_level=None)
    cur = con.cursor()
    cur.executescript(Path("schema.sql").read_text(encoding="utf-8"))
    cur.execute("PRAGMA wal_checkpoint(FULL);")
    cur.execute("PRAGMA journal_mode=DELETE;")
    con.close()

def insert_document(db: Path, pdf_path: Path, so_hieu: str, title: str, chapters):
    con = sqlite3.connect(str(db), timeout=60.0, isolation_level=None)
    cur = con.cursor()
    cur.execute("PRAGMA busy_timeout=20000;")

    checksum = sha256_file(pdf_path)
    if cur.execute("SELECT id FROM documents WHERE checksum=?", (checksum,)).fetchone():
        con.close()
        return {"status":"skipped (duplicate content)", "checksum": checksum}

    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    doc_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO documents(id, so_hieu, title, issued_date, unit, status, checksum, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (doc_id, so_hieu, title, None, "UIT", "effective", checksum, now, now)
    )

    ordinal = 0.0
    def add_item(level, title, heading, content, parent_id, path):
        nonlocal ordinal
        ordinal += 1.0
        iid = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO items(id, doc_id, parent_id, level, title, heading, content, ordinal, path, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (iid, doc_id, parent_id, level, title, heading, content, ordinal, path, now, now)
        )
        return iid

    for ch in chapters:
        ch_no = ch["no"] if ch["no"] else 0
        ch_title = f"Chương {ch_no}" if ch_no else "Chương"
        ch_id = add_item("chuong", ch_title, ch["heading"], "", None, f"{ch_no}")
        for ar in ch["articles"]:
            ar_id = add_item("dieu", f"Điều {ar['no']}", ar["heading"], ar["content"], ch_id, f"{ch_no}/{ar['no']}")
            from parse_structure import split_khoan_diem
            for kno, kbody, dparts in split_khoan_diem(ar["content"]):
                k_id = add_item("khoan", f"Khoản {kno}", "", kbody, ar_id, f"{ch_no}/{ar['no']}/{kno}")
                for chx, seg in dparts:
                    add_item("diem", f"Điểm {chx})", "", seg, k_id, f"{ch_no}/{ar['no']}/{kno}/{chx}")

    src_id = str(uuid.uuid4())
    cur.execute("INSERT INTO sources(id, doc_id, file_name, file_path, pages, imported_at) VALUES (?,?,?,?,?,?)",
                (src_id, doc_id, pdf_path.name, str(pdf_path), -1, now))
    cur.execute("PRAGMA wal_checkpoint(FULL);")
    cur.execute("PRAGMA journal_mode=DELETE;")
    con.close()
    return {"status":"imported", "doc_id": doc_id, "checksum": checksum}

import sqlite3, argparse, datetime, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Create UIT law DB schema")
    ap.add_argument("--db", required=True, help="Path to SQLite DB (will be created if not exists)")
    args = ap.parse_args()

    db = Path(args.db)
    schema = Path("schema.sql").read_text(encoding="utf-8")

    con = sqlite3.connect(str(db), timeout=60.0, isolation_level=None)
    cur = con.cursor()
    cur.execute("PRAGMA busy_timeout=20000;")
    cur.executescript(schema)
    cur.execute("PRAGMA wal_checkpoint(FULL);")
    cur.execute("PRAGMA journal_mode=DELETE;")
    con.close()
    print(f"Initialized schema at {db}")

if __name__ == "__main__":
    main()

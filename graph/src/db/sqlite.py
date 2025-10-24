import random
import sqlite3

def init_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return conn, cursor

def extract_from_sqlite(cursor, id: int, include_parent=False):
    cursor.execute("SELECT * FROM laws WHERE id = ?", (id,))
    rows = cursor.fetchall()

    columns = [col[0] for col in cursor.description]
    results = [dict(zip(columns, row)) for row in rows]

    if include_parent:
        all_nodes = []
        for r in results:
            current = r
            while current['parent_id'] is not None:
                cursor.execute("SELECT * FROM laws WHERE id = ?", (current['parent_id'],))
                parent = cursor.fetchone()
                if parent is None:
                    break
                parent_dict = dict(zip(columns, parent))
                all_nodes.append(parent_dict)
                current = parent_dict
        results.extend(all_nodes)

    return results[::-1]

def extract_random_diem_from_sqlite(cursor, include_parent=True):
    cursor.execute("SELECT * FROM laws WHERE title LIKE '%Điểm%'")
    rows = cursor.fetchall()
    if not rows:
        return []

    columns = [col[0] for col in cursor.description]
    row = random.choice(rows)
    result = dict(zip(columns, row))

    results = [result]

    if include_parent:
        current = result
        while current['parent_id'] is not None:
            cursor.execute("SELECT * FROM laws WHERE id = ?", (current['parent_id'],))
            parent = cursor.fetchone()
            if parent is None:
                break
            parent_dict = dict(zip(columns, parent))
            results.append(parent_dict)
            current = parent_dict

    return results[::-1]

def extract_random_from_sqlite(cursor):
    cursor.execute("SELECT * FROM laws ORDER BY RANDOM() LIMIT 1")
    row = cursor.fetchone()
    if not row:
        return None

    columns = [col[0] for col in cursor.description]
    return dict(zip(columns, row))

def extract_all_from_sqlite(cursor, table_name):
    query = f"SELECT * FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()
    if not rows:
        return None
    return rows

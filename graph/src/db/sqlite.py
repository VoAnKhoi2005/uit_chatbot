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

def extract_random_leaf_from_sqlite(cursor, table_name: str, include_parent=True):
    # Find all leaf nodes (rows whose id is not referenced as a parent_id)
    cursor.execute(f"""
        SELECT * FROM {table_name}
        WHERE id NOT IN (
            SELECT DISTINCT parent_id FROM {table_name}
            WHERE parent_id IS NOT NULL
        )
    """)
    leaves = cursor.fetchall()
    if not leaves:
        return []

    columns = [col[0] for col in cursor.description]

    # Randomly select one leaf
    leaf_row = random.choice(leaves)
    result = dict(zip(columns, leaf_row))

    results = [result]

    # Optionally include the parent chain
    if include_parent:
        current = result
        while current.get('parent_id') is not None:
            cursor.execute(f"SELECT * FROM {table_name} WHERE id = ?", (current['parent_id'],))
            parent = cursor.fetchone()
            if parent is None:
                break
            parent_dict = dict(zip(columns, parent))
            results.append(parent_dict)
            current = parent_dict

    return results[::-1]  # From root to leaf


def extract_all_laws_from_sqlite(cursor):
    cursor.execute("SELECT * FROM laws")
    rows = cursor.fetchall()

    columns = [col[0] for col in cursor.description]
    results = [dict(zip(columns, row)) for row in rows]

    return results

def extract_all_from_sqlite(cursor, table_name):
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    if not rows:
        return []

    # Get column names from the cursor description
    columns = [desc[0] for desc in cursor.description]

    # Convert each row (tuple) to a dict
    return [dict(zip(columns, row)) for row in rows]

def extract_random_rows(cursor, table_name: str, limit: int = 1):
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()
    if not rows:
        return []

    columns = [col[0] for col in cursor.description]

    # Choose random rows without replacement
    limit = min(limit, len(rows))
    selected = random.sample(rows, limit)

    return [dict(zip(columns, r)) for r in selected]

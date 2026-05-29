import sqlite3

db_path = "/workspace/code/project/solution/data/core/udahub.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute("PRAGMA table_info(knowledge);")
for row in cur.fetchall():
    print(dict(row))

print("\nSample rows:")
cur.execute("SELECT * FROM knowledge LIMIT 3;")
for row in cur.fetchall():
    print(dict(row))

conn.close()